"""AIRS-Bench -> Harbor task directories.

Upstream (``third_party/airs-bench``) publishes task *definitions* only: a prompt
(``project_description.md``), a data preparer (``prepare.py``), a gold-label preparer
(``evaluate_prepare.py``), a grader (``evaluate.py``) and a task card
(``metadata.yaml``). It publishes no runner, and the public ``aira-dojo`` cannot run
AIRS at all (harness_facts/airs_bench.md section 0). So we supply the runner, in
Harbor's task-directory format, and never reimplement the scoring: the generated
``tests/test.sh`` shells out to the upstream ``evaluate.py`` and only *normalises* the
number it prints, which is the one piece of the protocol upstream ships no code for.

Three things the generated directory deliberately does **not** contain:

* **data** -- ``prepare.py`` runs once on the host (:func:`stage`), the result is
  mounted read-only at ``/app/data``. Baking it into the image would rebuild gigabytes
  per task and per rebuild.
* **gold labels, at agent time** -- ``tests/eval_data/test_with_labels`` is produced on
  the host by the upstream ``evaluate_prepare.py`` and lands in ``tests/``, which Harbor
  uploads into the container *only for the verifier phase*. Mounting the raw HF dataset
  instead (what a literal reading of the upstream contract implies) would put the test
  labels inside the agent's sandbox for the whole run. ``tests/eval_data/`` is
  gitignored; :func:`stage` (re)creates it.
* **anchors as prose** -- ``s_min``/``sota``/``s_opt`` are written to
  ``tests/anchors.json`` *and* to ``task.toml`` from the same read of ``metadata.yaml``.
  AutoLab's habit of hard-coding anchors a second time inside ``test.sh`` is exactly how
  its flux task ended up scoring against a stale reference.

CLI::

    python -m awm.adapters.airs generate --all [--profile smoke]
    python -m awm.adapters.airs stage --task TextualClassificationSickAccuracy
    python -m awm.adapters.airs plan
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from awm import scope
from awm.paths import REPO_ROOT, data_root, ensure, tasks_dir

UPSTREAM_ROOT = REPO_ROOT / "third_party" / "airs-bench"
UPSTREAM_TASKS = UPSTREAM_ROOT / "airsbench" / "tasks" / "rad"

#: Upstream is CC BY-NC 4.0; every generated directory says so.
UPSTREAM_LICENSE = "CC BY-NC 4.0"
UPSTREAM_URL = "https://github.com/facebookresearch/airs-bench"



# --------------------------------------------------------------------- normalisation
# The block between the markers is copied verbatim into the generated tests/score.py,
# so the formula the verifier runs and the formula our tests check are one text.
# --8<-- normalise start
def phi(s: float, s_opt: float) -> float:
    """The paper's log transform, ``phi(s) = -log10(|s - s_opt|)``.

    A submission that hits the optimum exactly has ``phi = +inf``; callers turn that
    into a normalised score of ``+inf`` and then clip, rather than dividing by zero.
    """
    d = abs(float(s) - float(s_opt))
    if d == 0.0:
        return math.inf
    return -math.log10(d)


def normalised_score(raw: float, s_min: float, s_sota: float, s_opt: float) -> float:
    """AIRS-Bench Eq. 2-3: ``(phi(s) - phi(s_min)) / (phi(s_sota) - phi(s_min))``.

    Unclipped on purpose -- above 1 means the run beat the published SOTA, below 0 means
    it was worse than the worst score Meta observed, and both facts are worth keeping.
    ``reward`` is the clipped copy.
    """
    lo = phi(s_min, s_opt)
    hi = phi(s_sota, s_opt)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi == lo:
        raise ValueError(f"degenerate anchors: s_min={s_min} s_sota={s_sota} s_opt={s_opt}")
    top = phi(raw, s_opt)
    if math.isinf(top):
        return math.inf
    return (top - lo) / (hi - lo)


def reward_from(raw: float, s_min: float, s_sota: float, s_opt: float) -> float:
    """The normalised score clipped to Harbor's [0, 1] reward convention."""
    ns = normalised_score(raw, s_min, s_sota, s_opt)
    return max(0.0, min(1.0, ns))
# --8<-- normalise end


def _normalise_block() -> str:
    """The marked source block above, for embedding in the generated score.py."""
    src = Path(__file__).read_text(encoding="utf-8")
    start = src.index("# --8<-- normalise start\n") + len("# --8<-- normalise start\n")
    end = src.index("# --8<-- normalise end")
    return src[start:end].rstrip() + "\n"


# ------------------------------------------------------------------------- data paths


def assets_root() -> Path:
    return data_root() / "assets" / "airs"


def raw_root() -> Path:
    """Where the upstream download script's ``save_to_disk`` trees live."""
    return assets_root() / "raw"


def prepared_root() -> Path:
    """Per-task agent-visible data: the output of the upstream ``prepare.py``."""
    return assets_root() / "prepared"


def out_root() -> Path:
    return tasks_dir() / "airs"


# --------------------------------------------------------------------------- metadata


def upstream_dir(task: str) -> Path:
    d = UPSTREAM_TASKS / task
    if not d.is_dir():
        raise FileNotFoundError(
            f"no upstream task at {d}; is the airs-bench submodule checked out?"
        )
    return d


def read_metadata(task: str) -> dict[str, Any]:
    return yaml.safe_load((upstream_dir(task) / "metadata.yaml").read_text(encoding="utf-8"))


def upstream_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(UPSTREAM_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return out.stdout.strip()
    except Exception:  # a tarball checkout has no .git; the SHA is provenance, not logic
        return "unknown"


@dataclass(frozen=True)
class Anchors:
    """Everything the verifier needs to turn one raw metric into a reward."""

    task: str
    metric: str
    direction: str
    s_sota: float
    s_min: float
    s_opt: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "metric": self.metric,
            "direction": self.direction,
            "s_sota": self.s_sota,
            "s_min": self.s_min,
            "s_opt": self.s_opt,
        }


def anchors(task: str) -> Anchors:
    """The scoring anchors, read straight out of upstream ``metadata.yaml``."""
    meta = read_metadata(task)
    li = meta["logging_info"]
    return Anchors(
        task=task,
        metric=li["metric"],
        direction="lower_is_better" if meta["metric_lower_is_better"] else "higher_is_better",
        s_sota=float(li["sota"][0]["sota_score"]),
        s_min=float(li["estimated_worst_score"]),
        s_opt=float(li["optimal_score"]),
    )


def dataset_of(task: str) -> tuple[str, str]:
    """``(hf_dataset_id, config)`` -- the pair the upstream download script keys on."""
    li = read_metadata(task)["logging_info"]
    return li["dataset"], li["config"]


# --------------------------------------------------------------------------- profiles


@dataclass(frozen=True)
class Profile:
    """Resources and timeouts. The official protocol is unaffordable for a smoke run."""

    name: str
    cpus: int | None
    memory_mb: int | None
    storage_mb: int | None
    gpus: int
    agent_timeout_sec: int
    verifier_timeout_sec: int
    build_timeout_sec: int
    #: pip index for torch; CPU wheels keep the smoke image ~2 GB instead of ~8 GB.
    torch_index: str | None
    base_image: str


#: What section 7 of the design spec calls the official AIRS protocol, adjusted only
#: where the machine differs (Ada, not H200 -- ``gpu_types`` is therefore left unset so
#: the local Docker environment does not reject the card).
OFFICIAL = Profile(
    name="official",
    cpus=24,
    memory_mb=200 * 1024,
    storage_mb=200 * 1024,
    gpus=1,
    agent_timeout_sec=24 * 3600,
    verifier_timeout_sec=3600,
    build_timeout_sec=3600,
    torch_index=None,
    base_image="pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime",
)

#: A few minutes of compute: enough to prove build -> agent -> verify -> reward.
SMOKE = Profile(
    name="smoke",
    cpus=8,
    memory_mb=16 * 1024,
    storage_mb=20 * 1024,
    gpus=0,
    agent_timeout_sec=1800,
    verifier_timeout_sec=1800,
    build_timeout_sec=2400,
    torch_index="https://download.pytorch.org/whl/cpu",
    base_image="python:3.12-slim",
)

PROFILES = {p.name: p for p in (OFFICIAL, SMOKE)}


# ------------------------------------------------------------------------- generation


def _pip_requirements(meta: dict[str, Any]) -> list[str]:
    """The union of the agent-container and evaluate-container pins.

    Harbor gives a task one image, so both lists must be installed in it. Upstream keeps
    them apart because aira-dojo used a separate grading container.
    """
    seen: list[str] = []
    for key in ("container_python_requirements", "evaluate_container_python_requirements"):
        for req in meta.get(key) or []:
            if req not in seen:
                seen.append(req)
    return seen


def _needs_torch(reqs: Iterable[str], task: str) -> bool:
    joined = " ".join(reqs)
    if "torch" in joined:
        return True
    # Every upstream evaluate.py imports torch whether or not it uses it.
    return "import torch" in (upstream_dir(task) / "evaluate.py").read_text(encoding="utf-8")


def _dockerfile(task: str, meta: dict[str, Any], prof: Profile) -> str:
    reqs = _pip_requirements(meta)
    lines = [
        f"# Generated by awm/adapters/airs.py for {task} (profile: {prof.name}).",
        "# pip pins come from the upstream metadata.yaml; nothing is downloaded at run",
        "# time (the task runs with no network), and no dataset is baked in.",
        f"FROM {prof.base_image}",
        "",
        "ENV DEBIAN_FRONTEND=noninteractive \\",
        "    PYTHONUNBUFFERED=1 \\",
        "    PIP_DISABLE_PIP_VERSION_CHECK=1 \\",
        "    HF_HUB_OFFLINE=1 \\",
        "    HF_DATASETS_OFFLINE=1",
        "",
        "RUN apt-get update && apt-get install -y --no-install-recommends \\",
        "        bash ca-certificates curl git procps tmux \\",
        "    && rm -rf /var/lib/apt/lists/*",
        "",
    ]
    if prof.torch_index and _needs_torch(reqs, task):
        lines += [
            "# CPU-only torch: the smoke profile has no GPU and the CUDA wheels are 6 GB.",
            f"RUN pip install --no-cache-dir --index-url {prof.torch_index} torch",
            "",
        ]
    if reqs:
        joined = " \\\n        ".join(f'"{r}"' for r in reqs)
        lines += [f"RUN pip install --no-cache-dir \\\n        {joined}", ""]
    lines += [
        "WORKDIR /app",
        "",
        "# /app/data is a read-only bind mount (see docker-compose.yaml); the directory",
        "# must exist in the image so compose does not create it root-owned and empty.",
        "RUN mkdir -p /app/data",
        "",
    ]
    return "\n".join(lines)


def _compose(task: str, prof: Profile) -> str:
    """Bind the host-prepared agent data read-only, and reserve GPUs when asked.

    ``AWM_AIRS_PREPARED`` must be an ABSOLUTE host path and has no default: compose
    resolves a relative volume source against the file's own directory, so
    ``./data/...`` silently binds a non-existent path and the task scores 0 with a
    missing submission rather than failing loudly. Baking in the absolute path this
    machine happens to resolve ``data/`` to would commit one developer's layout, so
    the variable is required instead, and `awm run` fills it in from awm.paths.
    """
    gpu = ""
    if prof.gpus:
        gpu = (
            "    deploy:\n"
            "      resources:\n"
            "        reservations:\n"
            "          devices:\n"
            "            - driver: nvidia\n"
            f"              count: {prof.gpus}\n"
            "              capabilities: [gpu]\n"
        )
    return (
        f"# Generated by awm/adapters/airs.py for {task}.\n"
        "# The agent-visible data is prepared once on the host by\n"
        "#   python -m awm.adapters.airs stage --task <task>\n"
        "# and mounted read-only. Gold labels are NOT here: they ride in tests/, which\n"
        "# Harbor uploads only for the verifier phase.\n"
        "#\n"
        "# AWM_AIRS_PREPARED must be an ABSOLUTE host path -- compose resolves a relative\n"
        "# volume source against this file's directory, which would bind nothing and let\n"
        "# the task score 0 on a missing submission instead of failing. `awm run` sets it;\n"
        "# calling harbor directly needs --env-file or an exported value.\n"
        "services:\n"
        "  main:\n"
        "    volumes:\n"
        f"      - ${{AWM_AIRS_PREPARED:?set to the absolute path of data/assets/airs/prepared}}"
        f"/{task}:/app/data:ro\n"
        f"{gpu}"
    )


def _instruction(task: str, prof: Profile) -> str:
    """``project_description.md`` verbatim, plus a section on where things live.

    Upstream's prompt is harness-agnostic: it says ``./data/train`` and
    ``submission.csv`` and nothing about a container. The addendum is the adapter's, and
    is clearly marked as such so it is obvious what the agent saw that Meta's did not.
    """
    body = (upstream_dir(task) / "project_description.md").read_text(encoding="utf-8")
    hours = prof.agent_timeout_sec / 3600
    gpu = f"{prof.gpus} GPU" if prof.gpus else "no GPU"
    return (
        body.rstrip()
        + "\n\n---\n\n## Environment (added by the harness, not part of the upstream task)\n\n"
        "| Item | Value |\n|---|---|\n"
        "| Working directory | `/app` (you are root) |\n"
        "| Data | `/app/data/{train,test,...}`, read-only. `./data/...` resolves here. |\n"
        "| Submission | write `/app/submission.csv` before your time is up |\n"
        f"| Resources | {prof.cpus} CPUs, {(prof.memory_mb or 0) // 1024} GB RAM, {gpu} |\n"
        f"| Time budget | {hours:g} h |\n"
        "| Network | none |\n\n"
        "Only `/app/submission.csv` is graded. It is scored against a held-out copy of\n"
        "the test labels that is not present in this container during your run.\n"
    )


_TEST_SH = r"""#!/usr/bin/env bash
# Generated by awm/adapters/airs.py for {task}. Harbor uploads this directory to /tests
# and runs this script only after the agent phase has ended.
#
# The grader itself is upstream's: tests/upstream/evaluate.py, unmodified. This script
# only reproduces the second half of upstream's evaluate_prepare.py (put the gold labels
# and the submission where evaluate.py looks for them) and then normalises the number.
set -uo pipefail

LOGDIR=/logs/verifier
mkdir -p "$LOGDIR"
EVAL_ROOT=/tmp/airs_eval
rm -rf "$EVAL_ROOT"
mkdir -p "$EVAL_ROOT/data"

score() {{ python3 /tests/score.py "$@"; }}

# Some graders read the agent-visible splits too (e.g. the time-series tasks need the
# history), so link them in alongside the gold labels.
if [ -d /app/data ]; then
  for entry in /app/data/*; do
    [ -e "$entry" ] || continue
    ln -s "$entry" "$EVAL_ROOT/data/$(basename "$entry")"
  done
fi

if [ ! -d /tests/eval_data/test_with_labels ]; then
  score --invalid "gold labels missing: run 'python -m awm.adapters.airs stage --task {task}' on the host"
  exit 0
fi
rm -rf "$EVAL_ROOT/data/test_with_labels"
cp -r /tests/eval_data/test_with_labels "$EVAL_ROOT/data/test_with_labels"

if [ ! -f /app/submission.csv ]; then
  score --invalid "no /app/submission.csv"
  exit 0
fi
cp /app/submission.csv "$EVAL_ROOT/data/submission.csv"

cd "$EVAL_ROOT"
python3 /tests/upstream/evaluate.py --submission-file ./data/submission.csv \
  > "$LOGDIR/evaluate_stdout.txt" 2> "$LOGDIR/evaluate_stderr.txt"
rc=$?
tail -n 40 "$LOGDIR/evaluate_stdout.txt" "$LOGDIR/evaluate_stderr.txt"

score --stdout "$LOGDIR/evaluate_stdout.txt" --exit-code "$rc"
"""


_SCORE_PY = r'''#!/usr/bin/env python3
"""Generated by awm/adapters/airs.py -- turn upstream evaluate.py's stdout into a reward.

Upstream ships no normalisation code (harness_facts/airs_bench.md section 4): the paper
defines the formula, the repo does not implement it. The formula below is a verbatim
copy of awm/adapters/airs.py's, so the two cannot drift.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

ANCHORS = json.loads(Path(__file__).with_name("anchors.json").read_text())

#: Harbor parses every value in reward.json as float|int (VerifierResult.rewards), so a
#: string or a null anywhere in it fails the whole trial with a ValidationError. Numbers
#: go here; everything a human needs goes in the sidecar next to it.
REWARD_PATH = Path("/logs/verifier/reward.json")
DETAIL_PATH = Path("/logs/verifier/airs_score.json")

# The marker upstream's evaluate.py prints, and the regex upstream's own
# test_task_folder.py uses to parse what follows it.
RESULT_RE = re.compile(r"--- EVALUATION RESULT ---\s*(\{{[\s\S]*?\}})")


{normalise}

def write(reward: float, ns: float | None, raw: float | None, *, valid: bool,
          error: str | None = None, extra: dict | None = None) -> None:
    # Harbor turns every key here into a reported metric and averages it across trials,
    # so constants like the anchors would clutter the leaderboard. They live in the
    # sidecar and in task.toml.
    numeric = {{
        "reward": reward,
        "valid_submission": int(valid),
    }}
    if ns is not None and math.isfinite(ns):
        numeric["normalized_score"] = ns
    if raw is not None:
        numeric["raw_metric"] = raw
    REWARD_PATH.parent.mkdir(parents=True, exist_ok=True)
    REWARD_PATH.write_text(json.dumps(numeric, indent=2) + "\n")

    detail = dict(ANCHORS)
    detail.update(numeric)
    detail["valid_submission"] = valid
    detail["error"] = error
    detail["normalized_score_raw"] = ns
    detail.update(extra or {{}})
    DETAIL_PATH.write_text(json.dumps(detail, indent=2) + "\n")
    print(json.dumps(detail, indent=2))


def invalid(reason: str) -> None:
    write(0.0, 0.0, None, valid=False, error=reason)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stdout", type=Path)
    ap.add_argument("--exit-code", type=int, default=0)
    ap.add_argument("--invalid")
    args = ap.parse_args()

    if args.invalid:
        invalid(args.invalid)
        return
    if args.exit_code != 0:
        invalid(f"evaluate.py exited {{args.exit_code}}")
        return

    text = args.stdout.read_text(errors="replace") if args.stdout else ""
    m = RESULT_RE.search(text)
    if m is None:
        invalid("no '--- EVALUATION RESULT ---' block in evaluate.py stdout")
        return
    try:
        result = json.loads(m.group(1))
    except ValueError as e:
        invalid(f"unparseable evaluation result: {{e}}")
        return

    name = ANCHORS["metric"]
    if name in result:
        raw = result[name]
    elif len(result) == 1:
        # An upstream grader may spell the key differently from logging_info.metric;
        # a single-key result leaves no ambiguity about which number is meant.
        raw = next(iter(result.values()))
    else:
        invalid(f"metric {{name!r}} not in evaluation result {{sorted(result)}}")
        return
    if raw is None or (isinstance(raw, float) and math.isnan(raw)):
        invalid(f"metric {{name}} is {{raw!r}}")
        return

    raw = float(raw)
    ns = normalised_score(raw, ANCHORS["s_min"], ANCHORS["s_sota"], ANCHORS["s_opt"])
    write(max(0.0, min(1.0, ns)), ns, raw, valid=True, extra={{"all_metrics": result}})


if __name__ == "__main__":
    main()
'''


_SOLVE_SH = r"""#!/usr/bin/env bash
# Reference solution for {task}, run by Harbor's `oracle` agent from /solution.
# {note}
set -euo pipefail
cd /app
python3 /solution/reference.py
test -f /app/submission.csv
wc -l /app/submission.csv
"""

_NO_REFERENCE = r"""#!/usr/bin/env bash
# No reference solution is registered for {task} yet.
# `harbor run -a oracle` on this task will fail on purpose rather than score a
# fabricated submission: see REFERENCE_SOLUTIONS in awm/adapters/airs.py.
set -euo pipefail
echo "no reference solution for {task}" >&2
exit 1
"""


#: Reference solutions, keyed by task. They exist to prove the grader responds to
#: quality, not to approach SOTA -- a run whose reward moves when the submission gets
#: better is the only evidence that the scoring path is real.
REFERENCE_SOLUTIONS: dict[str, str] = {}

REFERENCE_SOLUTIONS["TextualClassificationSickAccuracy"] = r'''#!/usr/bin/env python3
"""TF-IDF + logistic regression on the SICK sentence pairs.

Nowhere near the 0.905 SOTA (a fine-tuned RoBERTa); the point is a submission that is
clearly better than any constant one, so a reward that moves proves the grader works.
Runs in well under a minute on CPU.
"""

import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train = load_from_disk("./data/train")
test = load_from_disk("./data/test")

a_tr = [s.lower() for s in train["sentence_A"]]
b_tr = [s.lower() for s in train["sentence_B"]]
a_te = [s.lower() for s in test["sentence_A"]]
b_te = [s.lower() for s in test["sentence_B"]]

word = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
word.fit(a_tr + b_tr)


def feats(a, b):
    """Entailment is about the *difference* between the two sentences, so the pairwise
    features (elementwise product and absolute difference) carry most of the signal."""
    A, B = word.transform(a), word.transform(b)
    prod = A.multiply(B)
    diff = abs(A - B)
    return sparse.hstack([A, B, prod, diff]).tocsr()


X = feats(a_tr, b_tr)
y = np.asarray(train["label"])
clf = LogisticRegression(max_iter=2000, C=4.0)
clf.fit(X, y)
print("train accuracy:", clf.score(X, y))

pred = clf.predict(feats(a_te, b_te))
pd.DataFrame({"label": pred}).to_csv("/app/submission.csv", index=False)
print("wrote /app/submission.csv", pred.shape, np.bincount(pred))
'''


def _task_toml(task: str, prof: Profile, entry: scope.Entry | None) -> str:
    meta = read_metadata(task)
    li = meta["logging_info"]
    a = anchors(task)
    sota = li["sota"][0]

    def q(v: Any) -> str:
        return json.dumps(v)

    lines = [
        f"# Generated by awm/adapters/airs.py from {UPSTREAM_URL}",
        f"# @ {upstream_commit()} : airsbench/tasks/rad/{task}/metadata.yaml",
        f"# Regenerate: python -m awm.adapters.airs generate --task {task} --profile {prof.name}",
        '# Upstream task definition is CC BY-NC 4.0. Do not edit by hand.',
        'version = "1.4"',
        "",
        "[metadata]",
        'benchmark = "airs"',
        f'task_id = "airs/{task}"',
        'author = "AIRS-Bench (task) / agentic-world-model adapter (harness)"',
        f"license = {q(UPSTREAM_LICENSE)}",
        f"upstream_repo = {q(UPSTREAM_URL)}",
        f"upstream_commit = {q(upstream_commit())}",
        f"upstream_task = {q(f'airsbench/tasks/rad/{task}')}",
        f"profile = {q(prof.name)}",
        f"category = {q(li['category'])}",
        f"research_problem = {q(li['research_problem'])}",
        f"dataset = {q(li['dataset'])}",
        f"dataset_config = {q(str(li['config']))}",
        f"train_split = {q(str(li['train_split']))}",
        f"test_split = {q(str(li['test_split']))}",
        f"test_shape = {q(str(li['shape']))}",
        f"scope_gpu_type = {q(str((entry.resources if entry else {}).get('gpu_type', '')))}",
        f"official_budget_h = {(entry.budget.get('official_h') if entry else 0) or 0}",
        "tags = [\"airs-bench\", \"track-optimize\", "
        f"{q(li['category'])}, {q(li['metric'])}]",
        "",
        "# Read by tests/score.py from tests/anchors.json, which is generated in the",
        "# same pass from the same metadata.yaml. Repeated here as documentation.",
        "[metadata.metric]",
        f"name = {q(a.metric)}",
        f"direction = {q(a.direction)}",
        f"sota_score = {a.s_sota!r}",
        f"sota_paper = {q(sota['sota_paper_title'])}",
        f"s_min = {a.s_min!r}",
        f"s_opt = {a.s_opt!r}",
        'normalisation = "AIRS-Bench paper Eq. 2-3: NS = (phi(s) - phi(s_min)) / '
        '(phi(s_sota) - phi(s_min)), phi(s) = -log10(|s - s_opt|); reward = clip(NS, 0, 1)"',
        "",
        "[agent]",
        f"timeout_sec = {prof.agent_timeout_sec}",
        "",
        "[verifier]",
        f"timeout_sec = {prof.verifier_timeout_sec}",
        "",
        "[environment]",
        f"build_timeout_sec = {prof.build_timeout_sec}",
        f"cpus = {prof.cpus}",
        f"memory_mb = {prof.memory_mb}",
        f"storage_mb = {prof.storage_mb}",
        f"gpus = {prof.gpus}",
        'network_mode = "no-network"',
        "",
    ]
    return "\n".join(lines)


def generate(task: str, profile: str = "smoke", out_dir: Path | None = None) -> Path:
    """Write one Harbor task directory. Returns its path. Idempotent."""
    prof = PROFILES[profile]
    meta = read_metadata(task)
    src = upstream_dir(task)
    dest = Path(out_dir) if out_dir else out_root() / task

    entry = None
    try:
        entry = scope.get(f"airs/{task}")
    except KeyError:
        pass

    for sub in ("environment", "tests/upstream", "solution"):
        ensure(dest / sub)

    (dest / "task.toml").write_text(_task_toml(task, prof, entry), encoding="utf-8")
    (dest / "instruction.md").write_text(_instruction(task, prof), encoding="utf-8")
    (dest / "environment" / "Dockerfile").write_text(
        _dockerfile(task, meta, prof), encoding="utf-8"
    )
    (dest / "environment" / "docker-compose.yaml").write_text(
        _compose(task, prof), encoding="utf-8"
    )

    # Upstream python, byte-for-byte. evaluate.py imports utils/custom_labels/testing_util
    # as siblings, so the whole set travels together.
    for py in sorted(src.glob("*.py")):
        shutil.copy2(py, dest / "tests" / "upstream" / py.name)

    (dest / "tests" / "anchors.json").write_text(
        json.dumps(anchors(task).as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    (dest / "tests" / "score.py").write_text(
        _SCORE_PY.format(normalise=_normalise_block()), encoding="utf-8"
    )
    test_sh = dest / "tests" / "test.sh"
    test_sh.write_text(_TEST_SH.format(task=task), encoding="utf-8")
    test_sh.chmod(0o755)

    solve = dest / "solution" / "solve.sh"
    ref = REFERENCE_SOLUTIONS.get(task)
    if ref:
        (dest / "solution" / "reference.py").write_text(ref, encoding="utf-8")
        solve.write_text(
            _SOLVE_SH.format(
                task=task,
                note="Deliberately far from SOTA: it exists to prove the grader responds.",
            ),
            encoding="utf-8",
        )
    else:
        (dest / "solution" / "reference.py").unlink(missing_ok=True)
        solve.write_text(_NO_REFERENCE.format(task=task), encoding="utf-8")
    solve.chmod(0o755)
    return dest


# ------------------------------------------------------------------------- host stage


def stage(task: str, raw_dir: Path | None = None, prepared_dir: Path | None = None,
          python: str | None = None, task_dir: Path | None = None) -> dict[str, Any]:
    """Run the upstream preparers once on the host.

    Produces two things: the agent-visible splits under ``prepared/<task>/`` (mounted
    read-only at ``/app/data``) and ``tests/eval_data/test_with_labels`` inside the
    generated task directory (uploaded by Harbor only at verify time).

    ``evaluate_prepare.py`` also copies the agent's ``submission.csv`` next to the gold
    labels, which cannot happen on the host before the agent has run -- so it is handed
    a throwaway one and only ``test_with_labels`` is kept. ``tests/test.sh`` does the
    submission copy at verify time instead.
    """
    raw = Path(raw_dir) if raw_dir else raw_root()
    prepared = ensure((Path(prepared_dir) if prepared_dir else prepared_root()) / task)
    tdir = Path(task_dir) if task_dir else out_root() / task
    py = python or sys.executable
    src = upstream_dir(task)

    if not raw.is_dir():
        raise FileNotFoundError(f"raw dataset root {raw} does not exist")

    def run(script: str, mount: Path, log: Path) -> None:
        cmd = [py, str(src / script),
               "--global-shared-data-dir", str(raw),
               "--agent-data-mount-dir", str(mount),
               "--agent-log-dir", str(log)]
        subprocess.run(cmd, check=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        log = ensure(tmp_path / "log")
        run("prepare.py", prepared, log)

        # evaluate_prepare.py unconditionally copies {agent_log_dir}/submission.csv.
        (log / "submission.csv").write_text("placeholder\n", encoding="utf-8")
        eval_mount = ensure(tmp_path / "eval")
        run("evaluate_prepare.py", eval_mount, log)

        gold_src = eval_mount / "test_with_labels"
        if not gold_src.is_dir():
            raise RuntimeError(f"{task}: evaluate_prepare.py produced no test_with_labels")
        gold_dst = tdir / "tests" / "eval_data" / "test_with_labels"
        if gold_dst.exists():
            shutil.rmtree(gold_dst)
        ensure(gold_dst.parent)
        shutil.copytree(gold_src, gold_dst)

    return {
        "task": task,
        "prepared": str(prepared),
        "prepared_bytes": _du(prepared),
        "gold": str(gold_dst),
        "gold_bytes": _du(gold_dst),
    }


def _du(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def raw_present(task: str, raw_dir: Path | None = None) -> bool:
    hf_id, config = dataset_of(task)
    root = Path(raw_dir) if raw_dir else raw_root()
    return (root / hf_id / str(config)).is_dir()


# --------------------------------------------------------------------------------- CLI


def selected_tasks() -> list[str]:
    return [e.task for e in scope.load("airs")]


def _cmd_generate(args: argparse.Namespace) -> int:
    tasks = args.task or (selected_tasks() if args.all else [])
    if not tasks:
        print("nothing to do: pass --task NAME or --all", file=sys.stderr)
        return 2
    for t in tasks:
        d = generate(t, profile=args.profile, out_dir=args.out)
        print(f"{t}: {d}")
    return 0


def _cmd_stage(args: argparse.Namespace) -> int:
    for t in args.task:
        info = stage(t, raw_dir=args.raw, prepared_dir=args.prepared, python=args.python)
        print(json.dumps(info))
    return 0


def _cmd_plan(args: argparse.Namespace) -> int:
    rows = []
    for e in scope.load("airs"):
        hf_id, config = dataset_of(e.task)
        a = anchors(e.task)
        rows.append({
            "task": e.task,
            "dataset": f"{hf_id}:{config}",
            "raw_present": raw_present(e.task),
            "raw_bytes": _du(raw_root() / hf_id / str(config))
            if raw_present(e.task) else 0,
            "reference_solution": e.task in REFERENCE_SOLUTIONS,
            "metric": a.metric,
            "direction": a.direction,
        })
    print(json.dumps(rows, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="awm.adapters.airs", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="write Harbor task directories")
    g.add_argument("--task", action="append")
    g.add_argument("--all", action="store_true")
    g.add_argument("--profile", default="smoke", choices=sorted(PROFILES))
    g.add_argument("--out", type=Path)
    g.set_defaults(func=_cmd_generate)

    s = sub.add_parser("stage", help="run the upstream preparers on the host")
    s.add_argument("--task", action="append", required=True)
    s.add_argument("--raw", type=Path)
    s.add_argument("--prepared", type=Path)
    s.add_argument("--python", help="interpreter with the upstream 'datasets' pin")
    s.set_defaults(func=_cmd_stage)

    p = sub.add_parser("plan", help="what data each task needs and whether we have it")
    p.add_argument("--all", action="store_true", help="include excluded tasks")
    p.set_defaults(func=_cmd_plan)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
