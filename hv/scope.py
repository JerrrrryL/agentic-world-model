"""Which tasks we run, in the form code can read.

``scope/<benchmark>.yaml`` lists the tasks that are in scope and the few facts a
program needs about each: what the metric is, what it is anchored against, and
what the task costs to run. Nothing here records *why* a task is in scope — the
selection, the gate reasoning and everything that was dropped live in
``doc/meeting/aug_24_data_select.md``, which is prose because that is what
judgement reads like. Duplicating it as structured data produced 3,000 lines of
YAML in which the same paragraph appeared thirty-two times.

File shape: keys above ``tasks`` are benchmark-wide and every task inherits
them; a task may override any of them. ``variants`` means the task is run once
per value, which is how PostTrainBench's four base models work.

:func:`check` is the part that earns its keep. The AIRS metric anchors are copies
of upstream ``metadata.yaml`` values, so it reads them back and reports any that
have drifted, and it reconciles the task counts and GPU-hour rows against the
numbers written in the document. Two hand-maintained copies of the same table
diverge otherwise.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterator

import yaml

from hv.paths import REPO_ROOT, scope_dir

#: Scope file per benchmark. The slug is the file stem and each entry's ``id`` prefix.
BENCHMARKS = ("posttrainbench", "airs", "speedrun_pi")

#: Keys a benchmark may set once for all its tasks, and a task may override.
INHERITED = ("resources", "budget", "metric", "family", "variants", "self_run")

DOC_PATH = REPO_ROOT / "doc" / "meeting" / "aug_24_data_select.md"
AIRS_UPSTREAM = REPO_ROOT / "third_party" / "airs-bench" / "airsbench" / "tasks" / "rad"


@dataclass(frozen=True)
class Entry:
    """One task in scope."""

    id: str
    benchmark: str
    metric: dict[str, Any]
    resources: dict[str, Any] = field(default_factory=dict)
    budget: dict[str, Any] = field(default_factory=dict)
    family: str | None = None
    #: Values this task is run once per, e.g. PostTrainBench's four base models.
    variants: tuple[str, ...] = ()
    #: False for a task we only analyse published trajectories of, never run
    #: ourselves. It costs no GPU, so the budget totals leave it out.
    self_run: bool = True

    @property
    def task(self) -> str:
        """The id without its benchmark prefix."""
        return self.id.split("/", 1)[1]

    @property
    def n_runs(self) -> int:
        """Runs one iteration of this task costs, counting variants."""
        return max(len(self.variants), 1)

    @property
    def gpu_hours(self) -> float | None:
        """One iteration at the official budget, or None if either number is absent."""
        if not self.self_run:
            return 0.0
        hours = self.budget.get("official_h")
        gpus = self.resources.get("gpus")
        return None if hours is None or gpus is None else hours * gpus * self.n_runs


class ScopeError(ValueError):
    """A scope file is malformed."""


def _parse(benchmark: str, doc: Any) -> list[Entry]:
    if not isinstance(doc, dict) or "tasks" not in doc:
        raise ScopeError(f"scope/{benchmark}.yaml: expected a mapping with a `tasks` key")
    unknown = set(doc) - set(INHERITED) - {"tasks"}
    if unknown:
        raise ScopeError(f"scope/{benchmark}.yaml: unknown top-level key(s) {sorted(unknown)}")
    shared = {k: doc[k] for k in INHERITED if k in doc}

    entries = []
    for i, raw in enumerate(doc["tasks"]):
        if not isinstance(raw, dict) or "id" not in raw:
            raise ScopeError(f"scope/{benchmark}.yaml: task #{i} has no id")
        merged = {**shared, **raw}
        unknown = set(merged) - set(INHERITED) - {"id"}
        if unknown:
            raise ScopeError(f"{benchmark}/{raw['id']}: unknown key(s) {sorted(unknown)}")
        metric = merged.get("metric")
        if not isinstance(metric, dict):
            raise ScopeError(f"{benchmark}/{raw['id']}: no metric, and none inherited")
        if metric.get("direction") not in ("higher_is_better", "lower_is_better"):
            raise ScopeError(
                f"{benchmark}/{raw['id']}: metric.direction is {metric.get('direction')!r}, "
                "want higher_is_better or lower_is_better"
            )
        entries.append(
            Entry(
                id=f"{benchmark}/{raw['id']}",
                benchmark=benchmark,
                metric=metric,
                resources=merged.get("resources") or {},
                budget=merged.get("budget") or {},
                family=merged.get("family"),
                variants=tuple(merged.get("variants") or ()),
                self_run=bool(merged.get("self_run", True)),
            )
        )
    return entries


@lru_cache(maxsize=None)
def _load_file(benchmark: str) -> tuple[Entry, ...]:
    path = scope_dir() / f"{benchmark}.yaml"
    if not path.exists():
        raise ScopeError(f"no scope file at {path}")
    return tuple(_parse(benchmark, yaml.safe_load(path.read_text(encoding="utf-8"))))


def load(benchmark: str | None = None) -> list[Entry]:
    """Every task in scope, or just one benchmark's."""
    names = (benchmark,) if benchmark else BENCHMARKS
    for name in names:
        if name not in BENCHMARKS:
            raise ScopeError(f"unknown benchmark {name!r}; want one of {BENCHMARKS}")
    return [e for name in names for e in _load_file(name)]


def get(entry_id: str) -> Entry:
    """One task by its full ``<benchmark>/<task>`` id."""
    prefix = entry_id.split("/", 1)[0]
    for e in load(prefix if prefix in BENCHMARKS else None):
        if e.id == entry_id:
            return e
    raise KeyError(entry_id)


def select(**filters: Any) -> list[Entry]:
    """Tasks whose attributes all equal the given values; None filters are ignored."""
    wanted = {k: v for k, v in filters.items() if v is not None}
    return [e for e in load() if all(getattr(e, k) == v for k, v in wanted.items())]


def tasks(benchmark: str | None = None) -> Iterator[str]:
    """Just the ids, for scripting."""
    return (e.id for e in load(benchmark))


def summary() -> list[tuple[str, int, int, float | None]]:
    """``(benchmark, tasks, runs, gpu_hours)`` for one iteration of each benchmark."""
    out = []
    for name in BENCHMARKS:
        entries = load(name)
        hours = [e.gpu_hours for e in entries]
        out.append(
            (
                name,
                len(entries),
                sum(e.n_runs for e in entries),
                None if any(h is None for h in hours) else sum(hours),
            )
        )
    return out


def _doc_text() -> str:
    return DOC_PATH.read_text(encoding="utf-8")


#: What the document claims, so :func:`check` can prove the two still agree.
#: 3.4 heading: "AIRS-Bench ... —— 20 题,✅ 8 + ⚠ 0(PoC 取 GPU-heavy 8;备选 6)"
_DOC_AIRS = re.compile(r"###\s*3\.4[^\n]*?(\d+)\s*题[^\n]*?✅\s*(\d+)\s*\+\s*⚠\s*(\d+)")
#: 3.2 body: "PoC 自跑:1 基准 × 1 模型 = 1 配置;观察组 = 0"
_DOC_PTB = re.compile(
    r"PoC 自跑[::]\s*(\d+)\s*基准\s*×\s*(\d+)\s*模型\s*=\s*(\d+)\s*配置[;;]\s*观察组\s*=\s*(\d+)"
)
#: Section 5.1 rows, each ending "| <runs> | <GPU-hours for one iteration> |".
_BUDGET_ROWS = (
    ("AIRS PoC", "airs", "poc_h"),
    ("PostTrainBench 自跑", "posttrainbench", "official_h"),
)


def _check_counts(problems: list[str]) -> None:
    doc = _doc_text()

    m = _DOC_AIRS.search(doc)
    if m is None:
        problems.append("doc 3.4: could not read the AIRS count claim; the regex is stale")
    else:
        selected = int(m.group(2))
        have = len(load("airs"))
        if selected != have:
            problems.append(
                f"airs: doc 3.4 claims {selected} selected tasks, scope/airs.yaml lists {have} "
                "(tasks still pending G1 belong in the doc, not here)"
            )

    m = _DOC_PTB.search(doc)
    if m is None:
        problems.append("doc 3.2: could not read the PostTrainBench count claim; regex is stale")
    else:
        n_core, n_models, n_core_cfg, n_observe_cfg = (int(g) for g in m.groups())
        ptb = load("posttrainbench")
        for label, want, have in (
            ("tasks", n_core + n_observe_cfg // n_models, len(ptb)),
            ("base models", n_models, len({v for e in ptb for v in e.variants})),
            ("configurations", n_core_cfg + n_observe_cfg, sum(e.n_runs for e in ptb)),
        ):
            if want != have:
                problems.append(f"posttrainbench: doc 3.2 implies {want} {label}, scope has {have}")


def _check_budget(problems: list[str]) -> None:
    """The GPU-hour rows in section 5.1 must still follow from the registry."""
    doc = _doc_text()
    for label, benchmark, budget_key in _BUDGET_ROWS:
        row = re.search(rf"\|\s*{re.escape(label)}[^\n]*?\|\s*(\d+)\s*\|\s*[≈~]?\s*(\d+)\s*\|", doc)
        if row is None:
            problems.append(f"doc 5.1: no budget row for {label!r}; the table or the regex moved")
            continue
        want_runs, want_hours = int(row.group(1)), int(row.group(2))
        entries = load(benchmark)
        runs = sum(e.n_runs for e in entries if e.self_run)
        hours = sum(
            (e.budget.get(budget_key) or 0) * (e.resources.get("gpus") or 0) * e.n_runs
            for e in entries
            if e.self_run
        )
        if runs != want_runs:
            problems.append(f"doc 5.1 {label!r}: table says {want_runs} runs, scope has {runs}")
        if hours != want_hours:
            problems.append(
                f"doc 5.1 {label!r}: table says {want_hours} GPU-hours, scope implies {hours} "
                f"({budget_key} x gpus x runs)"
            )


def _check_airs_upstream(problems: list[str]) -> None:
    """Re-read the AIRS anchors from upstream, since ours are copies of them.

    Skipped when the submodule is not checked out: that is a checkout state, not a
    registry inconsistency. Upstream tasks we do not list are not reported — this
    file holds what is in scope, and what was dropped is the document's business.
    """
    if not AIRS_UPSTREAM.is_dir():
        return
    for e in load("airs"):
        meta_path = AIRS_UPSTREAM / e.task / "metadata.yaml"
        if not meta_path.exists():
            problems.append(f"{e.id}: no upstream task directory at {AIRS_UPSTREAM}/{e.task}")
            continue
        meta = yaml.safe_load(meta_path.read_text(encoding="utf-8"))
        info = meta["logging_info"]
        want = {
            "name": info["metric"],
            "direction": "lower_is_better" if meta["metric_lower_is_better"] else "higher_is_better",
            "reference": info["sota"][0]["sota_score"],
            "s_min": info["estimated_worst_score"],
            "s_opt": info["optimal_score"],
        }
        for key, upstream in want.items():
            if e.metric.get(key) != upstream:
                problems.append(
                    f"{e.id}: metric.{key} is {e.metric.get(key)!r}, "
                    f"upstream metadata.yaml says {upstream!r}"
                )
        if e.family != info["category"]:
            problems.append(
                f"{e.id}: family is {e.family!r}, upstream category is {info['category']!r}"
            )


def check() -> list[str]:
    """Everything that could have drifted, as human-readable problems."""
    problems: list[str] = []
    seen: set[str] = set()
    for name in BENCHMARKS:
        try:
            entries = _load_file(name)
        except ScopeError as exc:
            problems.append(str(exc))
            continue
        for e in entries:
            if e.id in seen:
                problems.append(f"duplicate id {e.id}")
            seen.add(e.id)
    if problems:
        return problems  # the later checks assume every file loads

    _check_counts(problems)
    _check_budget(problems)
    _check_airs_upstream(problems)
    return problems
