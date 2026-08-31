"""Materialise the prior-runs directory a scientist gets to read.

For every run of a split (train side, or train + test), copy the raw run
directory — ``solve_out.txt``, ``solve_parsed.txt``, ``metrics.json``,
``task/`` — into ``<out>/<agent config>/<run>/`` exactly as the corpus lays it
out, and write ``INDEX.md`` / ``index.jsonl`` with base model, agent, official
accuracy, wall time, and path. Nothing is masked: this is the full-information
baseline, and the study decision (2026-08-31) is that scores and agent
identity stay visible.

Two versions per split, chosen with ``--sides``:

    python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_193 --sides train,test
    python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_143 --sides train

The output is meant to be bind-mounted read-only at ``/home/ben/prior_runs``
(see rollout/patches/apply_extra_binds.py); it is a copy, not symlinks, so it
resolves inside the sandbox.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUN_RE = re.compile(r"^(?P<bench>[^_]+)_(?P<model>.+)_(?P<cid>\d+)$")


def run_record(run: str, side: str, src: Path) -> dict:
    config, run_name = run.split("/", 1)
    m = RUN_RE.match(run_name)
    model = m.group("model").replace("_", "/", 1) if m else None
    metrics = None
    mp = src / "metrics.json"
    if mp.is_file():
        try:
            metrics = json.loads(mp.read_text())
        except json.JSONDecodeError:
            metrics = {"raw": mp.read_text()[:80]}
    acc = metrics.get("accuracy") if isinstance(metrics, dict) else None
    tt = (src / "time_taken.txt").read_text().strip() if (src / "time_taken.txt").is_file() else None
    task_files = sorted(p.name for p in (src / "task").iterdir()) if (src / "task").is_dir() else []
    return {
        "run": run, "agent_config": config, "run_name": run_name, "side": side,
        "base_model": model, "accuracy": acc, "time_taken": tt,
        "has_trace": (src / "solve_out.txt").is_file(),
        "trace_bytes": (src / "solve_out.txt").stat().st_size if (src / "solve_out.txt").is_file() else 0,
        "task_files": task_files,
    }


def build(runs: list[tuple[str, str]], raw_dir: Path, out: Path, *, copy: bool = True) -> dict:
    """runs = [(run, side)]. Returns the index summary."""
    out.mkdir(parents=True, exist_ok=True)
    rows, missing = [], []
    for run, side in runs:
        src = raw_dir / run
        if not src.is_dir():
            missing.append(run)
            continue
        dst = out / run
        if copy:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        rec = run_record(run, side, src)
        rec["path"] = f"/home/ben/prior_runs/{run}"
        rows.append(rec)
    rows.sort(key=lambda r: (-(r["accuracy"] if r["accuracy"] is not None else -1), r["run"]))
    with (out / "index.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    lines = [
        "# Prior runs",
        "",
        (f"{len(rows)} previous attempts at this task by autonomous agents, one directory each, "
         "laid out as `<agent config>/<run>/`. Each holds `solve_out.txt` (the agent's full session "
         "trace), `solve_parsed.txt` (condensed), `task/` (every script it wrote and its own eval "
         "outputs), `metrics.json` (its official accuracy), and `time_taken.txt`."),
        "",
        "Sorted by official accuracy, best first.",
        "",
        "| accuracy | base model | agent config | time | trace | path |",
        "|---:|---|---|---|---:|---|",
    ]
    for r in rows:
        acc = f"{r['accuracy']:.3f}" if r["accuracy"] is not None else "—"
        lines.append(f"| {acc} | {r['base_model']} | {r['agent_config']} | {r['time_taken'] or '—'} | "
                     f"{r['trace_bytes'] // 1024} KB | `{r['path']}` |")
    (out / "INDEX.md").write_text("\n".join(lines) + "\n")
    summary = {"runs": len(rows), "missing": missing, "out": str(out),
               "by_model": {}, "by_side": {}}
    for r in rows:
        summary["by_model"][r["base_model"]] = summary["by_model"].get(r["base_model"], 0) + 1
        summary["by_side"][r["side"]] = summary["by_side"].get(r["side"], 0) + 1
    (out / "README.md").write_text(
        "Read-only copy of prior PostTrainBench runs for this task, built by "
        "tools/build_prior_runs.py. Start with INDEX.md.\n")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("split_id", help="e.g. posttrainbench/gsm8k-gemma-holdout-v1")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sides", default="train", help="comma list of train,test (default train)")
    ap.add_argument("--raw-dir", type=Path, help="override <data>/traj/raw/posttrainbench")
    ap.add_argument("--index-only", action="store_true", help="write INDEX.md/index.jsonl without copying")
    a = ap.parse_args()

    from awm import paths, splits

    s = splits.load(a.split_id)
    sides = [x.strip() for x in a.sides.split(",") if x.strip()]
    runs = [(r, side) for side in sides for r in getattr(s, side)]
    raw = a.raw_dir or paths.raw_dir("posttrainbench")
    summary = build(runs, raw, a.out, copy=not a.index_only)
    print(json.dumps(summary, indent=2))
    return 1 if summary["missing"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
