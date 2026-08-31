"""Frozen evaluators: build, hash, run, and diff them.

Two kinds behind one per-item result schema:

* ``official`` — the benchmark's ``evaluate.py --limit N``; the argv template
  comes from ``config.official_argv`` so a fake grader can stand in for tests.
  Emits ``metrics.json`` with ``accuracy`` and ``stderr``.
* ``custom`` — a jsonl of ``{id, question, gold}`` scored by
  ``config.custom_argv`` (default: ``awm.wm.score_items``). Emits
  ``metrics.json`` and ``items.jsonl`` with per-item ``correct``.

Both leave ``metrics.json`` as ``{"value": float, "n": int, "stderr": float}``
after normalisation, which is what observations read.
"""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .schema import WMError, dump_json, load_json, read_jsonl, sha256_file, sha256_obj

DEFAULT_OFFICIAL_ARGV = ["python", "evaluate.py", "--model-path", "{checkpoint}", "--limit", "{n}",
                         "--json-output-file", "{out}/metrics.json"]
DEFAULT_CUSTOM_ARGV = [sys.executable, "-m", "awm.wm.score_items", "--model", "{checkpoint}",
                       "--items", "{items}", "--out", "{out}", "--limit", "{n}"]


def evaluator_hash(spec: dict[str, Any]) -> str:
    payload = {k: v for k, v in spec.items() if k != "hash"}
    if spec.get("kind") == "custom" and spec.get("items"):
        payload["items_sha256"] = sha256_file(Path(spec["items"]))
    return sha256_obj(payload)


def freeze_evaluators(contract: dict[str, Any], evaluators_dir: Path) -> None:
    """Copy custom item files under the card and stamp every spec with its hash."""
    for spec in contract["evaluators"]:
        edir = evaluators_dir / spec["name"]
        edir.mkdir(parents=True, exist_ok=True)
        if spec["kind"] == "custom":
            src = Path(spec["items"])
            if not src.is_file():
                raise WMError(f"evaluator {spec['name']}: items file {src} missing")
            dst = edir / "items.jsonl"
            if src.resolve() != dst.resolve():
                dst.write_text(src.read_text())
            spec["items"] = str(dst)
            rows = read_jsonl(dst)
            if not rows or not all({"id", "question", "gold"} <= set(r) for r in rows):
                raise WMError(f"evaluator {spec['name']}: items need id, question, gold")
            spec["n"] = min(int(spec["n"]), len(rows))
        spec["hash"] = evaluator_hash(spec)
        dump_json(edir / "spec.json", spec)


def _render(argv: list[str], **fills: Any) -> list[str]:
    out = []
    for a in argv:
        for k, v in fills.items():
            a = a.replace("{" + k + "}", str(v))
        out.append(a)
    return out


def run_evaluator(spec: dict[str, Any], checkpoint: Path, out_dir: Path, config: dict[str, Any],
                  session_dir: Path) -> dict[str, Any]:
    """Run one evaluator on one checkpoint. Returns the normalised metrics."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if spec["kind"] == "official":
        argv = _render(config.get("official_argv") or DEFAULT_OFFICIAL_ARGV,
                       checkpoint=checkpoint, n=spec["n"], out=out_dir)
        cwd = Path(config.get("official_cwd") or session_dir)
    else:
        argv = _render(config.get("custom_argv") or DEFAULT_CUSTOM_ARGV,
                       checkpoint=checkpoint, n=spec["n"], out=out_dir, items=spec["items"])
        cwd = session_dir
    log = out_dir / "run.log"
    started = time.monotonic()
    with log.open("w") as fh:
        fh.write("$ " + " ".join(shlex.quote(a) for a in argv) + f"\n(cwd {cwd})\n\n")
        fh.flush()
        proc = subprocess.run(argv, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, check=False,
                              timeout=config.get("evaluator_timeout_s", 4 * 3600))
    wall = time.monotonic() - started
    if proc.returncode != 0:
        raise WMError(f"evaluator {spec['name']} failed (exit {proc.returncode}); see {log}")
    metrics_path = out_dir / "metrics.json"
    raw = load_json(metrics_path)
    value = raw.get("value", raw.get(spec["metric"], raw.get("accuracy")))
    if value is None:
        raise WMError(f"evaluator {spec['name']}: no metric in {metrics_path}")
    n = int(raw.get("n", spec["n"]))
    stderr = raw.get("stderr")
    if stderr is None:
        stderr = math.sqrt(max(value * (1 - value), 0) / n) if 0 <= value <= 1 and n else None
    items_path = out_dir / "items.jsonl"
    metrics = {
        "evaluator": spec["name"], "metric": spec["metric"], "direction": spec["direction"],
        "value": float(value), "n": n, "stderr": float(stderr) if stderr is not None else None,
        "wall_s": round(wall, 1), "items": str(items_path) if items_path.is_file() else None,
        "raw": str(metrics_path), "evaluator_hash": spec["hash"],
    }
    dump_json(out_dir / "normalized.json", metrics)
    return metrics


def watch_transitions(before_items: Path | None, after_items: Path | None) -> dict[str, Any] | None:
    """fixed / still_failing / regressions between two per-item result files."""
    if not before_items or not after_items or not Path(before_items).is_file() or not Path(after_items).is_file():
        return None
    before = {str(r["id"]): bool(r.get("correct")) for r in read_jsonl(Path(before_items))}
    after = {str(r["id"]): bool(r.get("correct")) for r in read_jsonl(Path(after_items))}
    fixed = sorted(i for i in after if after[i] and not before.get(i, False))
    still = sorted(i for i in after if not after[i] and not before.get(i, False))
    regress = sorted(i for i in after if not after[i] and before.get(i, False))
    return {"fixed": len(fixed), "still_failing": len(still), "regressions": len(regress),
            "fixed_ids": fixed[:50], "regression_ids": regress[:50]}


def delta(a: dict[str, Any] | None, b: dict[str, Any] | None) -> float | None:
    if not a or not b or a.get("value") is None or b.get("value") is None:
        return None
    return round(float(a["value"]) - float(b["value"]), 6)


def bernoulli_stderr(value: float, n: int) -> float:
    return math.sqrt(max(value * (1 - value), 0) / max(n, 1))


def parse_json_maybe(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None
