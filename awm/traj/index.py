"""One flat table over every converted run.

The analysis layer selects runs from this table instead of opening two thousand
``*.meta.json`` files. It holds only what selection and filtering need — ids,
model/harness, budget, score, event and token counts, flags — plus the path to
the event stream, which is where everything else lives.

Two conventions, both load-bearing:

*   Absent is NA, never 0. A run whose harness reported no cost must not read as
    free, and a run we never counted thinking events for must not read as having
    thought zero times. Every numeric column is a pandas nullable dtype so the
    distinction survives, in memory and through parquet.
*   Counts that ``schema.summarize`` derived are complete for the stream, so a
    type absent from ``n_by_type`` really did occur zero times. An *empty*
    ``n_by_type`` means nobody counted, and becomes NA.
*   Token counts are the opposite: they are what the harness reported, not
    something we derived, so a *missing key* is already NA. A harness that
    counted zero of something writes the zero (``map_usage`` keeps an integer 0),
    while 106 converted Claude Code runs were killed before their ``result`` line
    and have no output count at all — ``tok_out`` for those is unknown, not zero.

``RunMeta.flags`` is free-form (each converter records the validity verdicts its
source published), so the two boolean-ish columns read it by rule rather than by
key: a key whose name is an explanation (``*_why``, ``justification_*``, ...) is
never a flag itself, and a verdict string that upstream uses to mean "fine" —
PI's ``validity: "healthy"``, and the like — does not fire. Measured shapes:
PI publishes ``validity`` in {healthy, flagged} with ``flagged_why``;
PostTrainBench publishes boolean ``contamination`` / ``disallowed_model`` beside
long ``justification_*`` strings.

``t_start`` / ``t_end`` stay strings: several sources carry no timestamps at all
and others use formats we must not silently reinterpret. ``duration_s`` is the
column to do arithmetic on.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from awm import paths
from awm.traj.schema import RunMeta, iter_runs, read_meta

#: Column -> pandas dtype, in table order. This is the whole contract of the index.
DTYPES: dict[str, str] = {
    "run_id": "string",
    "source": "string",
    "benchmark": "string",
    "task_id": "string",
    "model": "string",
    "harness": "string",
    "budget_hours": "Float64",
    "budget_gpus": "Int64",
    "t_start": "string",
    "t_end": "string",
    "duration_s": "Float64",
    "metric": "string",
    "score": "Float64",
    "score_normalized": "Float64",
    "direction": "string",
    "n_events": "Int64",
    "n_thinking": "Int64",
    "n_tool_use": "Int64",
    "n_text": "Int64",
    "n_harness_events": "Int64",
    "n_subagents": "Int64",
    "tok_in": "Int64",
    "tok_out": "Int64",
    "tok_cache_read": "Int64",
    "cost_usd": "Float64",
    "flagged": "boolean",
    "flag_reasons": "string",
    "events_path": "string",
}

COLUMNS: tuple[str, ...] = tuple(DTYPES)

#: Verdict strings upstream uses to mean "nothing wrong here".
_CLEAR_VERDICTS = frozenset(
    {"", "healthy", "valid", "ok", "okay", "clean", "pass", "passed", "none", "no", "false", "0"}
)

_EXPLANATION_WORDS = ("why", "reason", "reasons", "justification", "note", "notes", "detail",
                      "details", "comment", "comments")


def _is_explanation(key: str) -> bool:
    k = key.lower()
    return any(
        k == w or k.startswith(f"{w}_") or k.endswith(f"_{w}") for w in _EXPLANATION_WORDS
    )


def _fires(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in _CLEAR_VERDICTS
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    return bool(value)


def _explanation_for(key: str, flags: dict[str, Any]) -> str | None:
    candidates = [f"{key}_why", f"why_{key}", f"{key}_reason", f"justification_{key}"]
    # PI keys its explanation off the verdict's value ("flagged_why"), not its name.
    if key in ("validity", "valid", "flagged", "validity_verdict"):
        candidates += ["flagged_why", "why", "reason", "justification"]
    for c in candidates:
        v = flags.get(c)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def read_flags(flags: dict[str, Any]) -> tuple[bool | None, str | None]:
    """``(flagged, flag_reasons)`` for one run's published validity verdicts."""
    if not flags:
        return None, None
    reasons: list[str] = []
    for key in sorted(flags):
        value = flags[key]
        if _is_explanation(key) or not _fires(value):
            continue
        label = key if value is True else f"{key}={value}"
        why = _explanation_for(key, flags)
        reasons.append(f"{label}: {why}" if why else label)
    return bool(reasons), "; ".join(reasons) if reasons else None


def _count(counts: dict[str, int], key: str) -> int | None:
    """Zero only when somebody actually counted this stream."""
    return counts.get(key, 0) if counts else None


def _tokens(counts: dict[str, int], key: str) -> int | None:
    """A token stream nobody reported is NA — see the module docstring."""
    return counts.get(key)


def row_from_meta(meta: RunMeta, events_path: Path | str | None = None) -> dict[str, Any]:
    """Flatten one ``RunMeta`` into the index's columns."""
    budget = meta.budget or {}
    score = meta.final_score or {}
    tokens = meta.tokens or {}
    flagged, flag_reasons = read_flags(meta.flags)
    return {
        "run_id": meta.run_id,
        "source": meta.source,
        "benchmark": meta.benchmark,
        "task_id": meta.task_id,
        "model": meta.model,
        "harness": meta.harness,
        "budget_hours": budget.get("hours"),
        "budget_gpus": budget.get("gpus"),
        "t_start": meta.t_start,
        "t_end": meta.t_end,
        "duration_s": meta.duration_s,
        "metric": score.get("metric"),
        "score": score.get("value"),
        "score_normalized": score.get("normalized"),
        "direction": score.get("direction"),
        "n_events": meta.n_events,
        "n_thinking": _count(meta.n_by_type, "thinking"),
        "n_tool_use": _count(meta.n_by_type, "tool_use"),
        "n_text": _count(meta.n_by_type, "text"),
        "n_harness_events": _count(meta.n_by_origin, "harness"),
        "n_subagents": len(meta.subagents),
        "tok_in": _tokens(tokens, "in"),
        "tok_out": _tokens(tokens, "out"),
        "tok_cache_read": _tokens(tokens, "cache_read"),
        "cost_usd": meta.cost_usd,
        "flagged": flagged,
        "flag_reasons": flag_reasons,
        "events_path": None if events_path is None else str(events_path),
    }


def frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Rows -> the index frame, with the full column set and dtypes even when empty."""
    df = pd.DataFrame(rows, columns=list(COLUMNS))
    return df.astype(DTYPES)


def empty() -> pd.DataFrame:
    return frame([])


def default_events_root() -> Path:
    return paths.events_root()


def build(events_root: Path | None = None) -> pd.DataFrame:
    """Index every run under ``<events_root>/<source>/``, sorted by source then run."""
    root = Path(events_root) if events_root is not None else default_events_root()
    rows: list[dict[str, Any]] = []
    if root.is_dir():
        for source_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for _run_id, ep, mp in iter_runs(source_dir):
                # write_run lays down events before meta, so a run being converted
                # right now is half-visible; take it next time.
                if mp.exists():
                    rows.append(row_from_meta(read_meta(mp), ep))
    df = frame(rows)
    return df.sort_values(["source", "run_id"], kind="stable").reset_index(drop=True)


def build_source(source_dir: Path) -> pd.DataFrame:
    """Index one source directory, e.g. ``events/pi_speedrun``."""
    rows = [
        row_from_meta(read_meta(mp), ep)
        for _run_id, ep, mp in iter_runs(Path(source_dir))
        if mp.exists()
    ]
    return frame(rows)


def save(df: pd.DataFrame, path: Path | None = None) -> Path:
    p = Path(path) if path is not None else paths.index_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(p)
    return p


def load(path: Path | None = None) -> pd.DataFrame:
    p = Path(path) if path is not None else paths.index_path()
    return pd.read_parquet(p).astype(DTYPES)


__all__ = [
    "COLUMNS",
    "DTYPES",
    "build",
    "build_source",
    "default_events_root",
    "empty",
    "frame",
    "load",
    "read_flags",
    "row_from_meta",
    "save",
]
