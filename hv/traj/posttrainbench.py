"""PostTrainBench runs: directory conventions, the shared line reader, and RunMeta.

One run is a directory ``<raw_root>/<agent_config>/<run_dir>/`` whose
``solve_out.txt`` is the coding CLI's own JSONL, captured verbatim. Two CLIs are
covered here, each with its own converter module: Claude Code
(``--output-format stream-json``) and Codex (``exec --json``).

Upstream quirks a future reader would otherwise rediscover the hard way:

*   ``solve_out.txt`` lines are sometimes prefixed ``"[2026-06-07T21:31:06Z] "``
    by the harness's ``timestamp_lines.py`` and sometimes not — measured: every
    line of the claude sample, zero lines of the codex sample. The prefix is the
    only timestamp source; when it is absent the events carry no ``ts`` at all.
*   The file is not pure JSONL. The launcher's CUDA preamble, HF warnings and a
    final ``Terminated`` share the stream (11 of 663 lines in the claude sample,
    10 of 300 in codex). Those lines are counted into ``RunMeta.extra``, never
    silently dropped.
*   ``metrics.json`` is the published score; ``time_taken.txt`` is wall clock for
    the whole slot (solve + eval), so it is longer than the trajectory.
*   The contamination verdict is ``judgement_gpt5_4.json`` for most runs but
    ``judgement_api.json`` for some, and either may hold the plain text
    ``"Entry not found"`` instead of JSON (measured on the codex sample).
*   ``system_monitor.log`` (60 s samples) is recorded in ``source_paths`` and
    deliberately not parsed into events: it is machine telemetry, not agent
    behaviour.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from hv.traj.schema import MAIN_AGENT, Event, RunMeta, SubAgent, summarize, write_run

SOURCE = "posttrainbench"

#: ``(ts_or_None, parsed_object_or_None, line_number, raw_line_without_prefix)``.
LineRow = tuple[str | None, dict[str, Any] | None, int, str]

_TS_PREFIX = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T[\d:.]+Z)\] ")

_AGENT_CONFIG = re.compile(
    r"^(?P<agent>[A-Za-z0-9]+)_(?P<rest>.+?)_+(?P<hours>\d+(?:\.\d+)?)h(?:_run(?P<run>\d+))?$"
)
_RUN_DIR = re.compile(r"^(?P<benchmark>[^_]+)_(?P<model>.+)_(?P<cluster>\d+)$")

_JUDGEMENT_FILES = ("judgement_gpt5_4.json", "judgement_api.json")


@dataclass(frozen=True)
class RunDir:
    """One PostTrainBench run directory, with both directory names parsed."""

    path: Path
    agent_config: str
    agent: str
    config: str
    model: str
    hours: float
    run_index: int | None
    context_1m: bool
    benchmark: str
    hf_org: str
    base_model: str
    cluster_id: str

    @property
    def run_id(self) -> str:
        return f"{self.agent_config}__{self.path.name}"

    @property
    def solve_out(self) -> Path:
        return self.path / "solve_out.txt"


def parse_agent_config(name: str) -> dict[str, Any]:
    """``claude_non_api_max_claude-opus-4-8_10h_run1`` -> its parts.

    A ``_1m_`` segment marks a 1M-context variant and doubles the separator
    before the hours (``claude-fable-5_1m__10h``), hence the ``_+``.
    """
    m = _AGENT_CONFIG.match(name)
    if not m:
        raise ValueError(f"unparsable agent config directory name: {name!r}")
    parts = m.group("rest").split("_")
    context_1m = parts[-1] == "1m"
    if context_1m:
        parts = parts[:-1]
    run = m.group("run")
    return {
        "agent": m.group("agent"),
        "config": "_".join(parts[:-1]),
        "model": parts[-1],
        "hours": float(m.group("hours")),
        "run_index": int(run) if run is not None else None,
        "context_1m": context_1m,
    }


def parse_run_dir_name(name: str) -> dict[str, str]:
    """``gsm8k_Qwen_Qwen3-1.7B-Base_17315721`` -> benchmark, org, model, cluster id.

    The cluster id is the trailing integer and the benchmark the leading token;
    everything between is the HF model path with ``/`` replaced by ``_``.
    """
    m = _RUN_DIR.match(name)
    if not m:
        raise ValueError(f"unparsable run directory name: {name!r}")
    model = m.group("model")
    org, _, base = model.partition("_")
    return {
        "benchmark": m.group("benchmark"),
        "hf_org": org if base else "",
        "base_model": base or model,
        "cluster_id": m.group("cluster"),
    }


def make_run_dir(agent_config: str, path: Path) -> RunDir:
    return RunDir(path=path, agent_config=agent_config, **parse_agent_config(agent_config),
                  **parse_run_dir_name(path.name))


def iter_run_dirs(raw_root: Path) -> Iterator[RunDir]:
    """Every ``<agent_config>/<run_dir>`` under ``raw_root`` that has a solve_out."""
    for cfg in sorted(p for p in raw_root.iterdir() if p.is_dir() and not p.name.startswith(".")):
        for rd in sorted(p for p in cfg.iterdir() if p.is_dir() and not p.name.startswith(".")):
            if (rd / "solve_out.txt").exists():
                yield make_run_dir(cfg.name, rd)


def read_line_stream(solve_out: Path) -> Iterator[LineRow]:
    """Yield every line of a solve_out, JSON or not, with its optional timestamp."""
    with solve_out.open("r", encoding="utf-8", errors="replace") as fh:
        for lineno, line in enumerate(fh, 1):
            raw = line.rstrip("\n")
            ts = None
            m = _TS_PREFIX.match(raw)
            if m:
                ts = m.group(1)
                raw = raw[m.end():]
            obj: dict[str, Any] | None = None
            if raw.startswith("{"):
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, dict):
                    obj = parsed
            yield ts, obj, lineno, raw


def number_events(events: list[Event]) -> list[Event]:
    """Assign ``i`` per agent stream, in emission order. Converters build their
    events out of order (codex opens a tool call before it completes), so the
    numbering is a final pass rather than a running counter."""
    counters: dict[str, int] = {}
    for e in events:
        e.i = counters.get(e.agent_id, 0)
        counters[e.agent_id] = e.i + 1
    return events


def detect_harness(solve_out: Path) -> str:
    """Sniff the first JSON object: codex opens a thread, Claude Code inits a session."""
    for _ts, obj, _n, _raw in read_line_stream(solve_out):
        if obj is None:
            continue
        kind = obj.get("type", "")
        if kind == "thread.started" or kind.startswith(("item.", "turn.")):
            return "codex"
        if kind in ("system", "assistant", "user", "result", "rate_limit_event"):
            return "claude-code"
        return "unknown"
    return "unknown"


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    text = path.read_text(encoding="utf-8", errors="replace")
    # Some judgement files hold the plain text "Entry not found" instead of JSON.
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def _read_time_taken(path: Path) -> float | None:
    """``HH:MM:SS`` -> seconds."""
    if not path.exists():
        return None
    parts = path.read_text(encoding="utf-8").strip().split(":")
    if len(parts) != 3 or not all(p.strip().isdigit() for p in parts):
        return None
    h, m, s = (int(p) for p in parts)
    return float(h * 3600 + m * 60 + s)


def _judgement(run: RunDir) -> tuple[dict[str, Any], Path | None]:
    """The published verdicts, and the file they came from.

    Only verdicts go in the dict — which file was read is provenance and belongs
    in ``source_paths``. Putting it in ``flags`` would make every judged run look
    flagged to anything that reads a non-empty flag as a problem.
    """
    for name in _JUDGEMENT_FILES:
        p = run.path / name
        if not p.exists():
            continue
        obj = _read_json(p)
        if obj is None:
            raw = p.read_text(encoding="utf-8", errors="replace").strip()
            return {"judgement_unavailable": raw[:200]}, p
        return dict(obj), p
    return {}, None


def _source_paths(run: RunDir) -> dict[str, Any]:
    names = {
        "solve_out": "solve_out.txt",
        "solve_parsed": "solve_parsed.txt",
        "metrics": "metrics.json",
        "time_taken": "time_taken.txt",
        "system_monitor": "system_monitor.log",
        "error_log": "error.log",
    }
    paths = {k: str(run.path / v) for k, v in names.items() if (run.path / v).exists()}
    _, jpath = _judgement(run)
    if jpath is not None:
        paths["judgement"] = str(jpath)
    return paths


def _subagents(events: list[Event]) -> list[dict[str, Any]]:
    """Sub-agent streams, labelled from the tool call that spawned them."""
    spawns = {e.tool_use_id: e for e in events if e.type == "tool_use" and e.tool_use_id}
    out: dict[str, SubAgent] = {}
    for e in events:
        if e.agent_id == MAIN_AGENT:
            continue
        sa = out.get(e.agent_id)
        if sa is None:
            spawn = spawns.get(e.parent_tool_use or "")
            label = None
            if spawn is not None and spawn.args:
                label = spawn.args.get("description") or spawn.args.get("subagent_type")
            sa = SubAgent(id=e.agent_id, label=label, parent_tool_use=e.parent_tool_use,
                          t_start=e.ts)
            out[e.agent_id] = sa
        sa.n_events += 1
        if e.ts:
            sa.t_end = e.ts
    return [vars(sa) for sa in out.values()]


def build_meta(run: RunDir, events: list[Event], harness: str, extra: dict[str, Any]) -> RunMeta:
    """Assemble RunMeta from the converted events plus the run's sibling files.

    ``extra`` is the converter's own bag; the keys ``cost_usd`` and ``tokens``
    are promoted to their RunMeta fields and everything else is kept verbatim.
    """
    extra = dict(extra)
    cost_usd = extra.pop("cost_usd", None)
    stats = summarize(events)
    tokens = extra.pop("tokens", None) or stats["tokens"]
    metrics = _read_json(run.path / "metrics.json")
    flags, _ = _judgement(run)
    stamped = [e.ts for e in events if e.ts]

    final_score = None
    if metrics is not None and "accuracy" in metrics:
        final_score = {
            "metric": "accuracy",
            "value": metrics["accuracy"],
            "stderr": metrics.get("stderr"),
            "direction": "higher",
        }
    else:
        # Runs that never produced a scoreable model ship either no metrics.json
        # or one holding the plain text "No metrics.json produced."
        mpath = run.path / "metrics.json"
        extra["metrics_unavailable"] = (
            mpath.read_text(encoding="utf-8", errors="replace").strip()[:200]
            if mpath.exists()
            else "missing"
        )

    extra.update(
        {
            "agent": run.agent,
            "agent_config": run.agent_config,
            "config": run.config,
            "run_index": run.run_index,
            "context_1m": run.context_1m,
            "hf_org": run.hf_org,
            "base_model": run.base_model,
            "cluster_id": run.cluster_id,
        }
    )
    return RunMeta(
        run_id=run.run_id,
        source=SOURCE,
        benchmark=run.benchmark,
        task_id=f"{run.benchmark}/{run.base_model}",
        model=run.model,
        harness=harness,
        budget={"hours": run.hours},
        t_start=stamped[0] if stamped else None,
        t_end=stamped[-1] if stamped else None,
        duration_s=_read_time_taken(run.path / "time_taken.txt"),
        final_score=final_score,
        tokens=tokens or None,
        cost_usd=cost_usd,
        n_events=stats["n_events"],
        n_by_type=stats["n_by_type"],
        n_by_origin=stats["n_by_origin"],
        tools=stats["tools"],
        subagents=_subagents(events),
        flags=flags,
        source_paths=_source_paths(run),
        extra=extra,
    )


def build_run(run: RunDir) -> tuple[list[Event], RunMeta]:
    """Convert one run directory in memory, dispatching on the sniffed harness."""
    # Imported here, not at module level: the converters take LineRow and
    # number_events from this module, so the dependency only runs one way.
    from hv.traj import convert_claude_code, convert_codex

    harness = detect_harness(run.solve_out)
    rows = list(read_line_stream(run.solve_out))
    if harness == "claude-code":
        events, extra = convert_claude_code.convert(rows, run.run_id)
    elif harness == "codex":
        events, extra = convert_codex.convert(rows, run.run_id)
    else:
        raise ValueError(f"unknown CLI format in {run.solve_out}")
    non_json = [(n, raw) for _ts, obj, n, raw in rows if obj is None]
    extra["n_lines"] = len(rows)
    extra["n_non_json_lines"] = len(non_json)
    extra["non_json_lines"] = [{"line": n, "text": raw[:200]} for n, raw in non_json[:40]]
    return events, build_meta(run, events, harness, extra)


def convert_run_dir(run: RunDir, out_dir: Path) -> Path:
    events, meta = build_run(run)
    return write_run(events, meta, out_dir)
