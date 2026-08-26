"""Convert the Prime Intellect ``frontier-automated-speedrun`` release to the v0 schema.

PI already normalised nine harnesses into one event shape, so the per-event work
here is mostly renaming; the value this module adds is everything *around* the
events — sub-agent streams, launcher-injection classification, truncation flags,
the manifest, and the scratchpad.

Layout of the release (a git clone, ``traces/`` under the repo root)::

    traces/events-<run>.json.gz     {"run": str, "events": [...]}   NOT jsonl
    traces/subagents-<run>.json.gz  {"index": [...], "events": {id: [...]}}
    traces/scratch-<run>.json.gz    [{"name", "rel", "chars", "text"}, ...]
    traces/manifest.json.gz         {baseline, record_bar, target, human_record, runs: [41]}

Upstream quirks a future reader would otherwise rediscover the hard way
(all measured over the full 41-run release, 172,694 main + 118,980 sub events):

*   ``usage`` is repeated on every assistant event of a turn, so the repeats
    have to go or token totals come out inflated ~3x and ``validate_stream``
    rejects the stream. But a ``turn`` is not always one API call: codex and
    qwen-code log several calls under one turn number (5,288 such turns in
    ``openai-gpt-5-6-sol--codex--044f97fbcd18`` alone), and codex restarts turn
    numbering at every session (6,583 resets across 93 sessions in that run).
    Keeping only the first record therefore loses real calls — it recovers
    0.08x of ``economics.out_tok`` on qwen-code. What the release actually
    repeats is one *identical* record per event of a call, so the distinct
    consecutive records of a turn are summed onto the turn's first event: that
    reproduces ``economics.out_tok`` exactly for every codex, qwen-code, pi and
    prime-agent run. Usage from a repeated turn number therefore lands on the
    first event bearing that number, which for codex is an earlier session.
*   qwen-code spells its cache counter ``cache`` (3,860 events); it is the
    gemini-style ``cachedContentTokenCount``, i.e. a cache *read*, so it maps
    onto ``cache_read``. No other harness uses that key.
*   PI ships no tool-call ids. ``tool_use_id`` is therefore minted positionally
    as ``"<agent_id>#<i>"``. It is an identity, not an upstream value.
    ``tool_result`` is left unlinked: adjacency to its call holds for
    claude-code but breaks for codex (13,654 of the 33,707 results in codex
    main streams are not preceded by a ``tool_use``), so pairing would guess.
*   claude-code, kimi-code and grok-cli text is capped with a trailing
    ``…[+N chars]``: 6,070 ``text`` fields and 657 ``tool_use`` summaries,
    6,727 events in all. Always at the end of the string, never mid-text.
*   ``manifest.run["n_subagents"]`` is the *number* of sub-agents (len(index)),
    not an event count.
*   ``kimi-k3--kimi-code--512eb075aefa`` reuses sub-agent ids across sessions:
    66 index entries collapse onto 40 distinct ids, and the bundle's ``events``
    map only kept one stream per id. For every colliding id exactly one index
    entry has ``n_events == len(events[id])``, so that entry is the surviving
    one; the rest are recorded as dropped in ``RunMeta.extra``.
*   ``has_subagents``/``has_scratch`` are absent (not ``false``) on the 13 live
    runs, several of which do have sub-agents. Never read them as booleans.
*   ``progression`` has two shapes — 339 records use ``value`` (train_steps as a
    string) with ``agent_h``/``tok_at``/``cost``, 104 "live" records use
    ``steps``/``val_loss``/``t`` (unix seconds). Normalised into one list here.
*   The grok-4.5 run carries no timestamps at all (``ts`` null throughout, index
    ``t_start`` null), and grok-4.6 sub-agent streams carry no ``i``/``turn``.
"""

from __future__ import annotations

import gzip
import json
import re
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from awm.traj.schema import Event, RunMeta, SubAgent, summarize, write_run

SOURCE = "pi_speedrun"
BENCHMARK = "nanogpt-speedrun"

#: The sandbox every run used, from PI's blog: one 8xH200 node, bwrap + netns.
#: The release publishes no configured wall-clock cap, so ``budget`` carries no
#: hours; realised agent-hours live in ``extra["agent_h"]``.
NODE = {"gpus": 8, "gpu_type": "H200"}

#: PI truncated claude-code, kimi-code and grok-cli strings with this marker,
#: always at the end.
TRUNC_MARK = re.compile(r"…\[\+\d+ chars\]\s*$")

#: Launcher-injected user messages. Matched only against ``text``/``user``
#: events; an over-broad rule here would turn agent decisions into scaffolding.
#: Counts are over the 41 main streams (3,051 user text events, 2,980 harness);
#: the 71 left as agent decisions were read one by one — 70 are sub-agent task
#: prompts the parent wrote, the last is a PI sanitizer placeholder whose
#: original author is unknowable, so it stays with the agent.
HARNESS_EXACT = frozenset(
    {
        "continue",  # 1029
        "Continue from where you left off.",  # 30
        "Reply with exactly: OK",  # qwen-code goal handshake
        "[Request interrupted by user]",  # launcher abort, no human present
        "[Request interrupted by user for tool use]",
    }
)

HARNESS_PREFIXES = (
    "Read program.md and follow it exactly.",  # the /goal prompt, re-injected
    "<subagent_notification>",  # codex
    "<environment_context>",  # codex
    "<codex_internal_context",
    "[context compacted — handoff summary]",  # codex
    "This session is being continued from a previous conversation",  # claude-code
    "<task-notification>",  # claude-code background tasks and Agent completions
    "<command-name>/goal</command-name>",  # claude-code-goal
    "<local-command-stdout>",
    "A session-scoped Stop hook is now active with condition:",
    "Stop hook feedback:",
    "[Your previous response had no visible output.",
    "Continue working on the active Goal.",  # qwen-code
    "<user_info>",  # grok-cli preamble
    "<user_query>",  # grok-cli goal injection; only ever wraps launcher text
    "<turn_aborted>",  # codex, after a launcher-side interrupt
    "<system-reminder>",  # grok-cli background-task / todo / skill reminders
    "[SYSTEM NOTIFICATION - NOT USER INPUT]",
    "[cross-attempt filesystem output redacted for release",  # PI sanitizer
)

#: Tool a harness uses to spawn a sub-agent, per manifest ``harness`` string.
SPAWN_TOOLS = {
    "claude-code": ("Agent",),
    "claude-code-goal": ("Agent",),
    "kimi-code": ("Agent", "AgentSwarm"),
    "kimi-code-goal": ("Agent", "AgentSwarm"),
    "codex": ("spawn_agent",),
    "grok-cli": ("spawn_subagent",),
}

#: A child's ``t_start`` is stamped after the spawning call returns. Measured
#: gaps: 3-9 ms (claude-code), up to 17.3 s (kimi-code, background spawns).
LINK_WINDOW_S = 120.0

#: Event keys that become Event fields or source_ref; everything else in an
#: upstream event is copied into ``Event.extra`` verbatim.
_CONSUMED = frozenset(
    {"type", "role", "ts", "i", "turn", "text", "redacted", "tool", "summary",
     "args", "is_error", "usage", "source_file", "source_line", "file", "old", "new"}
)


def _traces(raw_root: Path) -> Path:
    """Accept either the clone root or its ``traces/`` directory."""
    sub = raw_root / "traces"
    return sub if sub.is_dir() else raw_root


def _read_gz(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def load_manifest(raw_root: Path) -> dict[str, Any]:
    return _read_gz(_traces(raw_root) / "manifest.json.gz")


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def is_harness_message(text: str | None) -> bool:
    """True for a launcher-injected user message, false for anything an agent wrote."""
    if not text:
        return False
    s = text.strip()
    return s in HARNESS_EXACT or s.startswith(HARNESS_PREFIXES)


def _usage(raw: dict[str, Any] | None) -> tuple[dict[str, int] | None, str | None]:
    """Canonical counters plus grok-cli's prose ``note`` key, which is not a count."""
    if not raw:
        return None, None
    out: dict[str, int] = {}
    note = None
    for k, v in raw.items():
        if isinstance(v, str):
            note = v
        else:
            out["cache_read" if k == "cache" else k] = int(v)
    return (out or None), note


def _truncated(raw: dict[str, Any]) -> bool:
    return any(
        isinstance(raw.get(f), str) and TRUNC_MARK.search(raw[f]) for f in ("text", "summary")
    )


def _args(raw: dict[str, Any]) -> dict[str, Any] | None:
    """Tool input: codex/pi/prime-agent ship ``args``; Claude-Code ships file/old/new."""
    if raw.get("args") is not None:
        return raw["args"]
    if raw.get("type") != "tool_use":
        return None
    built = {k: raw[k] for k in ("file", "old", "new") if k in raw}
    return built or None


def to_event(raw: dict[str, Any], run_id: str, agent_id: str, i: int) -> Event:
    """One upstream event -> one Event. Pure: no filesystem, no cross-event state."""
    typ = raw["type"]
    extra = {k: v for k, v in raw.items() if k not in _CONSUMED}
    if typ != "tool_use":
        extra.update({k: raw[k] for k in ("file", "old", "new") if k in raw})
    usage, usage_note = _usage(raw.get("usage"))
    if usage_note:
        extra["usage_note"] = usage_note
    # 194 grok-cli events carry role "system" (the harness system prompt), which
    # the schema has no room for. Folded onto the only non-assistant role, always
    # origin "harness"; the upstream role stays in extra.
    role = raw["role"]
    if role not in ("user", "assistant"):
        extra["role_upstream"] = role
        role = "user"
    src = raw.get("source_file")
    return Event(
        run_id=run_id,
        agent_id=agent_id,
        i=i,
        type=typ,
        role=role,
        ts=raw.get("ts"),
        turn=raw.get("turn"),
        text=raw.get("text"),
        redacted=bool(raw.get("redacted", False)),
        truncated=_truncated(raw),
        tool=raw.get("tool"),
        args=_args(raw),
        summary=raw.get("summary"),
        is_error=raw.get("is_error"),
        tool_use_id=f"{agent_id}#{i}" if typ == "tool_use" else None,
        usage=usage,
        origin="harness"
        if "role_upstream" in extra
        or (typ == "text" and role == "user" and is_harness_message(raw.get("text")))
        else "agent",
        source_ref={"file": src, "line": raw.get("source_line")} if src else None,
        extra=extra or None,
    )


def convert_stream(raw_events: Iterable[dict[str, Any]], run_id: str, agent_id: str) -> list[Event]:
    """Convert one agent's events, renumbering ``i`` and folding per-turn usage.

    A call's usage is repeated verbatim on each of its events, so a record that
    differs from the previous one is a new call, not a repeat. The distinct
    records of a turn are summed onto that turn's first event — see the module
    docstring for why keeping only the first would drop real calls.
    """
    out: list[Event] = []
    head_of_turn: dict[int, Event] = {}
    last_of_turn: dict[int, dict[str, int]] = {}
    for i, raw in enumerate(raw_events):
        e = to_event(raw, run_id, agent_id, i)
        if e.usage is not None and e.turn is not None:
            call, e.usage = e.usage, None
            head = head_of_turn.get(e.turn)
            if head is None:
                e.usage = dict(call)
                head_of_turn[e.turn] = e
            elif call != last_of_turn[e.turn]:
                merged = dict(head.usage or {})
                for k, v in call.items():
                    merged[k] = merged.get(k, 0) + v
                head.usage = merged
            last_of_turn[e.turn] = call
        out.append(e)
    return out


def _dedupe_index(index: list[dict[str, Any]], streams: dict[str, list[dict[str, Any]]]) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]]
]:
    """Pick one index entry per sub-agent id; return (kept, dropped).

    Only ``kimi-k3--kimi-code--512eb075aefa`` needs this: its ids were reused
    across sessions and the bundle kept a single stream per id. The entry whose
    ``n_events`` matches the surviving stream is the one that stream belongs to
    (unique for each of the 26 ids that collide, so nothing is guessed).
    """
    by_id: dict[str, list[dict[str, Any]]] = {}
    for entry in index:
        by_id.setdefault(entry["id"], []).append(entry)
    kept, dropped = [], []
    for sid, entries in by_id.items():
        if len(entries) == 1:
            kept.append(entries[0])
            continue
        n = len(streams.get(sid, []))
        match = [e for e in entries if e.get("n_events") == n]
        winner = match[0] if match else entries[0]
        kept.append(winner)
        dropped.extend(e for e in entries if e is not winner)
    return kept, dropped


def _link_subagents(
    index: list[dict[str, Any]], main: list[Event], harness: str
) -> tuple[dict[str, str], dict[str, Any]]:
    """Map sub-agent id -> parent ``tool_use_id``, per-harness. Never guesses.

    * claude-code / kimi-code / grok-cli: nearest preceding spawn call within
      ``LINK_WINDOW_S``, one call per child.
    * codex: ``index.agent_path`` basename == ``spawn_agent`` ``args.task_name``.
    * prime-agent: children come from ``rlm(...)`` inside an ``ipython`` cell —
      match the child's label text against the cell source.
    """
    spawns = [e for e in main if e.type == "tool_use" and e.tool in SPAWN_TOOLS.get(harness, ())]
    linked: dict[str, str] = {}
    used: set[str] = set()
    stats: dict[str, Any] = {"n_subagents": len(index), "n_spawn_calls": len(spawns)}

    if harness == "prime-agent":
        cells = [(e, ((e.args or {}).get("code") or "")) for e in main if e.type == "tool_use"]
        for entry in index:
            probe = TRUNC_MARK.sub("", entry.get("label") or "").strip()[:60]
            t = _parse_ts(entry.get("t_start"))
            if not probe:
                continue
            hits = [
                e
                for e, code in cells
                if probe in code and (t is None or (_parse_ts(e.ts) or t) <= t)
            ]  # last matching cell at or before the child's start: rlm() is re-run
            if hits:
                linked[entry["id"]] = hits[-1].tool_use_id  # type: ignore[assignment]
        stats["method"] = "rlm-label-in-cell"
    elif harness == "codex":
        by_task: dict[str, list[Event]] = {}
        for e in spawns:
            name = (e.args or {}).get("task_name")
            if name:
                by_task.setdefault(name, []).append(e)
        for entry in index:
            path = entry.get("agent_path")
            cands = by_task.get(path.rsplit("/", 1)[-1], []) if path else []
            free = [e for e in cands if e.tool_use_id not in used]
            if free:
                linked[entry["id"]] = free[0].tool_use_id  # type: ignore[assignment]
                used.add(free[0].tool_use_id)  # type: ignore[arg-type]
        stats["method"] = "agent_path==task_name"
    else:
        pairs: list[tuple[float, str, str]] = []
        for entry in index:
            t = _parse_ts(entry.get("t_start"))
            if t is None:
                continue
            for e in spawns:
                et = _parse_ts(e.ts)
                if et is None or et > t:
                    continue
                d = (t - et).total_seconds()
                if d <= LINK_WINDOW_S:
                    pairs.append((d, entry["id"], e.tool_use_id))  # type: ignore[arg-type]
        for d, sid, tuid in sorted(pairs):
            if sid not in linked and tuid not in used:
                linked[sid] = tuid
                used.add(tuid)
        stats["method"] = "nearest-preceding-spawn"
        stats["window_s"] = LINK_WINDOW_S
    stats["n_linked"] = len(linked)
    return linked, stats


def normalize_progression(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten PI's two record shapes into one list.

    Shape A (339 records) is the validated-record log: ``value`` is train_steps
    as a *string*, with agent-hours/tokens/cost at the time. Shape B (104, the
    "live" runs) is a val-loss snapshot: ``steps``/``val_loss``/``t`` unix.
    """
    out = []
    for r in records:
        if "value" in r:
            ts = r.get("ts")
        else:
            t = r.get("t")
            ts = (
                datetime.fromtimestamp(t, tz=timezone.utc).isoformat().replace("+00:00", "Z")
                if t is not None
                else None
            )
        out.append(
            {
                "step_value": int(r["value"]) if "value" in r else r.get("steps"),
                "val_loss": r.get("val_loss"),
                "at_agent_h": r.get("agent_h"),
                "at_tokens": r.get("tok_at"),
                "at_cost": r.get("cost"),
                "ts": ts,
            }
        )
    return out


def _scratch_files(scratch: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Relative paths of the run's scratchpad. 26 entries (qwen-code + grok-4.6)
    carry only ``name``/``text`` — no ``rel``/``chars``."""
    files = []
    for f in scratch:
        rel = f.get("rel") or f["name"]
        rel = "/".join(p for p in Path(rel).parts if p not in ("/", "..", "."))
        files.append({"path": rel, "chars": f.get("chars", len(f.get("text") or ""))})
    return files


def convert_run(
    raw_root: Path, run_id: str, manifest_run: dict[str, Any]
) -> tuple[list[Event], RunMeta]:
    """Convert one run: main stream first, then each sub-agent stream."""
    tr = _traces(raw_root)
    main_raw = _read_gz(tr / f"events-{run_id}.json.gz")["events"]
    bundle = _read_gz(tr / f"subagents-{run_id}.json.gz")
    scratch = _read_gz(tr / f"scratch-{run_id}.json.gz")
    harness = manifest_run["harness"]

    events = convert_stream(main_raw, run_id, "main")

    index, dropped = _dedupe_index(bundle["index"], bundle["events"])
    index.sort(key=lambda x: (x.get("t_start") is None, x.get("t_start") or "", x["id"]))
    linked, link_stats = _link_subagents(index, events, harness)

    subagents: list[dict[str, Any]] = []
    for entry in index:
        sid = entry["id"]
        agent_id = f"sub-{sid}"
        stream = convert_stream(bundle["events"].get(sid, []), run_id, agent_id)
        parent = linked.get(sid)
        if parent and stream:
            stream[0].parent_tool_use = parent
        events.extend(stream)
        record = asdict(
            SubAgent(
                id=agent_id,
                label=entry.get("label"),
                parent_tool_use=parent,
                t_start=entry.get("t_start"),
                t_end=entry.get("t_end"),
                n_events=len(stream),
            )
        )
        record.update(
            (k, entry[k])
            for k in ("n_thinking", "n_tool_use", "paper_calls", "cot", "parent_id", "agent_path")
            if k in entry
        )
        subagents.append(record)

    econ = manifest_run.get("economics") or {}
    t0, t1 = _parse_ts(manifest_run.get("t_start")), _parse_ts(manifest_run.get("t_end"))
    counts = summarize(events)
    main_counts = summarize([e for e in events if e.agent_id == "main"])

    meta = RunMeta(
        run_id=run_id,
        source=SOURCE,
        benchmark=BENCHMARK,
        task_id=manifest_run.get("track"),
        model=manifest_run.get("model"),
        harness=harness,
        budget=dict(NODE),
        t_start=manifest_run.get("t_start"),
        t_end=manifest_run.get("t_end"),
        duration_s=(t1 - t0).total_seconds() if t0 and t1 else None,
        final_score={
            "metric": "train_steps",
            "value": manifest_run.get("best_record"),
            "direction": "lower",
            "baseline": manifest_run.get("baseline"),
            "reference": 2600,  # manifest["human_record"], constant across the release
        },
        # USAGE_KEYS spelling, so the index reads the same columns here as for
        # every other source. 16 runs carry no per-call usage at all; for those
        # the manifest's proxy-side out_tok is the only number there is, and it
        # counts API calls rather than events. The manifest's own totals stay
        # verbatim in extra["economics"].
        tokens=counts["tokens"] or ({"out": econ["out_tok"]} if econ.get("out_tok") else None),
        cost_usd=manifest_run.get("cost_usd"),
        n_events=len(events),
        n_by_type=counts["n_by_type"],
        n_by_origin=counts["n_by_origin"],
        tools=counts["tools"],
        subagents=subagents,
        # Verdicts only. `fidelity` describes how complete the published trace is
        # ("full" is the good value), so it is provenance, not a problem report.
        flags={
            "validity": manifest_run.get("validity"),
            "flagged_why": manifest_run.get("flagged_why"),
        },
        source_paths={
            "events": f"traces/events-{run_id}.json.gz",
            "subagents": f"traces/subagents-{run_id}.json.gz",
            "scratch": f"traces/scratch-{run_id}.json.gz",
            "manifest": "traces/manifest.json.gz",
        },
        extra={
            "progression": normalize_progression(manifest_run.get("progression") or []),
            "note": manifest_run.get("note"),
            "track": manifest_run.get("track"),
            "effort": manifest_run.get("effort"),
            "n_records": manifest_run.get("n_records"),
            "scratch_files": _scratch_files(scratch),
            "label": manifest_run.get("label"),
            "backend": manifest_run.get("backend"),
            "model_id": manifest_run.get("model_id"),
            "model_family": manifest_run.get("model_family"),
            "seed": manifest_run.get("seed"),
            "outcome": manifest_run.get("outcome"),
            "delta": manifest_run.get("delta"),
            "cot_available": manifest_run.get("cot_available"),
            "agent_h": manifest_run.get("agent_h"),
            "economics": econ,
            "tokens_from_events": counts["tokens"],
            "tokens_source": "events" if counts["tokens"] else "manifest_economics",
            "n_main_events": main_counts["n_events"],
            "n_sub_events": len(events) - main_counts["n_events"],
            "subagent_linkage": link_stats,
            "dropped_index_entries": dropped,
        },
    )
    return events, meta


def write_scratch(raw_root: Path, run_id: str, out_dir: Path) -> list[str]:
    """Write the run's scratchpad under ``<run_id>.scratch/``; return the paths.

    thread.md runs to 183 kB and one scratch file to 1.4 MB, so these never go
    into an Event.
    """
    scratch = _read_gz(_traces(raw_root) / f"scratch-{run_id}.json.gz")
    base = out_dir / f"{run_id}.scratch"
    written = []
    for f, rec in zip(scratch, _scratch_files(scratch)):
        dest = base / rec["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(f.get("text") or "", encoding="utf-8")
        written.append(rec["path"])
    return written


def convert_all(raw_root: Path, out_dir: Path, limit: int | None = None) -> list[RunMeta]:
    """Convert every run in the manifest into ``out_dir``."""
    manifest = load_manifest(raw_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for mr in manifest["runs"][:limit]:
        events, meta = convert_run(raw_root, mr["run"], mr)
        write_scratch(raw_root, mr["run"], out_dir)
        write_run(events, meta, out_dir)
        metas.append(meta)
    return metas
