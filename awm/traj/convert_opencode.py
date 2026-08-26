"""OpenCode ``run --format json`` -> v0 events.

One JSON object per line, ``{"type", "timestamp", "sessionID", "part"}``, plus a
rare ``{"type": "error", ...}`` that carries no part. Measured over the 359
published opencode runs (89,396 JSON lines): tool_use 30,367, step_start 27,366,
step_finish 27,337, text 4,271, error 5. The top-level ``type`` and
``part.type`` say the same thing in two spellings (``step_start`` /
``step-start``); the top-level one is read.

Upstream quirks worth recording:

*   A *step* is one API response, and ``part.messageID`` is its identity: every
    part of a step carries it, so a turn is a message id here exactly as it is
    in Claude Code. Bracketing turns with the ``step_start`` line instead would
    lose the 29 steps whose ``step_finish`` never arrived because the run was
    killed mid-response.
*   ``part.id`` is unique within a run; ``part.callID`` is NOT. The kimi runs
    re-emit ``functions.bash:0`` call after call — 1,950 colliding
    re-emissions — so the tool_use id is ``part.id`` and ``callID`` is kept
    beside it rather than used to link a result to its call.
*   ``step_finish.cost`` is PER STEP, not cumulative: 263 of the 319 costed runs
    are non-monotone, so the run's cost is the SUM. Claude Code's
    ``total_cost_usd`` is cumulative and the run's cost is the LAST one. The two
    conventions are opposite and neither may be copied from the other.
*   ``tokens.total`` disagrees with its own parts and is dropped. Across the
    corpus it equals in+out+cache_read 7,456 times, in+out+reasoning+cache_read
    6,653, in+out+reasoning 1,057 and in+out 795 — four different formulas. The
    four components are self-consistent and are what the events carry.
*   463 steps finish having emitted neither text nor a tool call, so their usage
    has no event to sit on. The run's ``tokens`` and ``cost_usd`` are therefore
    summed from every ``step_finish`` and passed up explicitly, rather than left
    to the sum over events, and ``n_steps_without_events`` records the gap.
*   A ``text`` part with ``synthetic: true`` is the scaffolding's continue
    nudge ("Continue if you have next steps"), not the model talking, so it is
    ``harness`` origin and must not be counted as a decision.
*   ``tool: "invalid"`` (54 calls) is OpenCode refusing a malformed tool call
    from the model. It is a real call that really failed, recorded as one.
*   ``state.metadata.output`` repeats ``state.output`` verbatim 21,356 times of
    22,628, so the metadata copy is dropped and the state one is the result
    text. ``metadata.exit`` is the bash exit code and is ``None`` for 1,176
    calls (killed or timed out), where ``is_error`` stays unknown rather than
    being called a success.
*   ``state.status`` is ``completed`` on all 30,367 calls in this release: no
    run captured a pending or errored call. The other branches are written
    anyway because the field exists and a future release can use it.
*   The ``task`` tool (49 calls) spawns a sub-agent whose stream is NOT in this
    file — only its session id, in the tool metadata. There are no sub-agent
    events to record, so none are invented.
"""

from __future__ import annotations

from typing import Any, Iterable

from awm.traj.posttrainbench import LineRow, compact, event_kind, iso_from_ms, number_events
from awm.traj.schema import MAIN_AGENT, USAGE_KEYS, Event

#: OpenCode token spelling -> canonical schema keys. ``cache`` is nested.
_USAGE_KEYS = {"input": "in", "output": "out", "reasoning": "reasoning_out"}
_CACHE_KEYS = {"read": "cache_read", "write": "cache_write"}


def map_usage(tokens: Any) -> dict[str, int] | None:
    """``{"input", "output", "reasoning", "cache": {"read", "write"}}`` -> schema keys.

    ``total`` is deliberately not read: see the module docstring.
    """
    if not isinstance(tokens, dict):
        return None
    out = {v: int(tokens[k]) for k, v in _USAGE_KEYS.items() if isinstance(tokens.get(k), int)}
    cache = tokens.get("cache")
    if isinstance(cache, dict):
        out.update(
            {v: int(cache[k]) for k, v in _CACHE_KEYS.items() if isinstance(cache.get(k), int)}
        )
    return out or None


def _is_error(tool: str, status: Any, metadata: dict[str, Any]) -> bool | None:
    """Did the call fail? ``None`` where the stream does not say."""
    if tool == "invalid" or status == "error":
        return True
    exit_code = metadata.get("exit")
    if isinstance(exit_code, int):
        return exit_code != 0
    return None


def convert(rows: Iterable[LineRow], run_id: str) -> tuple[list[Event], dict[str, Any]]:
    """Convert a ``solve_out.txt`` line stream into events plus a RunMeta extra bag."""
    events: list[Event] = []
    turn = -1
    turn_first: int | None = None
    message_id: str | None = None

    session_ids: list[str] = []
    step_costs: list[float] = []
    step_reasons: dict[str, int] = {}
    totals = {k: 0 for k in USAGE_KEYS}
    errors: list[dict[str, Any]] = []
    unknown_kinds: dict[str, int] = {}
    n_steps = 0
    n_steps_without_events = 0
    n_synthetic = 0

    def add(**kw: Any) -> Event:
        nonlocal turn_first
        e = Event(run_id=run_id, agent_id=MAIN_AGENT, i=0, turn=turn if turn >= 0 else None, **kw)
        if turn_first is None:
            turn_first = len(events)
        events.append(e)
        return e

    def open_step(mid: Any) -> None:
        """A new ``messageID`` is a new API response, and so a new turn."""
        nonlocal turn, turn_first, message_id
        if not isinstance(mid, str) or mid == message_id:
            return
        message_id = mid
        turn += 1
        turn_first = None

    for _ts, obj, lineno, _raw in rows:
        kind = event_kind(obj)
        if kind is None or obj is None:
            continue  # not JSON, or JSON that is not one of OpenCode's events
        ref = {"file": "solve_out.txt", "line": lineno}
        line_ts = iso_from_ms(obj.get("timestamp"))
        sid = obj.get("sessionID")
        if isinstance(sid, str) and sid not in session_ids:
            session_ids.append(sid)
        part = obj.get("part") if isinstance(obj.get("part"), dict) else {}
        open_step(part.get("messageID"))

        if kind == "step_start":
            # The turn boundary is the whole content of the line.
            continue

        if kind == "step_finish":
            n_steps += 1
            reason = part.get("reason")
            if isinstance(reason, str):
                step_reasons[reason] = step_reasons.get(reason, 0) + 1
            cost = part.get("cost")
            if isinstance(cost, (int, float)):
                step_costs.append(float(cost))
            usage = map_usage(part.get("tokens"))
            if usage:
                for k, v in usage.items():
                    totals[k] += v
            if turn_first is None:
                n_steps_without_events += 1
            elif usage:
                events[turn_first].usage = usage
            continue

        if kind == "text":
            synthetic = bool(part.get("synthetic"))
            n_synthetic += synthetic
            add(
                type="text",
                role="user" if synthetic else "assistant",
                origin="harness" if synthetic else "agent",
                text=part.get("text"),
                ts=iso_from_ms((part.get("time") or {}).get("start")) or line_ts,
                source_ref=ref,
                extra={"part_id": part.get("id")},
            )
            continue

        if kind == "tool_use":
            state = part.get("state") if isinstance(part.get("state"), dict) else {}
            metadata = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}
            tool = part.get("tool") or "unknown"
            times = state.get("time") if isinstance(state.get("time"), dict) else {}
            call = part.get("id")
            add(
                type="tool_use",
                role="assistant",
                tool=tool,
                args=state.get("input"),
                tool_use_id=call,
                ts=iso_from_ms(times.get("start")) or line_ts,
                source_ref=ref,
                extra={"call_id": part.get("callID"), "title": state.get("title")},
            )
            add(
                type="tool_result",
                role="user",
                tool=tool,
                text=state.get("output"),
                parent_tool_use=call,
                ts=iso_from_ms(times.get("end")) or line_ts,
                truncated=bool(metadata.get("truncated")),
                is_error=_is_error(tool, state.get("status"), metadata),
                source_ref=ref,
                extra={
                    "status": state.get("status"),
                    "result_meta": compact(metadata, drop=("output",)),
                },
            )
            continue

        if kind == "error":
            err = obj.get("error") if isinstance(obj.get("error"), dict) else {}
            data = err.get("data") if isinstance(err.get("data"), dict) else {}
            message = data.get("message")
            errors.append({"line": lineno, "name": err.get("name"), "message": message})
            add(
                type="text",
                role="user",
                origin="harness",
                ts=line_ts,
                text=message if isinstance(message, str) else None,
                source_ref=ref,
                extra={"kind": "error", "error_name": err.get("name")},
            )
            continue

        # No line type may leave without an event.
        unknown_kinds[kind] = unknown_kinds.get(kind, 0) + 1
        add(type="text", role="user", origin="harness", ts=line_ts, source_ref=ref,
            extra={"kind": kind, "line": compact(obj)})

    extra: dict[str, Any] = {
        "session_ids": session_ids,
        "n_steps": n_steps,
        "n_steps_without_events": n_steps_without_events,
        "n_synthetic_prompts": n_synthetic,
        "step_reasons": step_reasons,
        # Per-step, in order: the whole spend curve for ~5 kB, and its sum is the
        # run cost. Keeping only the sum would make a cost spike unrecoverable.
        "step_costs": step_costs,
        "errors": errors,
        "unknown_line_kinds": unknown_kinds,
    }
    if step_costs:
        extra["cost_usd"] = sum(step_costs)
    tokens = {k: v for k, v in totals.items() if v}
    if tokens:
        extra["tokens"] = tokens
    return number_events(events), extra
