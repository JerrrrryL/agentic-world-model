"""Codex ``exec --json`` -> v0 events.

The stream is ``thread.started``, ``turn.started``, a long run of
``item.started`` / ``item.updated`` / ``item.completed``, then
``turn.completed``. Measured on the PostTrainBench gsm8k / gpt-5.4 sample
(290 JSON lines of 300): 1 thread, 1 turn, 65 started, 2 updated, 220 completed.

Upstream quirks worth recording:

*   The three ``item.*`` messages of one item share an ``item.id`` and describe
    the SAME item, so they must be collapsed. The 220 ``item.completed`` are the
    real item count: command_execution 63, reasoning 106, agent_message 41,
    file_change 8, todo_list 1, web_search 1. (Counting messages instead gives
    the inflated "126 command executions" — that is 63 starts plus 63
    completions of the same 63 commands.)
*   A command's ``item.started`` is emitted immediately and its
    ``item.completed`` can be many lines later, after other items started, so
    the ``tool_use`` is placed where the command was launched and the
    ``tool_result`` where it returned. Both are filled from the completed state,
    which is the only one carrying ``exit_code`` and the output.
*   Whether a line carries a timestamp is a property of the run, not of the
    format: the gsm8k / gpt-5.4 sample has none, but 24 of the 44 codex runs in
    the fetched batch have the launcher's ``[ISO] `` prefix on every line. The
    prefix is used when present and nothing is invented when it is not.
*   ``turn.completed`` holds the run's only usage, and **it is cumulative over
    the thread, not per turn**. 430 runs have a single turn, where the two are
    the same thing; the 76 ``reprompt`` runs have two to six, one thread id
    each, and their counts are monotone (output and reasoning on all 76, input
    and cache on 75 — one thread compacted and went down). So the turn's own
    usage is the DELTA, floored at zero, and the run's total is the LAST line;
    summing what the lines say counts a 6-turn run about four times over.
    Whether the usage of a harness is per step or cumulative has to be measured
    per harness — opencode's is per step, Claude Code's cost is cumulative
    while its tokens are per session, and this one is cumulative throughout.
*   A line can also be a bare ``{"type": "error", "message": "Reconnecting…"}``
    when the response stream drops (3 of the 82 fetched runs). It is the CLI
    talking, not the agent, so it becomes a harness event rather than vanishing.
*   Tool names are the codex item types (``command_execution``, ``file_change``,
    ``web_search``, ``todo_list``) rather than invented ones: codex does not
    name its tools in the stream.
*   Not every JSON object on the stream is codex's. Two runs have a
    ``generation_config.json`` blob printed there by a training subprocess; it
    has no ``type`` and is skipped by ``event_kind``, which is what stops it
    raising ``KeyError`` in the middle of an otherwise complete run.
"""

from __future__ import annotations

from typing import Any, Iterable

from awm.traj.posttrainbench import LineRow, event_kind, number_events
from awm.traj.schema import MAIN_AGENT, Event

_USAGE_KEYS = {
    "input_tokens": "in",
    "output_tokens": "out",
    "cached_input_tokens": "cache_read",
    "reasoning_output_tokens": "reasoning_out",
}

#: Item types that are a tool call, with the item fields that form their args.
_TOOL_ARGS = {
    "command_execution": ("command",),
    "file_change": ("changes",),
    "web_search": ("query", "action"),
    "todo_list": ("items",),
}


def map_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if not usage:
        return None
    out = {v: int(usage[k]) for k, v in _USAGE_KEYS.items() if isinstance(usage.get(k), int)}
    return out or None


def _delta(total: dict[str, int], prev: dict[str, int] | None) -> dict[str, int] | None:
    """One turn's own usage, out of the thread's running total."""
    if prev is None:
        return dict(total)
    # One thread's input count falls between turns — the CLI compacted it — and
    # a turn that consumed a negative number of tokens is not a thing to record.
    out = {k: max(0, v - prev.get(k, 0)) for k, v in total.items()}
    return {k: v for k, v in out.items() if v} or None


def _args(item: dict[str, Any]) -> dict[str, Any] | None:
    fields = _TOOL_ARGS[item["type"]]
    out = {k: item[k] for k in fields if item.get(k) is not None}
    return out or None


def convert(rows: Iterable[LineRow], run_id: str) -> tuple[list[Event], dict[str, Any]]:
    """Convert a ``solve_out.txt`` line stream into events plus a RunMeta extra bag."""
    events: list[Event] = []
    calls: dict[str, Event] = {}
    completed: set[str] = set()
    thread_id: str | None = None
    turn = -1
    turn_first: int | None = None
    turns: list[dict[str, Any]] = []
    #: The thread's running total, i.e. the last ``turn.completed`` seen.
    run_tokens: dict[str, int] | None = None

    def add(**kw: Any) -> Event:
        nonlocal turn_first
        e = Event(run_id=run_id, agent_id=MAIN_AGENT, i=0, turn=turn if turn >= 0 else None, **kw)
        if turn_first is None:
            turn_first = len(events)
        events.append(e)
        return e

    for ts, obj, lineno, _raw in rows:
        kind = event_kind(obj)
        if kind is None or obj is None:
            continue  # not JSON, or JSON that is not one of codex's events
        ref = {"file": "solve_out.txt", "line": lineno}

        if kind == "thread.started":
            thread_id = obj.get("thread_id")

        elif kind == "turn.started":
            turn += 1
            turn_first = None
            turns.append({"index": turn, "started_line": lineno})

        elif kind == "turn.completed":
            total = map_usage(obj.get("usage"))
            usage = _delta(total, run_tokens) if total else None
            if total:
                run_tokens = total
            if turn_first is not None and usage:
                events[turn_first].usage = usage
            if turns:
                turns[-1].update(
                    {"completed_line": lineno, "usage": usage, "usage_cumulative": total}
                )

        elif kind.startswith("item."):
            item = obj["item"]
            itype = item["type"]
            iid = item["id"]

            if itype == "command_execution":
                if kind == "item.started" and iid not in calls:
                    calls[iid] = add(type="tool_use", role="assistant", tool=itype, ts=ts,
                                     args=_args(item), tool_use_id=iid, source_ref=ref)
                elif kind == "item.completed":
                    call = calls.get(iid)
                    if call is None:
                        call = add(type="tool_use", role="assistant", tool=itype, ts=ts,
                                   args=_args(item), tool_use_id=iid, source_ref=ref)
                        calls[iid] = call
                    call.args = _args(item)
                    completed.add(iid)
                    exit_code = item.get("exit_code")
                    add(type="tool_result", role="user", text=item.get("aggregated_output"),
                        tool=itype, parent_tool_use=iid, ts=ts,
                        is_error=None if exit_code is None else exit_code != 0,
                        source_ref=ref,
                        extra={"exit_code": exit_code, "status": item.get("status")})
                continue

            if kind != "item.completed":
                continue
            completed.add(iid)
            if itype == "reasoning":
                add(type="thinking", role="assistant", text=item.get("text"), ts=ts,
                    source_ref=ref)
            elif itype == "agent_message":
                add(type="text", role="assistant", text=item.get("text"), ts=ts,
                    source_ref=ref)
            elif itype in _TOOL_ARGS:
                add(type="tool_use", role="assistant", tool=itype, args=_args(item), ts=ts,
                    tool_use_id=iid, source_ref=ref,
                    extra={"status": item["status"]} if item.get("status") else None)
            else:
                add(type="text", role="assistant", ts=ts, source_ref=ref,
                    extra={"kind": itype, "item": item})

        else:
            msg = obj.get("message")
            add(type="text", role="user", origin="harness", ts=ts, source_ref=ref,
                text=msg if isinstance(msg, str) else None, extra={"kind": kind})

    unfinished = sorted(iid for iid in calls if iid not in completed)
    extra: dict[str, Any] = {
        "thread_id": thread_id,
        "turns": turns,
        "n_turns": len(turns),
        "unfinished_items": unfinished,
        "tokens_source": "turn.completed" if run_tokens else "none",
    }
    if run_tokens:
        extra["tokens"] = run_tokens
    return number_events(events), extra
