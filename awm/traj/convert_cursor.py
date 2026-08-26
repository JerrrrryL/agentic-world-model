"""Cursor CLI ``--output-format stream-json`` -> v0 events.

Cursor names four of its line types exactly as Claude Code does — ``system``,
``assistant``, ``user``, ``result`` — and opens with
``{"type": "system", "subtype": "init", ...}``, which is why sniffing one object
sent all 56 of these runs to the Claude Code converter and why they must be
detected before it. What it does NOT share is where the work is: ``thinking``
and ``tool_call``, 128k of the 133k lines, and neither has a Claude Code
counterpart. Measured over the 56 published cursor runs: thinking/delta 98,719,
tool_call/started 10,623, tool_call/completed 10,292, thinking/completed 8,301,
system/task_notification 1,677, assistant 1,532, connection and retry 712 each,
interaction_query 392, system/init 56, user 56, result 48.

Upstream quirks worth recording:

*   Thinking is delta-streamed and the text is on the DELTA lines, at the top
    level — 98,719 deltas carry ``text`` and the 8,301 ``completed`` markers
    carry none. Deltas are joined into one event per block. A block is also
    flushed when any other line arrives, so a thinking event still lands before
    the tool call it reasoned about even if its ``completed`` never came.
*   ``result`` holds the run's only usage, in camelCase (``inputTokens`` and
    friends), and there is exactly one per run — 48 of the 56 runs; the other 8
    were killed before it. It is written onto the run's first turn-bearing
    event, as the schema requires.
*   Turns are ``model_call_id``, which is on the assistant and tool_call lines
    but NOT on thinking — and a response streams its thinking first, so at the
    time a thinking event is built its turn is not yet known. Turns are
    therefore numbered in a second pass, once the whole stream is in hand: a
    turn is one distinct ``model_call_id`` in order of first appearance, and an
    agent event that names none belongs to the next response, not the last one.
    Deciding it inline would also have mis-split parallel tool calls, whose
    completions can arrive out of order.
*   A ``toolCallId`` can be emitted more than once: 331 excess ``started`` lines
    against 356 reconnects, plus 32 ids completed twice. Those are the CLI
    replaying the transcript after ``connection``/``retry``, not the command
    running again, so the first of each is kept and the repeats are counted into
    ``n_replayed_tool_lines``.
*   The tool is named by the key, not a field: ``tool_call`` holds one
    ``<name>ToolCall`` object (shell, await, read, edit, grep, webSearch,
    updateTodos, glob, webFetch, delete). The outcome is likewise the key of the
    ``result`` object — ``success``, ``failure``, ``error`` or ``spawnError`` —
    which is what ``is_error`` reads. A shell ``success`` can still carry a
    non-zero ``exitCode``, and that counts as an error too.
*   ``shell`` results are read from ``interleavedOutput`` in preference to
    ``stdout``: it is stdout and stderr in the order the model saw them, and it
    is set on 30 results where ``stdout`` is empty.
*   ``shellToolCall.args.parsingResult`` is dropped. It is a re-derivation of
    ``command`` as an AST and it is 13.6 MB of the 37 MB of tool arguments in
    this release — a third of the payload, restating a string that is kept.
*   ``user`` appears once per run, on the first or second line, and is the task
    prompt: unlike ``claude --print``, cursor echoes what it was given. It is
    the only harness in this corpus that publishes the prompt, so it is
    ``human`` origin — the benchmark's instruction, not scaffolding noise. All
    56 runs put it in a one-element ``[{"type": "text"}]`` list and none in a
    bare string, so reading only the string form (as the identically-named
    Claude Code line needs) dropped ~4 KB of instruction from every run.
*   ``interaction_query`` is the CLI's own approval handshake for web search and
    web fetch (196 requests, 196 approvals). ``connection``/``retry`` are it
    reconnecting mid-run. All are harness events; dropping them would erase the
    fact that a run spent time disconnected.
"""

from __future__ import annotations

from typing import Any, Iterable

from awm.traj.posttrainbench import LineRow, compact, event_kind, iso_from_ms, number_events
from awm.traj.schema import MAIN_AGENT, Event

#: Cursor usage spelling -> canonical schema keys.
_USAGE_KEYS = {
    "inputTokens": "in",
    "outputTokens": "out",
    "cacheReadTokens": "cache_read",
    "cacheWriteTokens": "cache_write",
}

_TOOL_SUFFIX = "ToolCall"

#: Outcome keys, in the order a result is tested for one.
_OUTCOMES = ("success", "failure", "error", "spawnError")

#: Where each tool puts the text the model read back, best field first.
_RESULT_TEXT: dict[str, tuple[str, ...]] = {
    "shell": ("interleavedOutput", "stdout"),
    "read": ("content",),
    "webFetch": ("markdown",),
    "edit": ("diffString",),
}

#: Result fields kept out of the extra bag because they restate something else.
#: The two whole-file snapshots around an edit are together several times the
#: size of the diff that is already the event's text.
_RESULT_DROP: dict[str, tuple[str, ...]] = {
    "edit": ("afterFullFileContent", "beforeFullFileContent"),
}

#: Argument fields that are derived from another argument, not input.
_ARG_DROP: dict[str, tuple[str, ...]] = {"shell": ("parsingResult",)}


def _without(d: Any, drop: tuple[str, ...]) -> dict[str, Any] | None:
    """A tool's arguments or result, minus fields that restate another field.

    Nothing here is elided by size. These are the agent's own bytes — the
    content of an edit, the matches a grep returned — and a converter that
    summarised them would be deciding what the analysis layer gets to see.
    """
    if not isinstance(d, dict):
        return None
    out = {k: v for k, v in d.items() if k not in drop}
    return out or None


def map_usage(usage: Any) -> dict[str, int] | None:
    if not isinstance(usage, dict):
        return None
    out = {v: int(usage[k]) for k, v in _USAGE_KEYS.items() if isinstance(usage.get(k), int)}
    return out or None


def _tool_of(tool_call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """``{"shellToolCall": {...}}`` -> ``("shell", {...})``."""
    for key, body in tool_call.items():
        if key.endswith(_TOOL_SUFFIX) and isinstance(body, dict):
            return key[: -len(_TOOL_SUFFIX)], body
    return "unknown", {}


def _outcome(body: dict[str, Any]) -> tuple[str | None, dict[str, Any], dict[str, Any]]:
    """``(outcome, payload, siblings)`` out of a tool call's ``result``.

    A shell result carries scalars (``isBackground``, ``terminalsFolder``)
    alongside the outcome object, and 422 of them carry nothing else at all.
    """
    result = body.get("result")
    if not isinstance(result, dict):
        return None, {}, {}
    siblings = {k: v for k, v in result.items() if k not in _OUTCOMES}
    for key in _OUTCOMES:
        payload = result.get(key)
        if isinstance(payload, dict):
            return key, payload, siblings
    return None, {}, siblings


def _result_text(tool: str, payload: dict[str, Any]) -> tuple[str | None, tuple[str, ...]]:
    """The text the model read back, and the fields it came from."""
    fields = _RESULT_TEXT.get(tool, ())
    for field in fields:
        value = payload.get(field)
        if isinstance(value, str) and value:
            return value, fields
    error = payload.get("error")
    if isinstance(error, str) and error:
        return error, fields
    return None, fields


def _prompt_text(content: Any) -> tuple[str | None, list[Any] | None]:
    """The task prompt cursor echoes back, and any block that was not text.

    All 56 published runs put it in a one-element ``[{"type": "text", ...}]``,
    never a bare string — so a converter that read only the string form dropped
    the benchmark's whole instruction, ~4 KB, on every cursor run. A string is
    still accepted because Claude Code's identically-named line uses one.
    """
    if isinstance(content, str):
        return content or None, None
    if isinstance(content, list):
        texts = [
            b["text"]
            for b in content
            if isinstance(b, dict) and b.get("type") == "text" and isinstance(b.get("text"), str)
        ]
        rest = [b for b in content if not (isinstance(b, dict) and b.get("type") == "text")]
        return ("\n".join(texts) or None), (rest or None)
    return None, None


def _is_error(outcome: str | None, payload: dict[str, Any]) -> bool | None:
    if outcome in ("failure", "error", "spawnError"):
        return True
    if outcome == "success":
        exit_code = payload.get("exitCode")
        return exit_code != 0 if isinstance(exit_code, int) else False
    return None


def number_turns(events: list[Event], model_calls: list[str | None]) -> int:
    """Number each event's turn from the ``model_call_id`` its line named.

    A turn is one distinct id, in order of first appearance. An event whose line
    named none takes the turn of the NEXT event that does when it is the agent's
    (thinking streams ahead of the response it belongs to) and of the previous
    one when it is the harness's (a reconnect belongs to the response it
    interrupted). Returns the number of turns.
    """
    order: dict[str, int] = {}
    for mid in model_calls:
        if mid is not None and mid not in order:
            order[mid] = len(order)
    ahead: list[int | None] = [None] * len(events)
    nxt: int | None = None
    for idx in range(len(events) - 1, -1, -1):
        mid = model_calls[idx]
        if mid is not None:
            nxt = order[mid]
        ahead[idx] = nxt
    prev: int | None = None
    for idx, e in enumerate(events):
        mid = model_calls[idx]
        if mid is not None:
            prev = order[mid]
            e.turn = prev
        elif e.origin == "agent":
            e.turn = ahead[idx] if ahead[idx] is not None else prev
        else:
            e.turn = prev
    return len(order)


def convert(rows: Iterable[LineRow], run_id: str) -> tuple[list[Event], dict[str, Any]]:
    """Convert a ``solve_out.txt`` line stream into events plus a RunMeta extra bag."""
    events: list[Event] = []
    model_calls: list[str | None] = []

    thinking: list[str] = []
    thinking_ts: str | None = None
    thinking_line: int | None = None

    seen_tool_lines: set[tuple[str, str]] = set()
    tool_names: dict[str, str] = {}
    session_ids: list[str] = []
    result: dict[str, Any] | None = None
    reconnects = 0
    retries = 0
    unknown_kinds: dict[str, int] = {}
    n_replayed = 0
    init: dict[str, Any] = {}

    def add(mcid: Any = None, **kw: Any) -> Event:
        e = Event(run_id=run_id, agent_id=MAIN_AGENT, i=0, **kw)
        events.append(e)
        model_calls.append(mcid if isinstance(mcid, str) else None)
        return e

    def flush_thinking(ref: dict[str, Any]) -> None:
        """Emit the accumulated deltas as one thinking event, at their own line."""
        nonlocal thinking, thinking_ts, thinking_line
        if not thinking:
            return
        text, ts, line = "".join(thinking), thinking_ts, thinking_line
        thinking, thinking_ts, thinking_line = [], None, None
        where = {"file": "solve_out.txt", "line": line} if line else ref
        add(type="thinking", role="assistant", text=text or None, ts=ts, source_ref=where)

    for _ts, obj, lineno, _raw in rows:
        kind = event_kind(obj)
        if kind is None or obj is None:
            continue  # not JSON, or JSON that is not one of cursor's events
        ref = {"file": "solve_out.txt", "line": lineno}
        line_ts = iso_from_ms(obj.get("timestamp_ms"))
        sub = obj.get("subtype")
        sid = obj.get("session_id")
        if isinstance(sid, str) and sid not in session_ids:
            session_ids.append(sid)

        if kind == "thinking":
            if sub == "delta":
                if not thinking:
                    thinking_ts, thinking_line = line_ts, lineno
                text = obj.get("text")
                if isinstance(text, str):
                    thinking.append(text)
            else:
                flush_thinking(ref)
            continue

        flush_thinking(ref)

        if kind == "tool_call":
            tool_call = obj.get("tool_call") if isinstance(obj.get("tool_call"), dict) else {}
            call_id = tool_call.get("toolCallId") or obj.get("call_id")
            if isinstance(call_id, str) and (call_id, str(sub)) in seen_tool_lines:
                n_replayed += 1  # the CLI replaying the transcript after a reconnect
                continue
            if isinstance(call_id, str):
                seen_tool_lines.add((call_id, str(sub)))
            mcid = obj.get("model_call_id")
            tool, body = _tool_of(tool_call)
            if sub == "started":
                tool_names[str(call_id)] = tool
                args = _without(body.get("args"), _ARG_DROP.get(tool, ()))
                add(mcid, type="tool_use", role="assistant", tool=tool, args=args,
                    tool_use_id=call_id, ts=line_ts, source_ref=ref)
                continue
            if tool == "unknown":
                tool = tool_names.get(str(call_id), tool)
            if isinstance(call_id, str) and (call_id, "started") not in seen_tool_lines:
                # A completion whose start never reached the stream (7 of them).
                seen_tool_lines.add((call_id, "started"))
                add(mcid, type="tool_use", role="assistant", tool=tool, tool_use_id=call_id,
                    ts=line_ts, source_ref=ref, extra={"start_line_missing": True})
            outcome, payload, siblings = _outcome(body)
            text, promoted = _result_text(tool, payload)
            add(mcid, type="tool_result", role="user", tool=tool, text=text,
                parent_tool_use=call_id, ts=line_ts,
                is_error=_is_error(outcome, payload), source_ref=ref,
                extra={
                    "outcome": outcome,
                    "result": _without(payload, promoted + _RESULT_DROP.get(tool, ())),
                    "result_meta": compact(siblings) if siblings else None,
                })
            continue

        if kind == "assistant":
            mcid = obj.get("model_call_id")
            message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            content = message.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for block in blocks or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    add(mcid, type="text", role="assistant", text=block.get("text"), ts=line_ts,
                        source_ref=ref)
                else:
                    add(mcid, type="text", role="assistant", ts=line_ts, source_ref=ref,
                        extra={"kind": block.get("type"), "block": compact(block)})
            continue

        if kind == "user":
            message = obj.get("message") if isinstance(obj.get("message"), dict) else {}
            text, rest = _prompt_text(message.get("content"))
            prompt_extra: dict[str, Any] = {"kind": "prompt"}
            if rest:
                prompt_extra["content_blocks"] = compact({"blocks": rest})
            add(type="text", role="user", origin="human", ts=line_ts, text=text,
                source_ref=ref, extra=prompt_extra)
            continue

        if kind == "system" and sub == "init":
            init = {
                "session_id": sid,
                "model": obj.get("model"),
                "cwd": obj.get("cwd"),
                "permission_mode": obj.get("permissionMode"),
                "api_key_source": obj.get("apiKeySource"),
            }
            add(type="text", role="user", origin="harness", ts=line_ts, source_ref=ref,
                extra={"kind": "session_start", **init})
            continue

        if kind == "result":
            result = {
                "subtype": sub,
                "is_error": obj.get("is_error"),
                "duration_ms": obj.get("duration_ms"),
                "duration_api_ms": obj.get("duration_api_ms"),
                "request_id": obj.get("request_id"),
                "usage": map_usage(obj.get("usage")),
                "line": lineno,
            }
            add(type="text", role="user", origin="harness", ts=line_ts,
                text=obj.get("result") if isinstance(obj.get("result"), str) else None,
                is_error=obj.get("is_error"), source_ref=ref,
                extra={"kind": "result", "subtype": sub})
            continue

        # Everything the CLI says about itself: task notifications, reconnects,
        # retries and the web-tool approval handshake. Harness, never dropped.
        reconnects += kind == "connection"
        retries += kind == "retry"
        add(type="text", role="user", origin="harness", ts=line_ts,
            text=obj.get("title") if isinstance(obj.get("title"), str) else None,
            parent_tool_use=obj.get("task_id") if kind == "system" else None,
            source_ref=ref,
            extra=compact({
                "kind": kind if kind != "system" else sub,
                "status": obj.get("status"),
                "detail": obj.get("detail"),
                "attempt": obj.get("attempt"),
                "endpoint_url": obj.get("endpoint_url"),
                "is_resume": obj.get("is_resume"),
                "checkpoint_turn_count": obj.get("checkpoint_turn_count"),
                "query_type": obj.get("query_type"),
                "query": obj.get("query"),
                "response": obj.get("response"),
            }))
        if kind not in ("system", "connection", "retry", "interaction_query"):
            unknown_kinds[kind] = unknown_kinds.get(kind, 0) + 1

    flush_thinking({"file": "solve_out.txt", "line": 0})
    n_turns = number_turns(events, model_calls)

    usage = (result or {}).get("usage")
    if usage:
        # The run's only usage. It belongs on the first event that has a turn to
        # hang it on; the init marker precedes turn 0 and carries none.
        first = next((i for i, e in enumerate(events) if e.turn is not None), None)
        if first is not None:
            events[first].usage = usage

    extra: dict[str, Any] = {
        "session_ids": session_ids,
        "n_turns": n_turns,
        "n_reconnects": reconnects,
        "n_retry_lines": retries,
        "n_replayed_tool_lines": n_replayed,
        "unknown_line_kinds": unknown_kinds,
        "api_model": init.get("model"),
        "permission_mode": init.get("permission_mode"),
        "result": result,
    }
    return number_events(events), extra
