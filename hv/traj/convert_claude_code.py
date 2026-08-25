"""Claude Code ``--print --output-format stream-json`` -> v0 events.

The stream is one JSON object per line: ``system`` (subtypes ``init`` and the
background-task notifications), ``assistant``, ``user``, ``result`` and
``rate_limit_event``. Measured on the PostTrainBench gsm8k / opus-4-8 sample
(652 JSON lines): system 85, assistant 392, user 160, result 13, rate limit 2.

Upstream quirks worth recording:

*   Every ``assistant`` line carries exactly ONE content block, and consecutive
    lines sharing ``message.id`` are one API response — 392 lines, 155 distinct
    ids. ``usage`` is repeated identically on each of those lines, so a turn is
    a message id and the usage is kept on the turn's first event only.
*   The run is one CLI session id but thirteen ``init``/``result`` pairs: the
    launcher restarts the CLI with ``--continue`` whenever it exits (here, when
    a background task reports back). Sessions therefore cannot be told apart by
    ``session_id``; they are numbered by ``init`` order. ``total_cost_usd`` on
    each ``result`` is CUMULATIVE over the whole run, so the run's cost is the
    last one, never the sum.
*   The launcher's re-prompt never appears in the stream — a ``--print`` run
    does not echo its prompt — so no user text event is invented for it. The
    ``init`` line is emitted instead, as a ``harness`` marker with no text.
*   ``system:task_*`` events are background ``local_bash`` tasks the agent
    started with Bash, not sub-agents: they are harness notifications about an
    existing ``tool_use_id``, so they become harness-origin text events, not
    tool calls.
*   Thinking blocks carry a long base64 ``signature``; it is dropped.
*   PI's ``…[+N chars]`` marker never appears here, but Claude Code caps a
    background task's output itself and says so in the result text; those
    results are flagged ``truncated`` (90 of them across the fetched batch, all
    on ``TaskOutput``).
*   Sub-agents (the ``Task`` tool) surface as rows with ``parent_tool_use_id``
    set. The sample has none — all 552 rows are null — but the branch exists
    because the tool is in the run's tool list.
"""

from __future__ import annotations

import re
from typing import Any, Iterable

from hv.traj.posttrainbench import LineRow, number_events
from hv.traj.schema import MAIN_AGENT, Event

#: The only truncation the CLI announces in-band: a background task's output is
#: capped and the rest written to a file.
_TRUNCATED = re.compile(r"Output truncated \(\d+KB total\)\. Full output saved to: ")

#: Claude usage spelling -> canonical schema keys. ``cache_creation`` (a nested
#: breakdown), ``service_tier`` and ``inference_geo`` are not token counts.
_USAGE_KEYS = {
    "input_tokens": "in",
    "output_tokens": "out",
    "cache_read_input_tokens": "cache_read",
    "cache_creation_input_tokens": "cache_write",
}

_SCALARS = (bool, int, float, str)


def map_usage(usage: dict[str, Any] | None) -> dict[str, int] | None:
    if not usage:
        return None
    out = {v: int(usage[k]) for k, v in _USAGE_KEYS.items() if isinstance(usage.get(k), int)}
    return out or None


def _scalar_meta(d: Any) -> dict[str, Any] | None:
    """Keep the small scalar fields of ``tool_use_result``; its bulk is already
    the tool_result text."""
    if not isinstance(d, dict):
        return None
    out = {
        k: v
        for k, v in d.items()
        if isinstance(v, _SCALARS) and not (isinstance(v, str) and len(v) > 200)
    }
    return out or None


def _result_text(content: Any) -> tuple[str | None, list[Any] | None]:
    """Tool result content is usually a string; it can also be a block list
    (measured: ``tool_reference`` blocks returned by ToolSearch)."""
    if isinstance(content, str):
        return content, None
    if isinstance(content, list):
        texts = [b["text"] for b in content if isinstance(b, dict) and b.get("type") == "text"]
        rest = [b for b in content if not (isinstance(b, dict) and b.get("type") == "text")]
        return ("\n".join(texts) if texts else None), (rest or None)
    return None, None


def convert(rows: Iterable[LineRow], run_id: str) -> tuple[list[Event], dict[str, Any]]:
    """Convert a ``solve_out.txt`` line stream into events plus a RunMeta extra bag."""
    events: list[Event] = []
    sessions: list[dict[str, Any]] = []
    tool_names: dict[str, str] = {}
    session_ids: list[str] = []
    models: list[str] = []
    cli_versions: list[str] = []
    tools_available: list[str] = []
    open_session: dict[str, Any] | None = None
    last_message_id: str | None = None
    usage_seen: set[str] = set()
    # Turns count API responses over the whole run and never reset at a session
    # boundary: the sessions are one continued conversation, not thirteen runs.
    turn = -1

    def add(**kw: Any) -> Event:
        e = Event(run_id=run_id, i=0, **kw)
        events.append(e)
        return e

    for ts, obj, lineno, _raw in rows:
        if obj is None:
            continue
        ref = {"file": "solve_out.txt", "line": lineno}
        kind = obj["type"]
        agent_id = obj.get("parent_tool_use_id") or MAIN_AGENT
        parent = obj.get("parent_tool_use_id")

        if kind == "system":
            sub = obj.get("subtype")
            if sub == "init":
                session_ids.append(obj.get("session_id"))
                if obj.get("model"):
                    models.append(obj["model"])
                if obj.get("claude_code_version"):
                    cli_versions.append(obj["claude_code_version"])
                tools_available = obj.get("tools") or tools_available
                open_session = {
                    "index": len(sessions),
                    "session_id": obj.get("session_id"),
                    "init_line": lineno,
                    "t_start": ts,
                }
                sessions.append(open_session)
                add(
                    agent_id=MAIN_AGENT, type="text", role="user", origin="harness", ts=ts,
                    turn=turn if turn >= 0 else None, source_ref=ref,
                    extra={
                        "kind": "session_start",
                        "session_index": open_session["index"],
                        "session_id": obj.get("session_id"),
                        "model": obj.get("model"),
                        "cwd": obj.get("cwd"),
                        "permission_mode": obj.get("permissionMode"),
                        "cli_version": obj.get("claude_code_version"),
                    },
                )
            else:
                extra = {
                    "kind": sub,
                    "task_id": obj.get("task_id"),
                    "task_type": obj.get("task_type"),
                    "status": obj.get("status"),
                    "patch": obj.get("patch"),
                    "output_file": obj.get("output_file") or None,
                }
                add(
                    agent_id=MAIN_AGENT, type="text", role="user", origin="harness", ts=ts,
                    turn=turn if turn >= 0 else None,
                    text=obj.get("description") or obj.get("summary"),
                    parent_tool_use=obj.get("tool_use_id"), source_ref=ref,
                    extra={k: v for k, v in extra.items() if v is not None},
                )

        elif kind == "rate_limit_event":
            add(
                agent_id=MAIN_AGENT, type="text", role="user", origin="harness", ts=ts,
                turn=turn if turn >= 0 else None, source_ref=ref,
                extra={"kind": "rate_limit_event", "rate_limit_info": obj.get("rate_limit_info")},
            )

        elif kind == "assistant":
            msg = obj["message"]
            mid = msg.get("id")
            if mid != last_message_id:
                last_message_id = mid
                turn += 1
                # A sub-agent's lines can interrupt a parent message and let it
                # resume under a second turn number; its usage is the same
                # payload repeated, so it counts only the first time.
                usage = None if mid in usage_seen else map_usage(msg.get("usage"))
                usage_seen.add(mid)
            else:
                usage = None
            for block in msg.get("content", []):
                btype = block.get("type")
                if btype == "thinking":
                    add(agent_id=agent_id, type="thinking", role="assistant", ts=ts, turn=turn,
                        text=block.get("thinking"), usage=usage, parent_tool_use=parent,
                        source_ref=ref)
                elif btype == "redacted_thinking":
                    add(agent_id=agent_id, type="thinking", role="assistant", ts=ts, turn=turn,
                        redacted=True, usage=usage, parent_tool_use=parent, source_ref=ref)
                elif btype == "text":
                    add(agent_id=agent_id, type="text", role="assistant", ts=ts, turn=turn,
                        text=block.get("text"), usage=usage, parent_tool_use=parent,
                        source_ref=ref)
                elif btype == "tool_use":
                    tool_names[block["id"]] = block["name"]
                    add(agent_id=agent_id, type="tool_use", role="assistant", ts=ts, turn=turn,
                        tool=block["name"], args=block.get("input"), tool_use_id=block["id"],
                        usage=usage, parent_tool_use=parent, source_ref=ref)
                else:
                    add(agent_id=agent_id, type="text", role="assistant", ts=ts, turn=turn,
                        usage=usage, parent_tool_use=parent, source_ref=ref,
                        extra={"kind": btype, "block": block})
                usage = None

        elif kind == "user":
            content = obj["message"].get("content")
            meta = _scalar_meta(obj.get("tool_use_result"))
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content
            for block in blocks or []:
                if block.get("type") == "tool_result":
                    text, rest = _result_text(block.get("content"))
                    extra = {"result_meta": meta} if meta else {}
                    if rest:
                        extra["content_blocks"] = rest
                    add(agent_id=agent_id, type="tool_result", role="user", ts=ts, turn=turn,
                        text=text, tool=tool_names.get(block.get("tool_use_id")),
                        truncated=bool(text and _TRUNCATED.search(text)),
                        is_error=block.get("is_error"),
                        parent_tool_use=block.get("tool_use_id"), source_ref=ref,
                        extra=extra or None)
                else:
                    # A --print run never echoes the operator's prompt, so any
                    # user text here was injected by the launcher.
                    add(agent_id=agent_id, type="text", role="user", origin="harness", ts=ts,
                        turn=turn, text=block.get("text"), parent_tool_use=parent,
                        source_ref=ref)

        elif kind == "result":
            if open_session is None:
                open_session = {"index": len(sessions), "init_line": None}
                sessions.append(open_session)
            open_session.update(
                {
                    "result_line": lineno,
                    "t_end": ts,
                    "subtype": obj.get("subtype"),
                    "is_error": obj.get("is_error"),
                    "num_turns": obj.get("num_turns"),
                    "duration_ms": obj.get("duration_ms"),
                    "duration_api_ms": obj.get("duration_api_ms"),
                    "total_cost_usd": obj.get("total_cost_usd"),
                    "stop_reason": obj.get("stop_reason"),
                    "terminal_reason": obj.get("terminal_reason"),
                }
            )
            open_session = None

    costs = [s["total_cost_usd"] for s in sessions if s.get("total_cost_usd") is not None]
    extra: dict[str, Any] = {
        "sessions": sessions,
        "n_sessions": len(sessions),
        "session_ids": sorted({s for s in session_ids if s}),
        "cli_versions": sorted(set(cli_versions)),
        "api_model": models[0] if models else None,
        "tools_available": tools_available,
        "num_turns_reported": sum(s["num_turns"] for s in sessions if s.get("num_turns")),
        "duration_ms_sum": sum(s["duration_ms"] for s in sessions if s.get("duration_ms")),
    }
    if costs:
        extra["cost_usd"] = costs[-1]
    return number_events(events), extra
