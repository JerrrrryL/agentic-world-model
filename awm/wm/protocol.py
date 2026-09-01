"""The ping protocol: canonical files, the ledger, and the scientist-facing views.

``awm-ping-v1`` files under ``wm/cards/<card>/pings/`` are canonical. The
``inbox.md`` line, the stdout of the blocking command, and the Stop hook are
views onto them. Replies land in ``wm/cards/<card>/replies/`` and are
idempotent: the same choice twice is a no-op, a different choice is an error.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from .schema import (
    BRIEF_OPTIONS,
    DECISION_OPTIONS,
    PING_KINDS,
    PING_SCHEMA,
    REPLY_SCHEMA,
    YIELD_OPTIONS,
    WMError,
    dump_yaml,
    load_yaml,
    now,
)

REPLY_REQUIRED = {"brief": True, "notice": False, "yield_request": True, "decision": True, "question": True}


class Ledger:
    """Append-only ``events.jsonl`` with a monotonic ``seq``."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch()

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        with self.path.open("a+") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            fh.seek(0)
            seq = sum(1 for line in fh if line.strip()) + 1
            row = {"seq": seq, "at": now(), "event": event, **payload}
            fh.write(json.dumps(row, sort_keys=True, default=str) + "\n")
            fcntl.flock(fh, fcntl.LOCK_UN)
        return row

    def rows(self) -> list[dict[str, Any]]:
        out = []
        with self.path.open() as fh:
            for line in fh:
                if line.strip():
                    out.append(json.loads(line))
        return out


class Mailbox:
    """Pings and replies for one card."""

    def __init__(self, card_dir: Path, inbox: Path, ledger: Ledger):
        self.card_dir = card_dir
        self.card_id = card_dir.name
        self.pings_dir = card_dir / "pings"
        self.replies_dir = card_dir / "replies"
        self.inbox = inbox
        self.ledger = ledger
        self.pings_dir.mkdir(parents=True, exist_ok=True)
        self.replies_dir.mkdir(parents=True, exist_ok=True)

    # ---- pings

    def _next_seq(self) -> int:
        return len(list(self.pings_dir.glob("p-*.yaml"))) + 1

    def send(
        self,
        kind: str,
        summary: str,
        *,
        evidence: list[dict[str, Any]] | None = None,
        prediction: dict[str, Any] | None = None,
        options: list[dict[str, Any]] | None = None,
        timeout_action: dict[str, Any] | None = None,
        raised_by: str | None = None,
        observation: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if kind not in PING_KINDS:
            raise WMError(f"unknown ping kind {kind}")
        required = REPLY_REQUIRED[kind]
        if required and not options:
            raise WMError(f"{kind} pings need options")
        if required and not timeout_action:
            raise WMError(f"{kind} pings need a timeout_action")
        seq = self._next_seq()
        ping_id = f"p-{seq}"
        ping = {
            "schema_version": PING_SCHEMA,
            "ping_id": ping_id,
            "card_id": self.card_id,
            "seq": seq,
            "kind": kind,
            "created_at": now(),
            "summary": summary,
            "evidence": evidence or [],
            "prediction": prediction,
            "reply_required": required,
            "options": options or [],
            "timeout_action": timeout_action,
            "raised_by": raised_by,
            "observation": observation,
        }
        if extra:
            ping.update(extra)
        path = self.pings_dir / f"{ping_id}.yaml"
        dump_yaml(path, ping)
        self._inbox_line(ping, path)
        self.ledger.append("ping", card_id=self.card_id, ping_id=ping_id, kind=kind,
                           reply_required=required, raised_by=raised_by, path=str(path))
        return ping

    def _inbox_line(self, ping: dict[str, Any], path: Path) -> None:
        flag = "REPLY NEEDED" if ping["reply_required"] else "fyi"
        line = f"- [{ping['created_at']}] {self.card_id} {ping['ping_id']} `{ping['kind']}` ({flag}) — {ping['summary']}  →  {path}\n"
        with self.inbox.open("a") as fh:
            fh.write(line)

    def ping(self, ping_id: str) -> dict[str, Any]:
        return load_yaml(self.pings_dir / f"{ping_id}.yaml")

    def pings(self) -> list[dict[str, Any]]:
        out = [load_yaml(p) for p in self.pings_dir.glob("p-*.yaml")]
        return sorted(out, key=lambda p: p["seq"])

    def pending(self) -> list[dict[str, Any]]:
        return [p for p in self.pings() if p["reply_required"] and self.reply(p["ping_id"]) is None]

    # ---- replies

    def reply(self, ping_id: str) -> dict[str, Any] | None:
        path = self.replies_dir / f"{ping_id}.yaml"
        return load_yaml(path) if path.is_file() else None

    def record_reply(self, ping_id: str, choice: str, why: str | None = None,
                     amend: str | None = None, answer: str | None = None,
                     answers: dict[str, Any] | None = None) -> dict[str, Any]:
        ping = self.ping(ping_id)
        if not ping["reply_required"]:
            raise WMError(f"{ping_id} is a {ping['kind']}; it takes no reply")
        if ping["kind"] == "question" and (choice != "answer" or not (answer or answers)):
            raise WMError(f"{ping_id} is a question; reply with --answer \"...\" or --answer-file FILE")
        valid = {o["id"] for o in ping["options"]}
        if ping["kind"] != "question" and choice not in valid and not (ping["kind"] == "decision" and choice.startswith("select:")):
            raise WMError(f"{ping_id} accepts {sorted(valid)}, got {choice!r}")
        if ping["kind"] == "brief" and choice not in BRIEF_OPTIONS:
            raise WMError(f"brief accepts {BRIEF_OPTIONS}")
        if ping["kind"] == "yield_request" and choice not in YIELD_OPTIONS:
            raise WMError(f"yield_request accepts {YIELD_OPTIONS}")
        if ping["kind"] == "decision" and not (choice in DECISION_OPTIONS or choice.startswith("select:")):
            raise WMError(f"decision accepts {DECISION_OPTIONS} or select:<obs-id>")
        if choice == "override" and not why:
            raise WMError("override requires --why")
        if choice == "amend" and not (amend or answer or answers):
            raise WMError("amend requires --amend <file> or --answer \"field: value\"")
        existing = self.reply(ping_id)
        if existing is not None:
            if existing["choice"] == choice and ping["kind"] != "question":
                return existing
            raise WMError(f"{ping_id} already answered with {existing['choice']!r}; replies are immutable")
        reply = {
            "schema_version": REPLY_SCHEMA,
            "ping_id": ping_id,
            "card_id": self.card_id,
            "choice": choice,
            "why": why,
            "amend": amend,
            "answer": answer,
            "answers": answers or {},
            "created_at": now(),
            "by": os.environ.get("AWM_REPLY_BY", "scientist"),
        }
        dump_yaml(self.replies_dir / f"{ping_id}.yaml", reply)
        self.ledger.append("reply", card_id=self.card_id, ping_id=ping_id, choice=choice, why=why)
        return reply

    def record_timeout(self, ping_id: str) -> dict[str, Any]:
        ping = self.ping(ping_id)
        action = ping["timeout_action"]["action"]
        reply = {
            "schema_version": REPLY_SCHEMA,
            "ping_id": ping_id,
            "card_id": self.card_id,
            "choice": action,
            "why": f"no reply within {ping['timeout_action']['after_s']}s; timeout_action applied",
            "amend": None,
            "created_at": now(),
            "by": "timeout",
        }
        dump_yaml(self.replies_dir / f"{ping_id}.yaml", reply)
        self.ledger.append("timeout", card_id=self.card_id, ping_id=ping_id, choice=action)
        return reply


def render_ping(ping: dict[str, Any], path: Path | None = None) -> str:
    """The stdout view of a ping: what the blocking command prints."""
    lines = [f"[{ping['card_id']} {ping['ping_id']}] {ping['kind'].upper()}: {ping['summary']}"]
    if ping.get("prediction"):
        p = ping["prediction"]
        lines.append(f"  prediction: {p.get('metric')} {p.get('horizon')} "
                     f"delta {p.get('delta_mean'):+.3f} ± {p.get('delta_sd'):.3f} ({p.get('basis')})")
    for ev in ping.get("evidence", [])[:6]:
        lines.append(f"  evidence: {ev.get('path')} [{ev.get('locator')}] — {ev.get('observation', '')}")
    if ping.get("questions"):
        for i, q in enumerate(ping["questions"], 1):
            lines.append(f"  Q{i} [{q.get('field')}]: {q.get('question')}")
        lines.append(f"  reply: awm wm reply {ping['card_id']}/{ping['ping_id']} --answer \"<field>: <value>\\n...\"  "
                     f"(or --answer-file answers.yaml)")
    if ping.get("options"):
        lines.append("  options: " + " | ".join(
            f"{o['id']}" + (f" ({o['cost_min']} min)" if o.get("cost_min") else "") for o in ping["options"]))
        ta = ping.get("timeout_action") or {}
        lines.append(f"  on silence after {ta.get('after_s')}s: {ta.get('action')}")
        lines.append(f"  reply: awm wm reply {ping['ping_id']} --choose <option> [--why ...]")
    if path:
        lines.append(f"  file: {path}")
    return "\n".join(lines)
