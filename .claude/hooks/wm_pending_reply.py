#!/usr/bin/env python3
"""Stop hook: hold the turn while a WMA ping that needs a reply is unanswered.

Standard library only (the ping files are flat YAML, so a line scan is enough).
The session directory is ``AWM_SESSION_DIR`` or the hook's cwd.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path


def pending_pings(session_dir: Path) -> list[str]:
    out = []
    cards = session_dir / "wm" / "cards"
    if not cards.is_dir():
        return out
    for card_dir in sorted(cards.glob("exp-*")):
        state = card_dir / "state.json"
        try:
            status = json.loads(state.read_text()).get("status") if state.is_file() else None
        except json.JSONDecodeError:
            status = None
        if status == "closed":
            continue
        for ping in sorted(card_dir.glob("pings/p-*.yaml")):
            text = ping.read_text()
            if not re.search(r"^reply_required:\s*true\s*$", text, re.MULTILINE):
                continue
            if (card_dir / "replies" / ping.name).is_file():
                continue
            kind = re.search(r"^kind:\s*(\S+)", text, re.MULTILINE)
            summary = re.search(r"^summary:\s*(.*)$", text, re.MULTILINE)
            out.append(f"{card_dir.name}/{ping.stem} ({kind.group(1) if kind else '?'}): "
                       f"{(summary.group(1) if summary else '').strip()[:140]} -> {ping}")
    return out


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        print("{}")
        return 0
    session_dir = Path(os.environ.get("AWM_SESSION_DIR") or hook_input.get("cwd") or ".").expanduser()
    pending = pending_pings(session_dir)
    if pending and not hook_input.get("stop_hook_active", False):
        reason = ("A world-model ping needs your reply before this turn can end:\n  - "
                  + "\n  - ".join(pending)
                  + "\nRead the ping file, then run `awm wm reply <card>/<ping> --choose <option> [--why ...]`. "
                    "You may reject any yield_request; a decision defaults to its timeout_action if you stay silent.")
        print(json.dumps({"decision": "block", "reason": reason}))
    else:
        print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
