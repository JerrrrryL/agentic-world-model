#!/usr/bin/env python3
"""Expose the Claude session id to experiment commands in this session."""

from __future__ import annotations

import json
import os
import shlex
import sys
from pathlib import Path


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        print("{}")
        return 0
    session_id = hook_input.get("session_id")
    env_file = os.environ.get("CLAUDE_ENV_FILE")
    if isinstance(session_id, str) and session_id and env_file:
        with Path(env_file).open("a") as handle:
            handle.write(f"export AWM_CLAUDE_SESSION_ID={shlex.quote(session_id)}\n")
    print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

