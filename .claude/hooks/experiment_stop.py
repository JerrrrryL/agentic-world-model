#!/usr/bin/env python3
"""Ask Claude to review an executed experiment before ending the turn.

This hook intentionally uses only the standard library.  It still works before
the editable Python package and its dependencies have been installed.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        print("{}")
        return 0

    project = Path(os.environ.get("CLAUDE_PROJECT_DIR", hook_input.get("cwd", ".")))
    root = Path(
        os.environ.get(
            "AWM_EXPERIMENT_ROOT",
            str(Path(os.environ.get("AWM_DATA_ROOT", project / "data")) / "experiments"),
        )
    ).expanduser()
    active = []
    if root.is_dir():
        for state_path in sorted(root.glob("*/state.json")):
            try:
                state = json.loads(state_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if state.get("status") in {"queued", "running", "awaiting_review"}:
                active.append(
                    f"{state.get('experiment_id', state_path.parent.name)} "
                    f"({state['status']}, {state_path.parent})"
                )

    # One feedback turn is enough.  If Claude cannot close the bundle after the
    # hook has already continued it, allow the turn to end rather than loop.
    if active and not hook_input.get("stop_hook_active", False):
        reason = (
            "Experiment lifecycle is incomplete: " + "; ".join(active) + ". "
            "If execution finished, inspect its logs and measurements, complete the "
            "scientist-owned result.yaml, and run `python3 -m awm.cli experiment "
            "finalize <experiment-dir>`. If work is still running, report its status "
            "and continue monitoring."
        )
        print(json.dumps({"decision": "block", "reason": reason}))
    else:
        print("{}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

