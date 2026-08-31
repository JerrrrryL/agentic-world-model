#!/usr/bin/env python3
"""Copy the raw Claude Code transcript into the AWM trajectory volume."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def main() -> int:
    try:
        hook_input = json.load(sys.stdin)
    except (json.JSONDecodeError, OSError):
        return 0
    session_id = hook_input.get("session_id")
    transcript = Path(str(hook_input.get("transcript_path", ""))).expanduser()
    if not isinstance(session_id, str) or not session_id or not transcript.is_file():
        return 0

    project = Path(os.environ.get("CLAUDE_PROJECT_DIR", hook_input.get("cwd", ".")))
    data_root = Path(os.environ.get("AWM_DATA_ROOT", project / "data")).expanduser()
    destination = data_root / "traj" / "runs" / "claude-code" / session_id
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(transcript, destination / "trajectory.jsonl")

    subagents = transcript.parent / "subagents"
    if subagents.is_dir():
        shutil.copytree(subagents, destination / "subagents", dirs_exist_ok=True)

    experiments = []
    experiment_root = Path(
        os.environ.get("AWM_EXPERIMENT_ROOT", data_root / "experiments")
    ).expanduser()
    if experiment_root.is_dir():
        for state_path in sorted(experiment_root.glob("*/state.json")):
            try:
                state = json.loads(state_path.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if state.get("claude_session_id") == session_id:
                experiments.append({
                    "experiment_id": state.get("experiment_id", state_path.parent.name),
                    "path": str(state_path.parent.resolve()),
                    "status": state.get("status"),
                })

    atomic_json(destination / "session.json", {
        "schema_version": "awm-claude-rollout-v1",
        "session_id": session_id,
        "source_transcript_path": str(transcript.resolve()),
        "captured_at": now(),
        "cwd": hook_input.get("cwd"),
        "end_reason": hook_input.get("reason"),
        "experiments": experiments,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

