from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def run_hook(script: str, payload: dict, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["python3", str(REPO / ".claude/hooks" / script)],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=False,
        env={**os.environ, "CLAUDE_PROJECT_DIR": str(REPO), **env},
    )


def test_session_hooks_capture_and_link_raw_rollout(tmp_path: Path) -> None:
    session_id = "session-test-001"
    env_file = tmp_path / "claude.env"
    start = run_hook(
        "session_start.py",
        {"session_id": session_id, "cwd": str(REPO)},
        {"CLAUDE_ENV_FILE": str(env_file)},
    )
    assert start.returncode == 0
    assert f"AWM_CLAUDE_SESSION_ID={session_id}" in env_file.read_text()

    transcript_dir = tmp_path / "native"
    transcript_dir.mkdir()
    transcript = transcript_dir / "trajectory.jsonl"
    transcript.write_text('{"type":"assistant","message":"done"}\n')
    (transcript_dir / "subagents").mkdir()
    (transcript_dir / "subagents/child.jsonl").write_text('{"type":"assistant"}\n')

    data_root = tmp_path / "data"
    experiment = data_root / "experiments/exp-001"
    experiment.mkdir(parents=True)
    (experiment / "state.json").write_text(json.dumps({
        "experiment_id": "exp-001",
        "status": "closed",
        "claude_session_id": session_id,
    }))

    end = run_hook(
        "session_end.py",
        {
            "session_id": session_id,
            "cwd": str(REPO),
            "transcript_path": str(transcript),
            "reason": "other",
        },
        {"AWM_DATA_ROOT": str(data_root)},
    )
    assert end.returncode == 0
    captured = data_root / "traj/runs/claude-code" / session_id
    assert (captured / "trajectory.jsonl").read_text() == transcript.read_text()
    assert (captured / "subagents/child.jsonl").is_file()
    metadata = json.loads((captured / "session.json").read_text())
    assert metadata["experiments"][0]["experiment_id"] == "exp-001"

