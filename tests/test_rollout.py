"""The study harness pieces that can be checked without a cluster."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = mod
    spec.loader.exec_module(mod)
    return mod


def test_build_prior_runs_copies_and_indexes(tmp_path: Path) -> None:
    bpr = _load(REPO / "tools" / "build_prior_runs.py")
    raw = tmp_path / "raw"
    runs = [("cfg_a/gsm8k_google_gemma-3-4b-pt_1", "test", 0.61),
            ("cfg_b/gsm8k_Qwen_Qwen3-4B-Base_2", "train", 0.74),
            ("cfg_b/gsm8k_Qwen_Qwen3-1.7B-Base_3", "train", None)]
    for run, _side, acc in runs:
        d = raw / run
        (d / "task").mkdir(parents=True)
        (d / "solve_out.txt").write_text("trace " * 100)
        (d / "task" / "train.py").write_text("print(1)")
        if acc is not None:
            (d / "metrics.json").write_text(json.dumps({"accuracy": acc, "stderr": 0.02}))
        (d / "time_taken.txt").write_text("09:58:00")
    out = tmp_path / "prior_runs"
    summary = bpr.build([(r, s) for r, s, _ in runs] + [("cfg_z/missing_run_9", "train")], raw, out)
    assert summary["runs"] == 3 and summary["missing"] == ["cfg_z/missing_run_9"]
    assert summary["by_side"] == {"train": 2, "test": 1}
    assert (out / "cfg_b/gsm8k_Qwen_Qwen3-4B-Base_2/task/train.py").is_file()
    index = [json.loads(l) for l in (out / "index.jsonl").read_text().splitlines()]
    assert index[0]["accuracy"] == 0.74 and index[0]["base_model"] == "Qwen/Qwen3-4B-Base"
    assert index[-1]["accuracy"] is None
    assert index[1]["base_model"] == "google/gemma-3-4b-pt"
    md = (out / "INDEX.md").read_text()
    assert "| 0.740 | Qwen/Qwen3-4B-Base | cfg_b |" in md and "/home/ben/prior_runs/cfg_a/" in md


def test_build_prompts_is_ptb_plus_sections() -> None:
    bp = _load(REPO / "rollout" / "build_prompts.py")
    ptb = "intro `{model}` on {benchmark}\n## Rules\n1. x\n2. {num_hours} hours\n"
    wm = bp.wm_prompt(ptb, fulltraj=False)
    assert wm.startswith("intro `{model}`") and wm.endswith("2. {num_hours} hours\n")
    assert wm.count("## The world-model agent") == 1 and "## Prior runs" not in wm
    assert wm.index("## The world-model agent") < wm.index("## Rules")
    assert "awm wm propose" in wm and "awm wm finalize" in wm and "memory/index.md" not in wm
    wm_ft = bp.wm_prompt(ptb, fulltraj=True)
    assert wm_ft.index("## Prior runs") < wm_ft.index("## The world-model agent") < wm_ft.index("## Rules")
    c1 = bp.ptb_fulltraj(ptb)
    assert "## Prior runs" in c1 and "world-model" not in c1
    # the only difference between C1 and C2 prompts is the WMA section
    assert wm_ft.replace(bp.WMA_SECTION.rstrip("\n") + "\n\n", "") == c1
    with pytest.raises(SystemExit):
        bp.ptb_fulltraj("no rules heading")


def test_extra_binds_patch_is_idempotent(tmp_path: Path) -> None:
    src = REPO / "third_party" / "PostTrainBench" / "src" / "run_task.sh"
    if not src.is_file():
        pytest.skip("PostTrainBench submodule not checked out")
    patcher = _load(REPO / "rollout" / "patches" / "apply_extra_binds.py")
    once = patcher.apply(src.read_text())
    assert once != src.read_text()
    assert 'EXTRA_BIND_ARGS+=(--bind "$_b")' in once
    # exactly one exec line gains the extra binds, right after the HF cache bind
    agent_block = once.split(patcher.MARK, 1)[1]
    assert agent_block.count('"${EXTRA_BIND_ARGS[@]}" \\') == 1
    assert once.count('"${EXTRA_BIND_ARGS[@]}" \\') == 1
    assert patcher.apply(once) == once
