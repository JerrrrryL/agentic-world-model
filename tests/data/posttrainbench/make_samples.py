"""Rebuild the committed PostTrainBench samples from two full upstream runs.

The samples are verbatim line subsets of the real ``solve_out.txt`` files, chosen
so that every line kind a converter branches on is present (CUDA preamble,
init/result session boundaries, background-task notifications, a rate limit
event, list-valued tool results; on the codex side an item started far from its
completion, a file_change, a web_search and a todo_list). Sibling files are
copied whole — they are a few hundred bytes.

``solve_parsed.txt`` is regenerated for the trimmed claude sample with upstream's
own renderer, so the "one Tool call line per tool_use event" cross-check holds on
committed data too:

    python3 tests/data/posttrainbench/make_samples.py [FULL_RUNS_DIR]

FULL_RUNS_DIR must contain ``run_claude/`` and ``run_codex/``.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
DEFAULT_SRC = Path(
    "/tmp/claude-30001/-home-gangda-workspace-DeepCommit-ai-hierarchy-verifier"
    "/97160149-2e2d-43ef-a087-7b3437df11a9/scratchpad/hf"
)
PARSER = REPO / "third_party/PostTrainBench/src/trace_parsing/parse_trace.py"

SAMPLES = {
    "run_claude": (
        "claude_non_api_max_claude-opus-4-8_10h_run1",
        "gsm8k_Qwen_Qwen3-1.7B-Base_17315721",
        [(1, 24), (32, 34), (70, 76), (97, 100), (255, 262)],
        ("metrics.json", "time_taken.txt", "judgement_gpt5_4.json"),
        "claude_non_api_max",
    ),
    "run_codex": (
        "codex_non_api_high_gpt-5.4_10h_run1",
        "gsm8k_Qwen_Qwen3-1.7B-Base_16934887",
        [(1, 19), (85, 90), (113, 115), (195, 197), (298, 300)],
        ("metrics.json", "judgement_api.json"),
        None,
    ),
}


def main(src_root: Path) -> None:
    for src_name, (cfg, run, spans, siblings, agent) in SAMPLES.items():
        src = src_root / src_name
        dest = HERE / cfg / run
        dest.mkdir(parents=True, exist_ok=True)
        lines = src.joinpath("solve_out.txt").read_text(encoding="utf-8").split("\n")
        keep = [lines[i - 1] for a, b in spans for i in range(a, b + 1)]
        out = dest / "solve_out.txt"
        out.write_text("\n".join(keep) + "\n", encoding="utf-8")
        for name in siblings:
            if (src / name).exists():
                shutil.copy(src / name, dest / name)
        if agent and PARSER.exists():
            # The renderer prints every unparsable line and then exits non-zero
            # over a missing .env, after it has already written the output.
            subprocess.run(
                [sys.executable, str(PARSER), "--agent", agent, str(out),
                 "-o", str(dest / "solve_parsed.txt")],
                check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            assert (dest / "solve_parsed.txt").exists()
        print(f"{dest}: {out.stat().st_size} B, {len(keep)} lines")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SRC)
