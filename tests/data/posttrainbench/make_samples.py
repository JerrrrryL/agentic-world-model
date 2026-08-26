"""Rebuild the committed PostTrainBench samples from four full upstream runs.

The samples are verbatim line subsets of the real ``solve_out.txt`` files, chosen
so that every line kind a converter branches on is present:

*   ``run_claude`` — CUDA preamble, init/result session boundaries, background
    task notifications, a rate limit event, list-valued tool results.
*   ``run_codex`` — an item started far from its completion, a file_change, a
    web_search, a todo_list.
*   ``run_opencode`` — the ``opencode: command not found`` preamble's cousin (a
    non-JSON banner), step_start/step_finish framing, bash / read / write / edit
    / todowrite / websearch calls, a non-zero exit, a truncated read, and the
    ``tool: "invalid"`` line the CLI emits for a malformed call.
*   ``run_cursor`` — init and the echoed prompt, delta-streamed thinking, every
    tool key the release uses, a webFetch failure, the approval handshake, a
    connection/retry pair and the three replayed tool lines that follow it, a
    task notification, and the single ``result``.

Sibling files are copied whole — they are a few hundred bytes.

``solve_parsed.txt`` is regenerated for the trimmed claude sample with upstream's
own renderer, so the "one Tool call line per tool_use event" cross-check holds on
committed data too. It is NOT regenerated for the other three: codex and
opencode have no renderer entry, and upstream's cursor rendering does not dedupe
the CLI's replayed tool lines (measured: it matches ``tool_use`` plus the
replayed ``started`` lines on 41 of the 46 cursor runs that have one, and is
short by one or two on the other five), so it is not an oracle worth pinning.

    python3 tests/data/posttrainbench/make_samples.py [FULL_RUNS_DIR]

FULL_RUNS_DIR may hold ``run_claude/``, ``run_codex/``, ``run_opencode/`` and
``run_cursor/``. Anything it does not hold is read from the fetched release
under ``$AWM_DATA_ROOT/traj/raw/posttrainbench/<config>/<run>`` instead, which is
where ``awm traj fetch posttrainbench --all`` puts it.
"""

from __future__ import annotations

import os
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
    "run_opencode": (
        "opencode_opencode_kimi-k2.5_10h_run2",
        "gpqamain_Qwen_Qwen3-1.7B-Base_16853750",
        # Line 35 is a 23 KB bash result and line 40 a 21 KB codesearch; both
        # are skipped, so step 4 keeps its start and finish but loses its call.
        [(1, 23), (32, 34), (36, 36), (43, 46), (68, 76), (91, 93), (103, 108), (232, 236)],
        ("metrics.json", "time_taken.txt", "judgement_gpt5_4.json"),
        None,
    ),
    "run_cursor": (
        "cursor_cli_cursor-grok-4.5-high_10h_run2",
        "healthbench_google_gemma-3-4b-pt_17417310",
        # 785-796 is the reconnect: two started lines and one completed replayed
        # verbatim after the connection/retry pair, which is the only place the
        # replay counter can be exercised on committed data.
        [(1, 12), (18, 27), (54, 56), (99, 105), (188, 194), (196, 197), (513, 515),
         (545, 549), (785, 796), (844, 848), (856, 859), (1240, 1243), (1288, 1291),
         (1584, 1587), (1718, 1739)],
        ("metrics.json", "time_taken.txt", "judgement_gpt5_4.json"),
        None,
    ),
}


def _source(src_root: Path, src_name: str, cfg: str, run: str) -> Path:
    """The full run to trim: the hand-staged copy if there is one, else the
    fetched release."""
    staged = src_root / src_name
    if staged.is_dir():
        return staged
    root = Path(os.environ.get("AWM_DATA_ROOT", REPO / "data"))
    return root / "traj/raw/posttrainbench" / cfg / run


def main(src_root: Path) -> None:
    for src_name, (cfg, run, spans, siblings, agent) in SAMPLES.items():
        src = _source(src_root, src_name, cfg, run)
        if not src.is_dir():
            print(f"{src_name}: no source at {src} — skipped")
            continue
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
