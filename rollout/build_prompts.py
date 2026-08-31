"""Write the study's prompt files into a PostTrainBench checkout.

``get_prompt.py`` loads ``src/eval/general/${POST_TRAIN_BENCH_PROMPT}.txt`` and
fills ``{model} {benchmark} {num_hours} {gpu_info} {setup_other}
{decontamination_tool} {eval_api_note}`` by plain replacement, so every prompt
here is written in those placeholders. Four files:

    prompt_fulltraj.txt      PTB prompt + a "Prior runs" section          (C1: raw files, no WMA)
    prompt_wm.txt            our instruction.md in PTB placeholders        (C3: WMA, memory only)
    prompt_wm_fulltraj.txt   the same + the "Prior runs" section           (C2: raw files + WMA)

plus copies under rollout/prompts/ so the rendered text is reviewable in this
repo. Run by rollout/setup.sh:

    python rollout/build_prompts.py <ptb checkout>
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

PRIOR_RUNS_SECTION = """## Prior runs
- `/home/ben/prior_runs/` holds the complete, read-only records of previous attempts at this same task (post-training a base model for {benchmark} under these rules) by other autonomous agents. Start with `/home/ben/prior_runs/INDEX.md`: one row per run with its base model, the agent that ran it, its official accuracy, and its path.
- Each run directory has `solve_out.txt` (the agent's full session trace), `solve_parsed.txt` (the same, condensed), `task/` (every script it wrote: data preparation, training, its own eval outputs), `metrics.json` (its official score), and `time_taken.txt`.
- You may read anything there and reuse any idea, script, or recipe. Every rule below still applies to what you do with it: a prior run's training data is not exempt from the contamination rules, and a prior run's base model is not yours unless it is `{model}`.
- Reading files under `/home/ben/prior_runs` is an allowed exception to the working-directory rule.

"""

# instruction.md placeholders -> PTB placeholders / sandbox constants
INSTRUCTION_MAP = {
    "{dir}": "/home/ben/task",
    "{submission}": "/home/ben/task/final_model",
    "{time_limit}": "{num_hours} hours",
    "{gpu}": "one Nvidia H100 GPU",
}


def ptb_fulltraj(ptb_prompt: str) -> str:
    """The PTB prompt with the prior-runs section inserted before ## Rules."""
    anchor = "## Rules"
    if anchor not in ptb_prompt:
        raise SystemExit("PTB prompt.txt has no '## Rules' heading; update build_prompts.py")
    return ptb_prompt.replace(anchor, PRIOR_RUNS_SECTION + anchor, 1)


def wm_prompt(instruction: str, *, fulltraj: bool) -> str:
    """Our instruction.md rendered into PTB placeholders."""
    text = instruction
    for k, v in INSTRUCTION_MAP.items():
        text = text.replace(k, v)
    # PTB's per-benchmark fill-ins: the inspect-ai note and the decontamination tool
    env_anchor = "- A copy of the `{benchmark}` test set and a contamination checker are available."
    if env_anchor not in text:
        raise SystemExit("instruction.md changed: the test-set/contamination bullet is missing")
    text = text.replace(env_anchor, "{setup_other}{decontamination_tool}\n" + env_anchor, 1)
    if fulltraj:
        anchor = "## The world-model agent"
        if anchor not in text:
            raise SystemExit("instruction.md changed: '## The world-model agent' heading missing")
        text = text.replace(anchor, PRIOR_RUNS_SECTION + anchor, 1)
    leftover = [p for p in ("{dir}", "{submission}", "{time_limit}", "{gpu}") if p in text]
    if leftover:
        raise SystemExit(f"unrendered placeholders: {leftover}")
    return text


def main() -> int:
    ptb = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    instruction = (ROOT / "input" / "instruction.md").read_text()
    out_review = HERE / "prompts"
    out_review.mkdir(exist_ok=True)
    files = {
        "prompt_wm.txt": wm_prompt(instruction, fulltraj=False),
        "prompt_wm_fulltraj.txt": wm_prompt(instruction, fulltraj=True),
    }
    if ptb:
        ptb_prompt = (ptb / "src" / "eval" / "general" / "prompt.txt").read_text()
        files["prompt_fulltraj.txt"] = ptb_fulltraj(ptb_prompt)
    for name, text in files.items():
        (out_review / name).write_text(text)
        if ptb:
            (ptb / "src" / "eval" / "general" / name).write_text(text)
    where = f" and {ptb / 'src/eval/general'}" if ptb else " (no checkout given: PTB variant skipped)"
    print(f"wrote {', '.join(files)} to {out_review}{where}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
