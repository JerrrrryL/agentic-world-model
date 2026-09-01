"""Write the study's prompt files into a PostTrainBench checkout.

Every prompt is PostTrainBench's own ``prompt.txt`` plus at most two sections
inserted before ``## Rules`` — nothing else changes, so a cell differs from the
corpus runs only by what was added:

    prompt_fulltraj.txt      PTB prompt + "Prior runs"                          (C1: raw files, no WMA)
    prompt_wm.txt            PTB prompt + "The world-model agent"               (C3: WMA over memory)
    prompt_wm_fulltraj.txt   PTB prompt + "Prior runs" + "The world-model agent" (C2: raw files + WMA)

The WMA section is ``input/wma_section.md`` (the source of truth); the rendered
``prompt_wm.txt`` is also written to ``input/instruction.md`` for reference.
``get_prompt.py`` fills ``{model} {benchmark} {num_hours} ...`` by plain
replacement, so the sections use those placeholders too. Run by
rollout/setup.sh:

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

WMA_SECTION = (ROOT / "input" / "wma_section.md").read_text()


def _insert_before_rules(prompt: str, *sections: str) -> str:
    anchor = "## Rules"
    if anchor not in prompt:
        raise SystemExit("PTB prompt.txt has no '## Rules' heading; update build_prompts.py")
    block = "".join(sec if sec.endswith("\n\n") else sec.rstrip("\n") + "\n\n" for sec in sections)
    return prompt.replace(anchor, block + anchor, 1)


def ptb_fulltraj(ptb_prompt: str) -> str:
    """C1: the PTB prompt with the prior-runs section."""
    return _insert_before_rules(ptb_prompt, PRIOR_RUNS_SECTION)


def wm_prompt(ptb_prompt: str, *, fulltraj: bool) -> str:
    """C2/C3: the PTB prompt with the world-model section (and prior runs for C2)."""
    if fulltraj:
        return _insert_before_rules(ptb_prompt, PRIOR_RUNS_SECTION, WMA_SECTION)
    return _insert_before_rules(ptb_prompt, WMA_SECTION)


def find_ptb_prompt(ptb: Path | None) -> Path:
    for cand in ([ptb] if ptb else []) + [ROOT / "third_party" / "PostTrainBench"]:
        f = Path(cand) / "src" / "eval" / "general" / "prompt.txt"
        if f.is_file():
            return f
    raise SystemExit("no PostTrainBench prompt.txt found; pass the checkout path or init the submodule")


def main() -> int:
    ptb = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
    ptb_prompt = find_ptb_prompt(ptb).read_text()
    out_review = HERE / "prompts"
    out_review.mkdir(exist_ok=True)
    files = {
        "prompt_fulltraj.txt": ptb_fulltraj(ptb_prompt),
        "prompt_wm.txt": wm_prompt(ptb_prompt, fulltraj=False),
        "prompt_wm_fulltraj.txt": wm_prompt(ptb_prompt, fulltraj=True),
    }
    for name, text in files.items():
        (out_review / name).write_text(text)
        if ptb:
            (ptb / "src" / "eval" / "general" / name).write_text(text)
    (ROOT / "input" / "instruction.md").write_text(files["prompt_wm.txt"])
    where = f" and {ptb / 'src/eval/general'}" if ptb else ""
    print(f"wrote {', '.join(files)} to {out_review}{where}; input/instruction.md = prompt_wm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
