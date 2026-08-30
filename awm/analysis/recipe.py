"""Cut a run's event stream down to the events that can carry a post-training recipe.

A PostTrainBench run is an agent given ten hours and a base model. What we want
out of it is the *recipe* it actually shipped — which datasets in what mixture,
which algorithm, which hyper-parameters — because that, not the agent's name, is
the unit the outcome predictor is supposed to score. The recipe is never stated
in one place. It is spread across the training script the agent wrote, the
command that launched it, and the sentence where the agent explains why it is
about to change the mixture.

Reading a whole run to find that is wasteful and, past a point, impossible: the
median gsm8k run is 561 events and the longest is 67,420. This module drops the
stream to the events that mention something a recipe is made of. Measured over
the 143 train runs of ``gsm8k-gemma-holdout-v1``: median 54 events / 37.6 k
characters, worst case 125 k, and **no run is reduced to nothing**. That last
part is the property worth re-checking whenever ``RECIPE`` changes — a pattern
that silently stops matching turns a run into a blank record rather than an
error, which is the failure mode this whole file exists to avoid.

Four decisions that are load-bearing:

*   **Tool vocabularies are normalised, not assumed.** The four wire formats name
    the same action four ways — ``Bash`` / ``command_execution`` / ``shell`` /
    ``bash`` — and a filter written against one of them silently keeps nothing
    for the other three. 26 of the 143 train runs are opencode and 6 are cursor;
    both would read as "this agent ran no commands".
*   **Writes keep four times more text than commands.** A training script is the
    recipe; a shell one-liner mentioning ``--lr`` is a pointer to it.
*   **A result is kept only if it directly follows a kept call**, and only its
    tail, because that is where the loss numbers and the traceback are.
*   **The budget is spent from the end backwards.** An agent's first hour is
    exploration and its last is the run it submits, so when a trajectory does not
    fit, the part to lose is the beginning.

The digest deliberately carries **no accuracy and no score**. An extractor that
knows the run scored 0.86 will describe the recipe as if it were a good one; the
label is joined back on afterwards, by :func:`join_outcome`, once nothing can
edit the reading of the evidence.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Any, Iterable

#: The same action under four wire formats. See the module docstring — a filter
#: written against ``Bash`` alone is blind to a fifth of this split.
SHELL_TOOLS = frozenset({"bash", "command_execution", "shell"})
WRITE_TOOLS = frozenset({"write", "file_change", "edit"})
READ_TOOLS = frozenset({"read"})

#: What a post-training recipe is made of. Deliberately wide: a false positive
#: costs a few hundred characters of digest, a false negative loses the recipe.
RECIPE_SIGNAL = re.compile(
    r"""(?xi)
    \b(
      sft|dpo|grpo|ppo|orpo|kto|simpo|rloo|rlhf|rlaif|reinforce|
      lora|qlora|peft|adapter|full[_-]?finetune|
      trl|axolotl|llama[-_]?factory|unsloth|open[-_]?instruct|verl|openrlhf|
      deepspeed|fsdp|accelerate\s+launch|zero[-_]?[123]|
      load_dataset|datasets\.load|hf_hub_download|snapshot_download|
      learning[_-]?rate|--lr\b|num_train_epochs|per_device_train_batch|
      gradient_accumulation|max_seq_len|warmup|weight_decay|lr_scheduler|
      train(er|ing)?\.py|finetune|fine[_-]tune|distill|self[_-]?consistency|
      rejection[_-]?sampling|best[_-]?of[_-]?n|majority[_-]?vote|
      curriculum|mixture|upsample|downsample|dedup|
      checkpoint|save_model|merge_and_unload|push_to_hub
    )\b
    """
)

#: Sentences where an agent says which artefact it is submitting. Kept alongside
#: ``RECIPE_SIGNAL`` because "the recipe" and "the recipe I shipped" are
#: different questions and only the second one has a single answer.
DELIVERY_SIGNAL = re.compile(
    r"(?i)\b(submit|final|deliver|/output|output_dir|save_pretrained"
    r"|merged?_model|best[_-]?checkpoint)\b"
)

WRITE_CAP = 12_000
SHELL_CAP = 3_000
RESULT_CAP = 1_500
SAY_CAP = 2_500
DEFAULT_BUDGET = 180_000


def normalise_tool(tool: str | None) -> str:
    """Map a harness's tool name onto the action it performs."""
    t = (tool or "").lower()
    if t in SHELL_TOOLS:
        return "shell"
    if t in WRITE_TOOLS:
        return "write"
    if t in READ_TOOLS:
        return "read"
    return t or "?"


def _flatten(args: Any) -> str:
    """Render tool arguments as text. Non-scalar values are dropped, not stringified."""
    if isinstance(args, str):
        return args
    if not isinstance(args, dict):
        return ""
    return "\n".join(f"{k}: {v}" for k, v in args.items() if isinstance(v, (str, int, float)))


def _events(path: Path) -> Iterable[dict]:
    with gzip.open(path, "rt") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def select(path: Path, budget: int = DEFAULT_BUDGET) -> list[dict]:
    """Return the recipe-bearing events of one run, newest-first-priority.

    ``budget`` caps the rendered characters. When a run overflows it, the events
    dropped are the *earliest* ones — see the module docstring on why the tail is
    the part worth keeping.
    """
    kept: list[dict] = []
    for e in _events(path):
        act = normalise_tool(e.get("tool"))
        kind = e.get("type")
        if kind == "tool_use" and act in ("shell", "write"):
            text = _flatten(e.get("args"))
            if not RECIPE_SIGNAL.search(text):
                continue
            cap = WRITE_CAP if act == "write" else SHELL_CAP
            kept.append(_clip(e, act, text, cap))
        elif kind == "tool_result":
            if kept and e.get("i") == kept[-1]["i"] + 1:
                text = _flatten(e.get("args")) or (e.get("summary") or "")
                if text:
                    kept.append({"i": e["i"], "act": "result", "text": text[-RESULT_CAP:]})
        elif kind == "text" and e.get("role") == "assistant":
            text = e.get("text") or ""
            if RECIPE_SIGNAL.search(text) or DELIVERY_SIGNAL.search(text):
                kept.append(_clip(e, "say", text, SAY_CAP))

    out: list[dict] = []
    spent = 0
    for item in reversed(kept):
        size = len(item["text"])
        if spent + size > budget:
            continue
        out.append(item)
        spent += size
    out.reverse()
    return out


def _clip(event: dict, act: str, text: str, cap: int) -> dict:
    clipped = text[:cap] + ("\n…[truncated]" if len(text) > cap else "")
    return {"i": event["i"], "turn": event.get("turn"), "act": act, "text": clipped}


HEADER_KEYS = ("trained_model", "benchmark", "trace_format", "time_budget_h")


def render(run: str, events: list[dict], meta: dict | None = None,
           include_agent: bool = False) -> str:
    """Render a digest as the text an extractor reads.

    ``meta`` carries only what the extractor needs to disambiguate — the base
    model being post-trained, the benchmark, the harness. It must not carry the
    score; see the module docstring.

    ``include_agent`` adds ``agent_model`` to the header. It defaults to False
    because which agent wrote the trajectory is the strongest single predictor
    of the outcome (66% of accuracy variance) and naming it invites the
    extractor to describe the agent instead of the recipe. Callers who want the
    old behaviour must ask for it. The ``run`` argument is the caller's label —
    pass the run name, not ``{experiment}__{run_name}``: 962 of 1175
    experiment names spell the agent out.
    """
    head = [f"# run: {run}"]
    for key in HEADER_KEYS + (("agent_model",) if include_agent else ()):
        if meta and key in meta:
            head.append(f"# {key}: {meta[key]}")
    head.append(f"# recipe-bearing events: {len(events)}")
    head.append("")
    body = []
    for e in events:
        turn = f" turn={e['turn']}" if e.get("turn") is not None else ""
        body.append(f"--- [{e['i']}]{turn} {e['act']} ---\n{e['text']}")
    return "\n".join(head + body)


def join_outcome(recipe: dict, catalog_row: dict) -> dict:
    """Attach the outcome to an extracted recipe, after extraction, never before."""
    return {
        **recipe,
        "accuracy": catalog_row.get("accuracy"),
        "stderr": catalog_row.get("stderr"),
        "total_cost_usd": catalog_row.get("total_cost_usd"),
        "num_turns": catalog_row.get("num_turns"),
        "duration_ms": catalog_row.get("duration_ms"),
    }
