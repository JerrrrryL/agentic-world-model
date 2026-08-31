# Experiment card v1

## Purpose

An agent post-training a base model for ten hours runs a sequence of
experiments. Each one is worth keeping only if someone else — a fresh instance
of the same agent after context loss, a different agent on the same task, or a
person reading the run afterwards — can answer two questions from the record
alone:

1. **What was this experiment testing?** A problem the agent had observed, and
   a hypothesis about what would change it.
2. **How do I run it again?** The checkpoint it started from, the data it
   trained on, the exact command, and the evaluation it was judged by — all as
   paths and argv inside the session directory.

The experiment card is that record. The template is
`doc/exp-card-specs/experiment-card.template.yaml`; a filled-in example is
`doc/exp-card-specs/example-card.yaml`. This document says when a card is
written, what each section must contain, and how cards make up the agent's
memory.

## One card, one experiment

A card is one problem, one hypothesis, one intervention, one result. Not a
sweep, not "try RFT and also DPO", not a run. A three-stage pipeline is three
cards, each naming the previous card's output as its `parent_checkpoint`.

The six sections and the question each answers:

| section | question | written |
|---|---|---|
| `problem` | what did you see going wrong, and where | before launch |
| `hypothesis` | what do you expect to change, against what, and what would prove you wrong | before launch |
| `setup` | parent checkpoint, data, method, hyper-parameters, exact command | before launch |
| `evaluation` | how will it be measured, against which comparator, under the same protocol | before launch |
| `result` | what physically happened: checkpoint, measurements, failure | after |
| `conclusion` | verdict on the hypothesis, decision, next step | after |

## Before launch

Sections 1–4 are written **before the command in `setup.command` runs**. This
is the only rule with teeth: a hypothesis written after the result is a
description of the result, and a setup written from memory is not one anyone
can rerun.

**Problem.** A concrete failure, with `evidence` pointing at real files in the
session directory — an eval output with item ids, a log with a line range, a
failure-tag file. "The model is bad at math" is not a problem; "41 of 96
dev-300 failures are arithmetic slips inside an otherwise correct chain of
thought, see `analysis/exp-02_failure_tags.json`" is.

Two things make the problem inspectable rather than asserted:

- `failure_examples` — three to ten actual items the model gets wrong: the
  question, the gold answer, the model's output trimmed to the failing step,
  and one line on what went wrong. A reader who disagrees with the problem
  statement can look at the items and say so.
- `watch_set` — the file of items the model currently fails on this problem
  (ids, questions), with how they were selected. It is the set the agent
  re-runs on the output checkpoint, and `result.watch_set_result` reports how
  many were fixed, how many still fail, and — if measured — how many items
  outside the set regressed.

Every item in both comes from the agent's own dev or probe sets, built from
permitted training data. The benchmark's test copy is input to the
contamination checker only; an item from it in a card is a rule violation.

**Hypothesis.** `claim` names the intervention and the metric. `mechanism`
says why it should work. `expected_effect` names the comparator and the
direction; `magnitude` is optional and may be `null`. `falsified_if` is an
observation, not a feeling. A target ("reach 85 %") is not a hypothesis and is
not accepted in `claim`.

**Setup.** Everything needed to rerun:

- `parent_checkpoint.path` — absolute, inside `{dir}`, and `origin` says which
  card produced it (or `base_model`).
- `data[]` — each file the trainer read, its `source` (a hub id, `synthetic:self`,
  `derived:<card_id>`, …), `n_examples`, the `build_command` that regenerates it,
  the `selection` rule, and whether the supplied contamination checker passed.
- `method` — family from the closed vocabulary, framework and version, PEFT,
  every hyper-parameter, seed, and the target format the grader will read.
- `command.argv` — the exact command, `cwd`, the script path, the env vars that
  matter, and where the log goes.

**Evaluation.** The protocol (`evaluate.py --limit N`, or a dev set the agent
built from permitted training data), its `n` and seed, and the comparator's
value **under the same protocol** with the path to that eval's output. A
comparator measured on a different `--limit` is not a comparator. `diagnostic`
is optional: a probe that tests the mechanism rather than the score (for the
example, the arithmetic-slip share of failures). Without one,
`conclusion.mechanism_verdict` must be `not_tested`.

## After

**Result** is physical: execution status, wall time, the output checkpoint path,
training summary, each measurement with the path to its eval output and its
delta against the comparator, the watch-set re-run (`fixed`, `still_failing`,
`regressions`), and the traceback tail on failure. A failed or
killed run still gets a result and a conclusion; a card is never deleted.

**Conclusion.** `verdict` is `supported` or `contradicted` only from a
measurement under the declared protocol; otherwise `inconclusive` (run failed,
cut short, or delta within noise). `mechanism_verdict` comes only from
`diagnostic`. `decision` is one of `adopt` (output becomes the incumbent),
`reject`, `iterate` (same hypothesis, changed setup — name the change in
`next_step`), or `abandon_line`. `summary` is two lines a reader can stop at.

## Memory

Cards live at `{dir}/memory/cards/exp-NN.yaml`, numbered in the order they
were opened, plus `{dir}/memory/index.md` — one line per card: id, elapsed
hour, family, parent, verdict, decision, best measurement. The agent keeps the
index current; it is the first thing a resumed instance reads.

A card chain is the run's history: follow `parent_checkpoint.origin` back from
the adopted card to the base model and you have the recipe that shipped;
every `reject`ed and `abandon_line`d card off that path is a negative result
worth as much as the positives.

## Reproduction check

A card is finished when a second agent, given only the card and `{dir}`, can:

1. run `setup.data[].build_command` and get files with the stated `n_examples`;
2. run `setup.command.argv` from `setup.command.cwd` and produce a checkpoint;
3. run `evaluation.protocol.command` on it and on the comparator, and get
   numbers comparable with `result.measurements`;
4. state the hypothesis and the verdict in one sentence each.

The reproduction does not need to match to the decimal; it needs to be the
same experiment.

## Wiring into the task prompt

The agent reads one prompt, `input/instruction.md`. Its **Experiments** section
requires a card for every experiment, names the four sections that must be
complete before the command runs, and rule 10 makes launching without one a
violation. The **Memory** section makes the cards and `memory/index.md` the core
of the agent's memory.

`input/` is the complete bundle placed in `{dir}`: `instruction.md` and
`exp-card.template.yaml`. The template under `doc/exp-card-specs/` is the
source of truth; `input/exp-card.template.yaml` is a copy and must be
refreshed whenever the source changes.
