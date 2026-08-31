# Reconstructing experiment cards from past rollouts

`experiment-card.template.yaml` is written by the agent that runs the
experiment. This protocol is for the other direction: an extractor reads a
**finished** PostTrainBench run and writes, after the fact, the cards that
agent *would* have written. The output is the same schema, plus a `provenance`
block, and with one rule that overrides everything else:

> **If the trajectory does not state it, the field is empty.** `null`, `[]`,
> or `""`. Never a default, never a plausible guess, never a number copied from
> anywhere but the run itself.

## What the extractor is given

For one run, under `<data>/exp-cards/<split>/`:

- `digests/<run_ref>.md` — the run's event stream, filtered to the events that
  can carry a recipe or an evaluation, each block headed `--- [i] turn=T act ---`
  where `i` is the event index. It carries **no score** and **no agent
  identity**: the header names only the base model, the benchmark, and the
  budget.
- `task/<run_ref>/` — the agent's workspace as it was at the end of the run:
  the training scripts it wrote (`train*.py`), data-prep scripts
  (`prepare_data*.py`), its own eval outputs (`eval_*.json`,
  `baseline_results.json`, `logs/*.json`), templates, `evaluate.py`. No
  checkpoints.
- The manifest line: `trained_model`, `side` (train/test), event counts.

The extractor is **not** given: the official accuracy, the agent model, the
harness, the experiment or run name. It must not try to infer them, and must
not write anything that names them.

## What one run yields

One card per **launch** the agent made — a training run, a merge, a packaging
step, or a decode/config change that produced a candidate — in the order they
were launched, numbered `exp-01`, `exp-02`, …. Abandoned and killed launches
are cards too (`conclusion.decision: reject` or `abandon_line`). A launch the
agent wrote a script for but never ran is not a card; mention it in the
previous card's `conclusion.next_step` if the agent said so.

Every card must cite its launch: `setup.command.argv` is the command as it
appears in the stream, and `provenance.launch_i` is that event's index. A card
whose launch cannot be found in the digest is not written.

**Smoke tests are not cards.** A deliberately truncated run — two steps, ten
examples — exists to check that the pipeline works, not to produce a
candidate. Recording each as an experiment would make a run that fought its
trainer's API five times look like five abandoned hypotheses. They go on the
next real launch's card as `provenance.smoke_runs: [{launch_i, outcome}]`,
which keeps the pitfalls ("SFTConfig has no `evaluation_strategy`") without
inflating the count. A run the agent meant as real and then killed, or that
crashed, is a card.

## Filling each section from a trajectory

| section | fill from | when empty |
|---|---|---|
| `problem.statement` | what the agent said was wrong *before* this launch — an eval it just read, a failure it named | `null` if the agent launched without saying why; `provenance.stated_by_agent.problem: false` |
| `problem.evidence[]` | `path` = the eval output in `task/` the agent had just looked at, or the digest; `locator` = `[i]` of the event | `[]` |
| `problem.failure_examples[]` | only if the agent printed specific failing items and they are in the stream or in `task/logs/*.json` | `[]` — do not fabricate items |
| `problem.watch_set` | only if the agent kept such a set | `null` |
| `hypothesis.claim` | the agent's own words on what it expected this launch to do; reconstruct minimally from what was launched only if `provenance.stated_by_agent.hypothesis` is set `false` | `null` |
| `hypothesis.falsified_if`, `expected_effect.magnitude` | only if stated | `null` — expected on most cards |
| `setup.parent_checkpoint` | `--model-name` / `from_pretrained` path in the launch; `origin` = the card whose `result.output_checkpoint` matches, else `base_model` | `path` from the command; `hash: null` |
| `setup.data[]` | the data files the launch read, and the `prepare_data*.py` that built them (`built_by`, `build_command` from its invocation); `source`, `n_examples`, `selection` from the script or the agent's prints | `n_examples: null` if never printed |
| `setup.method.hyperparams` | the launch args, then the script's argparse defaults **only if the script text is in `task/` and the launch did not override** — cite which | `null` |
| `setup.command` | verbatim argv, `cwd` from the stream, `script` = the `task/` file | `log: null` |
| `evaluation.protocol` | the `evaluate.py --limit N` the agent ran on this output | `null` if it never evaluated |
| `evaluation.comparator` | the value the agent compared against, under the same `--limit`, and the `task/eval_*.json` holding it | `null` if the agent compared nothing |
| `result.execution` | `completed` if a save/eval followed; `killed` if the agent killed it; `failed` on a traceback | `unknown` is not allowed — pick `killed` for a launch that simply stops appearing and say so in `notes` |
| `result.measurements[]` | the agent's own eval numbers on this output, with the `task/eval_*.json` path | `[]` |
| `result.watch_set_result` | only if a watch set existed | `null` |
| `conclusion.verdict` | `supported`/`contradicted` only if the agent measured against the comparator; else `inconclusive` | — |
| `conclusion.decision` | `adopt` if this output became `final_model`/the submission or the parent of the next launch; `reject` if evaluated and dropped; `abandon_line` if killed | — |
| `conclusion.summary` | the agent's own reading, quoted or closely paraphrased, with `[i]` | `null` |

`elapsed_h`: from event timestamps when the stream has them (turn timestamps
divided into the 10 h budget), else `null`. `created_at`: `null`.

## Proposed additions for reconstructed cards

These are not in the agent-written template; they are what the extraction
needs to be auditable. Prefixed so they can be stripped when a card is handed
to a consumer.

```yaml
provenance:
  kind: extracted                # vs agent-written
  run_ref: r-3f9a2c1b            # opaque; sources.json maps it, outside memory
  launch_i: 562                  # event index of setup.command
  stated_by_agent:
    problem: true | false        # did the agent say this, or was it reconstructed from the launch
    hypothesis: true | false
  anchors:                       # event indices the sections were read from
    problem: [540, 551]
    setup: [560, 562]
    evaluation: [601]
    result: [610, 618]
    conclusion: [622]
  snapshot_files:                # task/ files this card cites, so a reproducer can find them
    - train_v4.py
    - prepare_data_v4.py
    - eval_v4.json
  smoke_runs:                    # pipeline dry runs before this launch; not cards themselves
    - {launch_i: 54, outcome: "crashed: SFTConfig has no evaluation_strategy"}
    - {launch_i: 90, outcome: "passed (2 steps)"}
  extractor: <model id>
  unresolved:                    # what the stream does not settle about this card
    - "final_model may have been overwritten by v5 after [594]; last explicit cp is v4"
```

Two further fields that the trajectories support and the template lacks; both
worth considering for the agent-written version too:

- `setup.data[].repeats` — "x5" is the single most common data decision in the
  corpus and has nowhere to go but `selection`.
- `result.duration_h` and `result.steps` — the agent almost always prints them;
  `training_summary.notes` is too loose a home.

## Checks after extraction

1. Every `setup.command.argv` string is a substring of the digest block at
   `provenance.launch_i` (whitespace-collapsed).
2. Every `provenance.anchors.*` index exists in the digest.
3. No card contains an agent model name, a harness name, or the run/experiment
   string — checked over the serialised YAML.
4. `official_accuracy` is joined by the pipeline onto the single `adopt`ed card
   whose output is the submission, from `sources.json` + the catalogue. The
   extractor never writes it.
