---
name: experiment-card
description: Use whenever proposing, launching, monitoring, stopping, or interpreting a model-training or evaluation experiment. Creates a reproducible card before compute and a scientist-owned result afterward.
---

# Experiment card workflow

Treat one candidate-producing intervention as one experiment. The unit saved to
memory is the frozen card plus its full event ledger and scientist result.

## Before launch

1. Pick a unique directory below `${AWM_EXPERIMENT_ROOT:-data/experiments}`.
2. Run:

   ```bash
   python3 -m awm.cli experiment scaffold <directory> --id <id> --title <title>
   ```

3. Edit `card.yaml`. Replace every `REPLACE` placeholder. The observed problem
   must cite exact rollouts. Every artifact gets a direct path. Commands are argv
   lists, never opaque prose or shell pipelines.
4. Ensure the hypothesis is testable. Keep hopeful target numbers out of it.
5. Define at least one matched performance protocol. Define a diagnostic protocol
   if the mechanism will be assessed.
6. Validate and freeze:

   ```bash
   python3 -m awm.cli experiment validate <directory>
   python3 -m awm.cli experiment freeze <directory>
   ```

After freeze, `card.yaml` is immutable. A material plan change is a new card.

## Execute and observe

Run the card synchronously or under the detached local worker:

```bash
python3 -m awm.cli experiment run <directory>
python3 -m awm.cli experiment run <directory> --detach
```

On Slurm, put `sbatch --wait ...` in the phase argv. A bare `sbatch` only proves
submission, not experiment completion.

Training/evaluation scripts receive `AWM_EXPERIMENT_DIR`, `AWM_EXPERIMENT_ID`,
`AWM_PHASE_ID`, and `AWM_OBSERVATIONS_PATH`. Record causal intermediate evidence:

```bash
python3 -m awm.cli experiment observe <directory> \
  --kind diagnostic --phase <phase-id> --summary "..." \
  --data '{"metric":"...","value":0.0,"n":0}'
```

Do not call an unevaluated checkpoint bad, and do not assign zero to a missing
measurement. Preserve failures and rejected candidates.

## Scientist review

Execution creates `result.yaml` as an intentionally incomplete draft. Inspect
`events.jsonl`, `run_summary.json`, phase logs, artifacts, and measurement files.
Then replace every TODO.

- `result` contains physical observations only.
- `scientist_assessment.outcome` judges the performance claim.
- `scientist_assessment.mechanism` judges the proposed mechanism.
- `scientist_decision` records adopt/reject/continue/retry/abandon.
- Use supported/contradicted only with evidence references. Aggregate score can
  assess performance but cannot by itself validate a mechanism.

Finish with:

```bash
python3 -m awm.cli experiment finalize <directory>
```

Do not invent a verdict to make validation pass. Use `inconclusive` or
`not_tested` when the evidence is absent, censored, or protocol-incomparable.

