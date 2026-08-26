#!/usr/bin/env bash
# Reference "solution": the UNMODIFIED baseline.
#
# There is no oracle for this task — the whole point is that the agent must find
# an optimizer that clears the bar in fewer steps than 3290.  Running the stock
# recipe therefore produces a *valid but unimproved* record set: reward 0.0 by
# construction (clip((3290-3290)/(3290-2600)) == 0).  Its job is to prove the
# environment, the training script, the log format and the verifier all work.
#
# Smoke knobs (all default to the official protocol):
#   AWM_SPEEDRUN_TRIALS      number of trials              (default 8)
#   AWM_SPEEDRUN_TRAIN_STEPS overrides train_steps in the  (default: leave 3290)
#                           editable section of the script
#   RUN_TIMEOUT             run.sh watchdog               (default 2h)
set -euo pipefail

cd /app

TRIALS="${AWM_SPEEDRUN_TRIALS:-8}"

if [ -n "${AWM_SPEEDRUN_TRAIN_STEPS:-}" ]; then
  # `train_steps` is a plain assignment inside the editable section — editing it
  # is exactly what an agent does.  Keep the rest of the script byte-identical so
  # the source hash the verifier groups on stays meaningful.
  sed -i "s/^    train_steps = 3290$/    train_steps = ${AWM_SPEEDRUN_TRAIN_STEPS}/" train_gpt_simple.py
  grep -n "train_steps = " train_gpt_simple.py | head -3
fi

bash run.sh "$TRIALS"
