#!/bin/bash
# claude_fulltraj_noawm — study condition C1, the full-information baseline.
#
# Identical to PostTrainBench's own claude_non_api scaffold: same Claude Code
# invocation, same OAuth token, same effort level. The two things that differ
# are outside this file: the prompt (POST_TRAIN_BENCH_PROMPT=prompt_fulltraj,
# which adds a "Prior runs" section) and the read-only bind of the prior runs at
# /home/ben/prior_runs (POST_TRAIN_BENCH_EXTRA_BINDS, see
# rollout/patches/apply_extra_binds.py). No world-model agent, no runtime.
#
# AGENT_CONFIG is the Claude model id (claude-opus-4-6 | claude-opus-4-8 | claude-opus-5).
set -uo pipefail
echo "claude_fulltraj_noawm starting: model=${AGENT_CONFIG}"

if [ -f /home/ben/oauth_token ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$(cat /home/ben/oauth_token)"
else
    echo "ERROR: No oauth_token file found at /home/ben/oauth_token" >&2
    exit 1
fi

if [ -d /home/ben/prior_runs ]; then
    echo "prior_runs: $(find /home/ben/prior_runs -maxdepth 2 -mindepth 2 -type d | wc -l) run dirs, index: $(test -f /home/ben/prior_runs/INDEX.md && echo present || echo MISSING)"
else
    echo "WARNING: /home/ben/prior_runs is not mounted; this cell is running as a plain claude_non_api baseline" >&2
fi

export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="high"

bash /home/ben/update_agent_cli.sh claude

printf '%s' "$PROMPT" | claude --print --verbose --model "$AGENT_CONFIG" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions
rc=$?
echo "claude exit ${rc}"
ls -la /home/ben/task/final_model 2>/dev/null || echo "no final_model/"
echo "claude_fulltraj_noawm done"
