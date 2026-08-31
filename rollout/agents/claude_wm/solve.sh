#!/bin/bash
# claude_wm — study conditions C2 (raw files + WMA) and C3 (WMA with seeded memory).
#
# The same Claude Code invocation as claude_non_api, plus the world-model
# runtime beside it: this script clones the awm repo at a pinned ref, puts
# `awm` on PATH, initialises /home/ben/task/wm, installs the scientist's
# skill and Stop hook, and hands the agent our prompt (POST_TRAIN_BENCH_PROMPT
# = prompt_wm or prompt_wm_fulltraj, chosen by the pack script). Whether the
# prior runs are mounted at /home/ben/prior_runs is likewise the pack's call.
#
# AGENT_CONFIG = <claude model>[:<arm>[:<memory sides>[:ro]]]
#   claude-opus-4-8                      null arm, memory sides train, read-write
#   claude-opus-4-8:retrieval            retrieval arm
#   claude-opus-5:retrieval:train,test   retrieval over both sides (the "gemma in" version)
#   claude-opus-4-6:retrieval:train:ro   held-out cell: reads memory, never writes
#
# WMA memory is expected at /home/ben/wm-memory (bind it with
# POST_TRAIN_BENCH_EXTRA_BINDS). If it is absent the session gets a private,
# empty memory under /home/ben/wm-memory-local and says so.
set -uo pipefail
AWM_REPO_URL="${AWM_REPO_URL:-https://github.com/JerrrrryL/agentic-world-model.git}"
AWM_REPO_REF="${AWM_REPO_REF:-wm-runtime}"

echo "claude_wm starting: AGENT_CONFIG=${AGENT_CONFIG}"
IFS=: read -r MODEL ARM SIDES RO <<< "${AGENT_CONFIG}"
ARM="${ARM:-null}"; SIDES="${SIDES:-train}"
echo "model=${MODEL} arm=${ARM} memory_sides=${SIDES} readonly=${RO:-no}"

if [ -f /home/ben/oauth_token ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$(cat /home/ben/oauth_token)"
else
    echo "ERROR: No oauth_token file found at /home/ben/oauth_token" >&2
    exit 1
fi

# --- the runtime -----------------------------------------------------------
# PYTHONNOUSERSITE=1 is set in the sandbox, so `pip install --user` would be
# invisible; a clone on PYTHONPATH needs nothing but PyYAML, which transformers
# already pulls in.
git clone --quiet --depth 1 --branch "${AWM_REPO_REF}" "${AWM_REPO_URL}" /home/ben/awm \
    || { echo "ERROR: could not clone ${AWM_REPO_URL}@${AWM_REPO_REF}" >&2; exit 1; }
AWM_SHA="$(git -C /home/ben/awm rev-parse HEAD)"
echo "awm ${AWM_REPO_REF} @ ${AWM_SHA}"
export PYTHONPATH="/home/ben/awm${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /home/ben/.local/bin
printf '#!/bin/bash\nexec python3 -m awm.cli "$@"\n' > /home/ben/.local/bin/awm
chmod +x /home/ben/.local/bin/awm
export PATH="/home/ben/.local/bin:${PATH}"
python3 -c "import awm.wm.runtime, yaml; print('awm import ok')" || { echo "ERROR: awm does not import" >&2; exit 1; }

export AWM_SESSION_DIR=/home/ben/task
cp /home/ben/awm/input/exp-card.template.yaml /home/ben/task/exp-card.template.yaml
rm -rf /home/ben/task/.claude && cp -r /home/ben/awm/.claude /home/ben/task/.claude

MEM=/home/ben/wm-memory
if [ ! -d "$MEM" ]; then
    echo "WARNING: ${MEM} not mounted; using a private empty memory (no precedents)" >&2
    MEM=/home/ben/wm-memory-local; mkdir -p "$MEM"; RO=""
fi
INIT_ARGS=(--arm "${ARM}" --submission /home/ben/task/final_model --submission-mode copy
           --memory-root "${MEM}" --memory-sides "${SIDES}")
[ "${RO:-}" = "ro" ] && INIT_ARGS+=(--memory-readonly --split-side test)
awm wm --dir /home/ben/task init "${INIT_ARGS[@]}" || { echo "ERROR: awm wm init failed" >&2; exit 1; }
echo "${AWM_SHA}" > /home/ben/task/wm/awm_sha.txt
awm wm --dir /home/ben/task memory stats

if [ -d /home/ben/prior_runs ]; then
    echo "prior_runs: $(find /home/ben/prior_runs -maxdepth 2 -mindepth 2 -type d | wc -l) run dirs"
else
    echo "prior_runs: not mounted"
fi

# --- the scientist ----------------------------------------------------------
export BASH_MAX_TIMEOUT_MS="36000000"
export CLAUDE_CODE_EFFORT_LEVEL="high"
bash /home/ben/update_agent_cli.sh claude

printf '%s' "$PROMPT" | claude --print --verbose --model "$MODEL" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions
rc=$?
echo "claude exit ${rc}"

# --- what the runtime knows at the end ------------------------------------
awm wm --dir /home/ben/task status || true
awm wm --dir /home/ben/task pending || true
ls -la /home/ben/task/final_model 2>/dev/null || echo "no final_model/"
echo "claude_wm done"
