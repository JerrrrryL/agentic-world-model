#!/bin/bash
# claude_wm — study conditions C2 (raw files + WMA) and C3 (WMA over extracted cards).
#
# Two Claude Code sessions in one sandbox, started by this script:
#
#   the world-model agent   cwd /home/ben/wma   its own CLAUDE.md + skills (from the awm repo's wma/),
#                                               pinned to WMA_MODEL, serves `consult` for the whole run
#   the scientist           cwd /home/ben/task  PostTrainBench's own Claude scaffold, the study prompt
#                                               (PTB prompt + one section naming the wma session)
#
# They find each other by name (ListAgents) and talk with SendMessage; the scientist
# owns the GPU and every evaluation, the WMA only answers. Everything the WMA
# writes lands under /home/ben/task/wm/ so PTB's task snapshot keeps it:
# config.json, consults.jsonl, cards/, the WMA session transcript.
#
# AGENT_CONFIG = <scientist model>[:<arm>[:<memory sides>]]
#   claude-opus-4-8:traj             C2 — WMA reads the raw prior runs at /home/ben/prior_runs
#   claude-opus-4-8:retrieval        C3 — WMA reads the extracted cards in /home/ben/wm-memory
#   claude-opus-4-8:llm:train,test   both, both split sides
#   claude-opus-4-8:null             a WMA with no past experiments (control for the mechanism)
set -uo pipefail
AWM_REPO_URL="${AWM_REPO_URL:-https://github.com/JerrrrryL/agentic-world-model.git}"
AWM_REPO_REF="${AWM_REPO_REF:-wm-runtime}"
WMA_MODEL="${WMA_MODEL:-claude-opus-4-8}"

echo "claude_wm starting: AGENT_CONFIG=${AGENT_CONFIG}"
IFS=: read -r MODEL ARM SIDES <<< "${AGENT_CONFIG}"
ARM="${ARM:-null}"; SIDES="${SIDES:-train}"
echo "scientist=${MODEL} wma=${WMA_MODEL} arm=${ARM} memory_sides=${SIDES}"

if [ -f /home/ben/oauth_token ]; then
    export CLAUDE_CODE_OAUTH_TOKEN="$(cat /home/ben/oauth_token)"
else
    echo "ERROR: No oauth_token file found at /home/ben/oauth_token" >&2; exit 1
fi

# --- the awm toolbelt (PYTHONNOUSERSITE=1 in the sandbox: a clone on PYTHONPATH, no pip) ---
git clone --quiet --depth 1 --branch "${AWM_REPO_REF}" "${AWM_REPO_URL}" /home/ben/awm \
    || { echo "ERROR: could not clone ${AWM_REPO_URL}@${AWM_REPO_REF}" >&2; exit 1; }
AWM_SHA="$(git -C /home/ben/awm rev-parse HEAD)"
export PYTHONPATH="/home/ben/awm${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p /home/ben/.local/bin
printf '#!/bin/bash\nexec python3 -m awm.cli "$@"\n' > /home/ben/.local/bin/awm
chmod +x /home/ben/.local/bin/awm
export PATH="/home/ben/.local/bin:${PATH}"
python3 -c "import awm.wm.consult, yaml; print('awm import ok')" || { echo "ERROR: awm does not import" >&2; exit 1; }

# --- the WMA's evidence, per arm ---------------------------------------------------------
export AWM_SESSION_DIR=/home/ben/task
PRIOR=""; MEM=""
case "${ARM}" in
    traj|llm) [ -d /home/ben/prior_runs ] || { echo "ERROR: arm ${ARM} needs /home/ben/prior_runs (PRIOR_RUNS in the pack)" >&2; exit 1; }
              PRIOR=/home/ben/prior_runs ;;
esac
case "${ARM}" in
    retrieval|llm) [ -d /home/ben/wm-memory ] || { echo "ERROR: arm ${ARM} needs /home/ben/wm-memory (WM_MEMORY in the pack)" >&2; exit 1; }
                   MEM=/home/ben/wm-memory ;;
esac
BASE_MODEL="$(printf '%s' "$PROMPT" | grep -oE '`[^`]+/[^`]+`' | head -1 | tr -d '`')"
awm wm init --arm "${ARM}" ${PRIOR:+--prior-runs "$PRIOR"} ${MEM:+--memory-root "$MEM"} \
    --memory-sides "${SIDES}" --wma-model "${WMA_MODEL}" ${BASE_MODEL:+--base-model "$BASE_MODEL"} \
    || { echo "ERROR: awm wm init failed" >&2; exit 1; }
echo "${AWM_SHA}" > /home/ben/task/wm/awm_sha.txt

# --- the world-model agent session -------------------------------------------------------
rm -rf /home/ben/wma && cp -r /home/ben/awm/wma /home/ben/wma
export BASH_MAX_TIMEOUT_MS="36000000"
bash /home/ben/update_agent_cli.sh claude
WMA_LOG=/home/ben/task/wm/wma_session.jsonl
(
  cd /home/ben/wma
  env -u CLAUDECODE -u CLAUDE_CODE_ENTRYPOINT AWM_SESSION_DIR=/home/ben/task \
  claude --print --verbose --model "${WMA_MODEL}" --output-format stream-json \
      --allowedTools "Read,Grep,Glob,ListAgents,SendMessage,Bash(ls:*),Bash(head:*),Bash(tail:*),Bash(wc:*),Bash(grep:*),Bash(rg:*),Bash(find:*),Bash(cat:*),Bash(sleep:*),Bash(awm:*),Bash(python3 -m awm.cli:*),Bash(mkdir:*)" \
      --dangerously-skip-permissions \
      "You are the world-model agent for this session (read CLAUDE.md and the consult skill). A research scientist session will message you; serve its consults for the whole run per the standing order. Begin by running: sleep 120." \
      > "${WMA_LOG}" 2> /home/ben/task/wm/wma_session.err
  echo "wma exit $?" >> /home/ben/task/wm/wma_session.err
) &
WMA_PID=$!
sleep 20
echo "wma session pid=${WMA_PID}; peer sockets: $(ls /tmp/cc-socks 2>/dev/null | wc -l)"

# --- the scientist ------------------------------------------------------------------------
cd /home/ben/task
export CLAUDE_CODE_EFFORT_LEVEL="high"
printf '%s' "$PROMPT" | claude --print --verbose --model "$MODEL" \
    --output-format stream-json --thinking-display summarized \
    --dangerously-skip-permissions
rc=$?
echo "scientist exit ${rc}"

# --- wind down -----------------------------------------------------------------------------
echo "consults: $(wc -l < /home/ben/task/wm/consults.jsonl 2>/dev/null || echo 0)"
kill "${WMA_PID}" 2>/dev/null; sleep 5; kill -9 "${WMA_PID}" 2>/dev/null
ls -la /home/ben/task/final_model 2>/dev/null || echo "no final_model/"
echo "claude_wm done"
