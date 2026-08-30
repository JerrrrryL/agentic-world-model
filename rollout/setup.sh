#!/bin/bash
# Build a PRIVATE PostTrainBench checkout for the crossed rollout.
#
# Not the shared one. /rmeng_data/robtang/PostTrainBench is another session's
# live working directory and a commit there kills every job already running out
# of it (bash reads a script incrementally, so the ESTALE lands hours later when
# the long command returns). This makes an isolated clone pinned to an explicit
# SHA, with its own results dir, and installs the hv_recipe agent into it.
#
# Containers and the HF cache are shared read-only -- they are large, immutable,
# and nothing here writes to them.
#
#   bash rollout/setup.sh
set -euo pipefail

SRC=/rmeng_data/robtang/PostTrainBench
DST=${HV_PTB_DIR:-/rmeng_data/robtang/ptb-hvrecipe}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ ! -d "$DST/.git" ]; then
    echo "cloning $SRC -> $DST"
    git clone --quiet "$SRC" "$DST"
fi
PIN="${HV_PTB_SHA:-$(git -C "$SRC" rev-parse HEAD)}"
git -C "$DST" fetch --quiet origin
git -C "$DST" checkout --quiet --detach "$PIN"
echo "pinned to $(git -C "$DST" rev-parse HEAD)"

for a in hv_recipe hv_noop; do
    install -d "$DST/agents/$a"
    install -m 0755 "$HERE/agents/$a/solve.sh"     "$DST/agents/$a/solve.sh"
    install -m 0644 "$HERE/agents/$a/api_keys.json" "$DST/agents/$a/api_keys.json"
done
mkdir -p "$DST/slurm_logs"

# Own results dir; everything else copied from the shared checkout's .env so the
# container name and caches match what the corpus runs used.
RESULTS=/rmeng_data/robtang/ptb-hvrecipe-results
mkdir -p "$RESULTS"
sed -e "s#^POST_TRAIN_BENCH_RESULTS_DIR=.*#POST_TRAIN_BENCH_RESULTS_DIR=\"$RESULTS\"#" \
    -e 's#^POST_TRAIN_BENCH_EXPERIMENT_NAME=.*##' \
    "$SRC/.env" > "$DST/.env"
chmod 600 "$DST/.env"

echo "results dir : $RESULTS"
echo "checkout    : $DST"
grep -c . "$DST/.env" >/dev/null
echo "agent installed:"; ls -la "$DST/agents/hv_recipe"
