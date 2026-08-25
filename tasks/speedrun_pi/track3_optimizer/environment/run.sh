#!/usr/bin/env bash
# Launch the optimizer baseline on all visible GPUs (single node, {1,2,4,8} x {A,H}100).
#   bash run.sh        # 1 trial  — screen an idea
#   bash run.sh 8      # 8 trials — the fixed validation set (confirm a record)
set -uo pipefail
NGPU="$(nvidia-smi -L | wc -l)"
# Single-node rendezvous on loopback. (torchrun --standalone can hang resolving an
# unroutable node FQDN.) Keep NCCL on the loopback interface for a single-node job,
# and let Triton/Inductor find libcuda without /sbin/ldconfig (missing in some images).
export MASTER_ADDR=127.0.0.1
export NCCL_SOCKET_IFNAME=lo
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib/x86_64-linux-gnu}"
# Watchdog: auto-cancel a hung or runaway run. Default 2h (an 8-trial baseline set
# is ~1h); override with e.g. RUN_TIMEOUT=30m.
RUN_TIMEOUT="${RUN_TIMEOUT:-2h}"
status=0
timeout -k 30s "$RUN_TIMEOUT" \
  torchrun --nnodes=1 --nproc_per_node="${NGPU}" \
    --master_addr=127.0.0.1 --master_port=29500 \
    train_gpt_simple.py "${1:-1}" || status=$?
if [ "$status" -eq 124 ]; then
  echo "ERROR: run exceeded RUN_TIMEOUT=$RUN_TIMEOUT and was cancelled." >&2
fi
exit "$status"
