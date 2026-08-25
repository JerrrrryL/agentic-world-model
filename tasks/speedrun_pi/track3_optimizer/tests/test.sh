#!/usr/bin/env bash
# Harbor verifier.  Mounted at /tests only at verify time; must write
# /logs/verifier/reward.json.  Scores the run from /app/logs/*.txt alone —
# see tests/verify_record.py for the rules (source-hash grouping, the 8
# canonical seeds, the 3.27859 mean bar, lowest train_steps wins).
set -uo pipefail

LOGS_DIR="${HV_SPEEDRUN_LOGS_DIR:-/app/logs}"
OUT_DIR="/logs/verifier"
mkdir -p "$OUT_DIR"

PY=python3
[ -x /opt/venv/bin/python ] && PY=/opt/venv/bin/python

"$PY" /tests/verify_record.py --logs-dir "$LOGS_DIR" --out "$OUT_DIR/reward.json"
status=$?

if [ ! -s "$OUT_DIR/reward.json" ]; then
  # The verifier itself failed (missing logs dir, unreadable file, crash).
  # Harbor still needs a reward, and a crash is not a record.
  printf '{"reward": 0.0, "record_valid": false, "verifier_error": "verify_record.py exited %s without writing reward.json"}\n' \
    "$status" > "$OUT_DIR/reward.json"
fi

# reward.txt is Harbor's fallback path.
"$PY" -c "import json,sys; print(json.load(open('$OUT_DIR/reward.json'))['reward'])" > "$OUT_DIR/reward.txt"

exit 0
