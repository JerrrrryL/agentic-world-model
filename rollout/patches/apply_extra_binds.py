#!/usr/bin/env python3
"""Teach PostTrainBench's run_task.sh one env var: POST_TRAIN_BENCH_EXTRA_BINDS.

The agent sandbox is launched with ``-c --cleanenv`` and a fixed set of
``--bind`` mounts; nothing lets a study mount extra read-only data (the prior
runs, the WMA memory) into it. This adds::

    POST_TRAIN_BENCH_EXTRA_BINDS="/host/prior_runs:/home/ben/prior_runs:ro,/host/mem:/home/ben/wm-memory"

which becomes one ``--bind src:dst[:ro]`` per comma-separated entry on the
``apptainer exec`` line. Idempotent: running it twice changes nothing. Applied
by rollout/setup.sh to the private checkout, never to the shared one.

    python rollout/patches/apply_extra_binds.py <ptb>/src/run_task.sh
"""

from __future__ import annotations

import sys
from pathlib import Path

MARK = "# --- awm: extra binds (rollout/patches/apply_extra_binds.py) ---"

BLOCK = f'''{MARK}
    # POST_TRAIN_BENCH_EXTRA_BINDS="src:dst[:ro],src2:dst2" -> one --bind each. Read on
    # the host (this script), so nothing about it reaches the sandbox environment.
    EXTRA_BIND_ARGS=()
    if [ -n "${{POST_TRAIN_BENCH_EXTRA_BINDS:-}}" ]; then
        IFS=',' read -r -a _awm_binds <<< "${{POST_TRAIN_BENCH_EXTRA_BINDS}}"
        for _b in "${{_awm_binds[@]}}"; do
            [ -n "$_b" ] || continue
            _src="${{_b%%:*}}"
            [ -e "$_src" ] || {{ echo "ERROR: extra bind source missing: $_src" >&2; exit 1; }}
            EXTRA_BIND_ARGS+=(--bind "$_b")
        done
        echo "extra binds: ${{POST_TRAIN_BENCH_EXTRA_BINDS}}"
    fi
'''

ANCHOR_BEFORE = '    timeout --signal=TERM --kill-after=30s "$((NUM_HOURS * 60 + 5))m" \\\n    apptainer exec \\\n'
BIND_LINE = '        --bind "${HF_MERGED}:${HF_HOME_NEW}" \\\n'
NEW_BIND = '        "${EXTRA_BIND_ARGS[@]}" \\\n'


def apply(text: str) -> str:
    if MARK in text:
        return text
    if text.count(ANCHOR_BEFORE) != 1:
        raise SystemExit("run_task.sh: cannot find the agent apptainer exec block exactly once; "
                         "the runner changed shape — update apply_extra_binds.py")
    text = text.replace(ANCHOR_BEFORE, BLOCK + ANCHOR_BEFORE, 1)
    # only the agent exec (the first occurrence after our block) gets the extra binds
    head, tail = text.split(MARK, 1)
    if BIND_LINE not in tail:
        raise SystemExit("run_task.sh: HF cache bind line not found after the exec block")
    tail = tail.replace(BIND_LINE, BIND_LINE + NEW_BIND, 1)
    return head + MARK + tail


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "src/run_task.sh")
    text = path.read_text()
    new = apply(text)
    if new == text:
        print(f"{path}: already patched")
        return 0
    path.write_text(new)
    print(f"{path}: patched (POST_TRAIN_BENCH_EXTRA_BINDS)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
