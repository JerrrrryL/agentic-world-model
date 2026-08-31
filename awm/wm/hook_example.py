"""The checkpoint hook, for the scientist's training script.

Two ways to use it:

1. Any script — call ``awm_checkpoint`` after each save and act on the code::

       from awm.wm.hook_example import awm_checkpoint, HOOK_YIELD, HOOK_ABORT
       code = awm_checkpoint("exp-03", ckpt_dir, step=global_step, final=is_last)
       if code in (HOOK_YIELD, HOOK_ABORT):
           sys.exit(0)          # stop after the save; the runtime relaunches you on yield

2. HF/TRL ``Trainer`` — add the callback::

       from awm.wm.hook_example import AWMCheckpointCallback
       trainer.add_callback(AWMCheckpointCallback("exp-03"))

   Make sure a save lands on the last step (``save_steps`` divides
   ``max_steps``, or call ``trainer.save_model(...)`` and ``awm_checkpoint(...,
   final=True)`` yourself at the end).

The session directory comes from ``AWM_SESSION_DIR`` or ``--dir``; the
runtime relaunches you with ``setup.resume_argv`` from the card, so your
script must accept ``--resume_from_checkpoint <path>``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HOOK_CONTINUE = 0
HOOK_YIELD = 3
HOOK_ABORT = 4


def awm_checkpoint(card_id: str, checkpoint_dir: str | os.PathLike, *, step: int,
                   final: bool = False, session_dir: str | os.PathLike | None = None) -> int:
    """Report a saved checkpoint; return 0 continue, 3 yield (exit now), 4 abort (exit now)."""
    session = str(session_dir or os.environ.get("AWM_SESSION_DIR") or Path.cwd())
    argv = [sys.executable, "-m", "awm.cli", "wm", "--dir", session, "checkpoint", card_id,
            str(Path(checkpoint_dir).resolve()), "--step", str(int(step))]
    if final:
        argv.append("--final")
    proc = subprocess.run(argv, check=False, text=True, capture_output=True)
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n", flush=True)
    if proc.returncode not in (HOOK_CONTINUE, HOOK_YIELD, HOOK_ABORT):
        # a protocol error must not kill training silently; surface it and continue
        print(f"[awm wm] checkpoint hook error (exit {proc.returncode}):\n{proc.stderr}", file=sys.stderr, flush=True)
        return HOOK_CONTINUE
    out_dir = Path(checkpoint_dir).resolve().parent
    if proc.returncode == HOOK_YIELD:
        (out_dir / ".awm-resume").write_text(str(Path(checkpoint_dir).resolve()) + "\n")
    elif proc.returncode == HOOK_ABORT:
        (out_dir / ".awm-abort").write_text(str(Path(checkpoint_dir).resolve()) + "\n")
    return proc.returncode


try:  # transformers is optional at import time so this file can be copied anywhere
    from transformers import TrainerCallback  # type: ignore
except Exception:  # noqa: BLE001
    TrainerCallback = object  # type: ignore


class AWMCheckpointCallback(TrainerCallback):  # type: ignore[misc]
    """Calls the hook on every save; stops training when told to yield or abort."""

    def __init__(self, card_id: str, session_dir: str | None = None):
        self.card_id = card_id
        self.session_dir = session_dir

    def on_save(self, args, state, control, **kwargs):
        ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        final = bool(state.max_steps) and state.global_step >= state.max_steps
        code = awm_checkpoint(self.card_id, ckpt, step=state.global_step, final=final,
                              session_dir=self.session_dir)
        if code in (HOOK_YIELD, HOOK_ABORT):
            control.should_training_stop = True
        return control
