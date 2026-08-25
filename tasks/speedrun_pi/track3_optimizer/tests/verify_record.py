#!/usr/bin/env python3
"""hierarchy-verifier record checker for the PI NanoGPT optimizer speedrun.

Why this exists
---------------
The upstream `verify.py` that ships to the agent only regexes the final
`step:T/T val_loss:` lines out of ONE logfile and averages them.  It checks

  * nothing about the seeds (8 copies of seed 0 pass just as well as 0..7),
  * nothing about `train_steps` consistency across the 8 trials,
  * nothing about whether the 8 trials came from the same source code.

PI leaned on the frozen seed code plus an LLM monitor and a manual audit at
export time to cover that.  A machine-scored benchmark cannot, so this verifier
re-derives the record from the logs alone:

  1. Read every `logs/*.txt`.  The frozen logging infra writes the *entire source
     of the training script* as the first thing in the file, terminated by a line
     of exactly 100 '=' characters.  SHA-256 of that source is the group key.
  2. Split each logfile into trials on `seed:<n>` lines.  A trial's final val
     loss is the `step:T/T val_loss:` line whose step equals its train_steps; T
     is the trial's `train_steps`.
  3. A group is a VALID RECORD SET iff, pooling all its logfiles, it contains
     exactly the 8 canonical seeds 0xC0FFEE+0..7, each exactly once, all trials
     agree on one `train_steps` T, and the mean of the 8 final val losses is
     < 3.27859.
  4. The score is the LOWEST T over valid groups, reported next to the PI
     baseline (3290) and the human record (2600).

Reward: 0.0 when no group is valid, otherwise
    clip((3290 - T) / (3290 - 2600), 0, 1)
so the unmodified baseline earns 0.0 and matching the human record earns 1.0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants that define the record.  From PI's manifest.json:
#   {"baseline": 3290, "record_bar": 3.27859, "target": 3.28, "human_record": 2600}
# ---------------------------------------------------------------------------
TARGET = 3.28
RECORD_BAR = 3.27859          # 8-trial mean must be strictly below this
BASELINE_STEPS = 3290         # PI's stock recipe clears the bar here
HUMAN_RECORD_STEPS = 2600     # the human frontier PI quotes
SEED_BASE = 0xC0FFEE
N_TRIALS = 8
CANONICAL_SEEDS = tuple(SEED_BASE + i for i in range(N_TRIALS))

# The frozen logger emits `print0(code)` then `print0("="*100)`.
SOURCE_TERMINATOR = "=" * 100
_SEED_RE = re.compile(r"^seed:(\d+)\s*$")
_STEP_RE = re.compile(r"step:(\d+)/(\d+)\s+val_loss:([0-9]*\.?[0-9]+)")
_WORLD_RE = re.compile(r"Running PyTorch (\S+).*? on (.*?) with world_size (\d+)")


@dataclass
class Trial:
    """One `seed:<n>` block inside a logfile."""

    seed: int
    logfile: str
    train_steps: int | None = None
    final_val_loss: float | None = None
    n_val_points: int = 0

    @property
    def complete(self) -> bool:
        return self.train_steps is not None and self.final_val_loss is not None


@dataclass
class LogFile:
    path: str
    source_sha256: str
    source_chars: int
    trials: list[Trial] = field(default_factory=list)
    runtime: str | None = None
    world_size: int | None = None
    parse_error: str | None = None


def split_source(text: str) -> tuple[str | None, str]:
    """Split a logfile into (script source, rest).

    The source is everything before the FIRST line that is exactly 100 '='
    characters.  Using the first occurrence is deliberate: a candidate script
    that plants its own 100-'=' line can only truncate the source it is hashed
    on, which changes the hash and therefore splits it into its own group — it
    cannot make two different scripts collide into one group.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if line.rstrip("\r") == SOURCE_TERMINATOR:
            return "\n".join(lines[:i]), "\n".join(lines[i + 1 :])
    return None, text


def parse_logfile(path: Path) -> LogFile:
    text = path.read_text(errors="replace")
    source, rest = split_source(text)
    if source is None:
        return LogFile(
            path=path.name,
            source_sha256="",
            source_chars=0,
            parse_error="no source-snapshot terminator (a line of 100 '=') found; "
            "the frozen logging block was removed or the log is truncated",
        )

    lf = LogFile(
        path=path.name,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        source_chars=len(source),
    )

    m = _WORLD_RE.search(rest)
    if m:
        lf.runtime = f"torch {m.group(1)} on {m.group(2)}"
        lf.world_size = int(m.group(3))

    current: Trial | None = None
    for line in rest.split("\n"):
        sm = _SEED_RE.match(line.strip())
        if sm:
            current = Trial(seed=int(sm.group(1)), logfile=path.name)
            lf.trials.append(current)
            continue
        pm = _STEP_RE.search(line)
        if pm and current is not None:
            step, total, loss = int(pm.group(1)), int(pm.group(2)), float(pm.group(3))
            current.n_val_points += 1
            if current.train_steps is None:
                current.train_steps = total
            elif current.train_steps != total:
                # train_steps changed mid-trial: impossible from the frozen
                # logger, so treat the trial as unusable.
                current.train_steps = -1
            if step == total:
                current.final_val_loss = loss
    return lf


@dataclass
class Group:
    source_sha256: str
    logfiles: list[str]
    trials: list[Trial]
    valid: bool = False
    reasons: list[str] = field(default_factory=list)
    train_steps: int | None = None
    observed_train_steps: list[int] = field(default_factory=list)
    mean_val_loss: float | None = None
    losses_by_seed: dict[str, float] = field(default_factory=dict)


def build_groups(logfiles: list[LogFile]) -> list[Group]:
    by_hash: dict[str, Group] = {}
    for lf in logfiles:
        if lf.parse_error:
            continue
        g = by_hash.setdefault(lf.source_sha256, Group(lf.source_sha256, [], []))
        g.logfiles.append(lf.path)
        g.trials.extend(lf.trials)
    return list(by_hash.values())


def evaluate_group(g: Group) -> Group:
    reasons: list[str] = []

    seeds = [t.seed for t in g.trials]
    counts: dict[int, int] = {}
    for s in seeds:
        counts[s] = counts.get(s, 0) + 1

    missing = [s for s in CANONICAL_SEEDS if counts.get(s, 0) == 0]
    duplicated = sorted(s for s, c in counts.items() if c > 1)
    foreign = sorted(s for s in counts if s not in CANONICAL_SEEDS)

    if missing:
        reasons.append(
            "missing canonical seed(s) " + ", ".join(str(s) for s in missing)
        )
    if duplicated:
        reasons.append(
            "duplicated seed(s) "
            + ", ".join(f"{s}x{counts[s]}" for s in duplicated)
            + " (a record set is 8 distinct trials, not a re-run of a lucky one)"
        )
    if foreign:
        reasons.append(
            "non-canonical seed(s) " + ", ".join(str(s) for s in foreign)
        )

    incomplete = [t for t in g.trials if t.seed in CANONICAL_SEEDS and not t.complete]
    if incomplete:
        reasons.append(
            "trial(s) without a final step:T/T val_loss line for seed(s) "
            + ", ".join(str(t.seed) for t in incomplete)
        )

    usable = [
        t for t in g.trials if t.seed in CANONICAL_SEEDS and t.complete and counts[t.seed] == 1
    ]
    steps = sorted({t.train_steps for t in usable})
    g.observed_train_steps = sorted({t.train_steps for t in g.trials if t.train_steps})
    if len(steps) > 1:
        reasons.append(
            "trials disagree on train_steps: " + ", ".join(str(s) for s in steps)
        )
    if any(s is not None and s <= 0 for s in steps):
        reasons.append("a trial reported an inconsistent train_steps")

    if len(usable) == N_TRIALS and len(steps) == 1 and steps[0] and steps[0] > 0:
        g.train_steps = steps[0]
        g.losses_by_seed = {str(t.seed): t.final_val_loss for t in usable}
        g.mean_val_loss = sum(t.final_val_loss for t in usable) / N_TRIALS
        if not (g.mean_val_loss < RECORD_BAR):
            reasons.append(
                f"8-trial mean {g.mean_val_loss:.5f} is not below the record bar {RECORD_BAR}"
            )
    elif not reasons:
        reasons.append(f"only {len(usable)} usable trial(s), need {N_TRIALS}")

    g.reasons = reasons
    g.valid = not reasons
    return g


def score(logs_dir: Path) -> dict[str, Any]:
    paths = sorted(p for p in logs_dir.glob("*.txt") if p.is_file())
    logfiles = [parse_logfile(p) for p in paths]
    groups = [evaluate_group(g) for g in build_groups(logfiles)]

    valid = [g for g in groups if g.valid]
    best = min(valid, key=lambda g: g.train_steps) if valid else None

    if best is None:
        reward = 0.0
        best_steps = None
    else:
        best_steps = best.train_steps
        span = BASELINE_STEPS - HUMAN_RECORD_STEPS
        reward = max(0.0, min(1.0, (BASELINE_STEPS - best_steps) / span))
        if math.isnan(reward):
            reward = 0.0

    return {
        "reward": reward,
        "record_valid": best is not None,
        "train_steps": best_steps,
        "mean_val_loss": best.mean_val_loss if best else None,
        "final_val_loss_by_seed": best.losses_by_seed if best else {},
        "source_sha256": best.source_sha256 if best else None,
        "record_logfiles": best.logfiles if best else [],
        "baseline_steps": BASELINE_STEPS,
        "human_record_steps": HUMAN_RECORD_STEPS,
        "record_bar": RECORD_BAR,
        "target": TARGET,
        "canonical_seeds": list(CANONICAL_SEEDS),
        "beats_baseline": bool(best and best_steps < BASELINE_STEPS),
        "beats_human_record": bool(best and best_steps < HUMAN_RECORD_STEPS),
        "n_logfiles": len(paths),
        "n_source_groups": len(groups),
        "logfiles": [
            {
                "path": lf.path,
                "source_sha256": lf.source_sha256[:16],
                "runtime": lf.runtime,
                "world_size": lf.world_size,
                "n_trials": len(lf.trials),
                "seeds": [t.seed for t in lf.trials],
                "parse_error": lf.parse_error,
            }
            for lf in logfiles
        ],
        "groups": [
            {
                "source_sha256": g.source_sha256[:16],
                "logfiles": g.logfiles,
                "n_trials": len(g.trials),
                "seeds": sorted(t.seed for t in g.trials),
                "train_steps": g.train_steps,
                "observed_train_steps": g.observed_train_steps,
                "mean_val_loss": g.mean_val_loss,
                "valid": g.valid,
                "rejected_because": g.reasons,
            }
            for g in groups
        ],
    }


def scalar_rewards(result: dict[str, Any]) -> dict[str, float]:
    """The numbers-only view Harbor will accept as ``reward.json``.

    Nulls become the sentinel the field means when there is no record: no valid
    group scores 0 steps and a NaN mean, not "unknown", because Harbor's schema
    has no room for unknown.
    """
    return {
        "reward": float(result["reward"]),
        "record_valid": float(bool(result["record_valid"])),
        "train_steps": float(result["train_steps"] or 0),
        "mean_val_loss": float(result["mean_val_loss"] or math.nan),
        "n_valid_seeds": float(len(result.get("final_val_loss_by_seed") or {})),
        "n_logfiles": float(result["n_logfiles"]),
        "n_source_groups": float(result["n_source_groups"]),
        "baseline_steps": float(result["baseline_steps"]),
        "human_record_steps": float(result["human_record_steps"]),
        "record_bar": float(result["record_bar"]),
        "beats_baseline": float(bool(result["beats_baseline"])),
        "beats_human_record": float(bool(result["beats_human_record"])),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs-dir", default="/app/logs", help="directory holding logs/*.txt")
    ap.add_argument("--out", default=None, help="write reward.json here")
    args = ap.parse_args(argv)

    result = score(Path(args.logs_dir))

    print(f"logs dir      : {args.logs_dir}")
    print(f"logfiles      : {result['n_logfiles']}  ->  {result['n_source_groups']} source group(s)")
    for g in result["groups"]:
        head = (
            f"  group {g['source_sha256']}  n_trials={g['n_trials']}  "
            f"train_steps={g['observed_train_steps']}  seeds={g['seeds']}"
        )
        print(head)
        if g["mean_val_loss"] is not None:
            print(f"      mean final val_loss = {g['mean_val_loss']:.5f}  (bar < {RECORD_BAR})")
        if g["valid"]:
            print("      VALID RECORD SET")
        else:
            for r in g["rejected_because"]:
                print(f"      REJECTED: {r}")
    print("-" * 72)
    if result["record_valid"]:
        print(
            f"RESULT: valid record at train_steps={result['train_steps']} "
            f"(baseline {BASELINE_STEPS}, human record {HUMAN_RECORD_STEPS})"
        )
    else:
        print("RESULT: no valid 8-seed record set found")
    print(f"reward: {result['reward']:.6f}")

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        # Harbor validates reward.json into VerifierResult.rewards, which is
        # dict[str, float] — a null, list or nested dict in there fails the whole
        # trial with a pydantic ValidationError even when the reward is correct.
        # So reward.json carries only scalars, and the diagnostics that make a
        # rejection explainable (per-group reasons, per-logfile seeds) go to a
        # sibling file.
        out.write_text(json.dumps(scalar_rewards(result), indent=2) + "\n")
        detail = out.with_name("speedrun_record.json")
        detail.write_text(json.dumps(result, indent=2) + "\n")
        print(f"wrote {out} and {detail}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
