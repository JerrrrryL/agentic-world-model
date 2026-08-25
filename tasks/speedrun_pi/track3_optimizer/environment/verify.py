#!/usr/bin/env python3
"""Significance check for optimizer-benchmark runs.

Validation is either a single screening run or the fixed 8-trial set:
  - a single run with final val loss < 3.276 looks promising (screen only);
  - the 8-trial set (seeds 0xC0FFEE+0..7, `bash run.sh 8`) is a record iff its
    mean < 3.27859  ==  (3.28 - mean) * sqrt(8) >= 0.004  with sigma = 0.0013
    (one-sided p < 0.001).

Usage:
    python verify.py logs/<uuid>.txt           # parse final val losses from a logfile
    python verify.py --loss 3.2781 3.2790 ...   # pass final losses directly
"""

import sys, re, glob, argparse

TARGET = 3.28
RECORD_BAR = 3.27859   # 8-trial mean must be below this  (= 3.28 - 0.004/sqrt(8))
SCREEN_BAR = 3.276     # a single run below this looks promising (= 3.28 - 0.004)


def final_losses_from_file(path):
    """Return every final-step val loss in a logfile (one per trial)."""
    out = []
    with open(path) as f:
        for line in f:
            m = re.search(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)", line)
            if m and m.group(1) == m.group(2):  # step == train_steps -> final
                out.append(float(m.group(3)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="logfiles (globs ok) to parse")
    ap.add_argument("--loss", type=float, nargs="+", default=[], help="final val losses, passed directly")
    args = ap.parse_args()

    losses = list(args.loss)
    for p in args.paths:
        for path in sorted(glob.glob(p)):
            fl = final_losses_from_file(path)
            if not fl:
                print(f"  ! no final val_loss found in {path}")
            for v in fl:
                print(f"  {path}: {v:.5f}")
            losses.extend(fl)

    n = len(losses)
    if n == 0:
        print("No losses found. Pass a logfile or --loss values.")
        sys.exit(2)
    mu = sum(losses) / n
    print("-" * 56)
    if n == 8:
        passed = mu < RECORD_BAR
        print(f"8-trial set   mean = {mu:.5f}   (record bar: < {RECORD_BAR})")
        print("RESULT: " + ("PASS  record-valid (p<0.001)" if passed else "FAIL  mean is not below the record bar"))
        sys.exit(0 if passed else 1)
    if n == 1:
        ok = losses[0] < SCREEN_BAR
        print(f"single run    val = {mu:.5f}   (promising if < {SCREEN_BAR})")
        print("RESULT: " + ("PROMISING — confirm with the 8-trial set: bash run.sh 8" if ok else "not low enough to be worth confirming on its own"))
        sys.exit(0 if ok else 1)
    print(f"n = {n}: validation is 1 trial (screen) or 8 trials (record), not {n}.")
    print(f"mean so far = {mu:.5f}. Run the full 8-trial set to confirm a record: bash run.sh 8")
    sys.exit(2)


if __name__ == "__main__":
    main()
