"""Does the cheap extraction agree with the expensive one?

`tools/extract_recipes.py` replaces a ~8-call-per-run pipeline (extract, 2-4 LLM
review lenses, up to 6 repair rounds) with one call, a mechanical anchor check,
and at most one repair. Scaling 143 records to 1175 only buys statistical power
if the 1032 new ones are the same kind of measurement as the 143 old ones. If
the cheap tier is noisier, the extra rows add variance and the power gain is
imaginary -- worse, the 143 gold rows become a distinguishable subpopulation
inside the corpus.

So: run the cheap tier over the same 143 runs and score the two against each
other, field by field. Neither side is ground truth. Disagreement is a bound on
how much of what an extraction says is the extraction rather than the run.

Read the numbers as follows. Pipeline signature and stage count are the coarse
skeleton -- these should agree; if they do not, the two tiers disagree about
what the run *did*. Dataset ids and hyperparameters are the fine detail the
whole re-extraction exists to get; agreement there is the number that decides
whether `recipe_signal.py` can trust a finer `feat()`.

Run: python3 tools/extract_agree.py   (after extract_recipes.py has finished)
"""

from __future__ import annotations

import collections
import json
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT)]

from awm import paths  # noqa: E402

GOLD = ROOT / "splits/posttrainbench/gsm8k-gemma-holdout-v1.recipes.jsonl"
TIER1 = paths.data_root() / "recipes/posttrainbench/tier1.jsonl"

HP_NUM = ("lr", "epochs", "batch_size", "grad_accum", "max_seq_len", "warmup",
          "weight_decay")


def load(path):
    """Key on `experiment/run_name`.

    The two files disagree about what `run` holds -- gold writes
    `experiment/run_name`, tier-1 writes the run name alone -- so joining on the
    raw field silently matches nothing and prints a clean "0 runs in both".
    """
    out = {}
    for line in path.open():
        r = json.loads(line)
        if r.get("extraction", {}).get("ok") is False:
            continue
        run = r["run"]
        key = run if "/" in run else f'{r.get("experiment", "?")}/{run}'
        out[key] = r
    return out


def sig(rec):
    return ">".join(rec.get("pipeline") or []) or "(none)"


def ids(rec):
    s = set()
    for d in rec.get("datasets") or []:
        v = (d.get("dataset_id") or d.get("name") or "").strip().lower()
        if v:
            s.add(v.split("(")[0].strip().rstrip(",;"))
    return s


def jaccard(a, b):
    return 1.0 if not a and not b else len(a & b) / max(1, len(a | b))


def close(x, y, rel=0.02):
    if x is None and y is None:
        return "both-null"
    if x is None or y is None:
        return "one-null"
    try:
        if x == y or abs(x - y) <= rel * max(abs(x), abs(y), 1e-12):
            return "agree"
    except TypeError:
        return "agree" if str(x) == str(y) else "differ"
    return "differ"


def bar(frac, width=24):
    n = round(frac * width)
    return "#" * n + "." * (width - n)


def main() -> int:
    if not TIER1.exists():
        print(f"no tier-1 output at {TIER1}; run tools/extract_recipes.py first")
        return 1
    gold, tier1 = load(GOLD), load(TIER1)
    runs = sorted(set(gold) & set(tier1))
    print(f"{len(gold)} gold records, {len(tier1)} tier-1 records, "
          f"{len(runs)} runs in both\n")
    if not runs:
        return 1

    print("=== 1. the coarse skeleton: do the two tiers agree what the run did? ===")
    same_sig = [r for r in runs if sig(gold[r]) == sig(tier1[r])]
    dn = [len(tier1[r].get("algorithms") or []) - len(gold[r].get("algorithms") or [])
          for r in runs]
    print(f"  pipeline signature identical  {len(same_sig)/len(runs):6.1%}  "
          f"{bar(len(same_sig)/len(runs))}  ({len(same_sig)}/{len(runs)})")
    print(f"  stage count identical         {sum(1 for d in dn if d == 0)/len(runs):6.1%}  "
          f"{bar(sum(1 for d in dn if d==0)/len(runs))}  "
          f"median delta {st.median(dn):+.0f}, tier-1 finds more in "
          f"{sum(1 for d in dn if d > 0)}, fewer in {sum(1 for d in dn if d < 0)}")
    dis = collections.Counter((sig(gold[r]), sig(tier1[r])) for r in runs
                              if sig(gold[r]) != sig(tier1[r]))
    if dis:
        print(f'\n  {"gold":>22}  {"tier-1":>22}   n')
        for (g, t), n in dis.most_common(8):
            print(f"  {g[:22]:>22}  {t[:22]:>22}  {n:2d}")

    # Split the disagreement in two. "Which families" is a claim about what
    # kind of training happened; "how many times" is a claim about how many
    # re-runs of the same family count as shipped stages rather than as
    # discarded attempts. They are not the same question and the second is the
    # one the two tiers actually argue about.
    def fams(rec):
        return "".join(sorted(set(rec.get("pipeline") or []))) or "(none)"

    def collapse(rec):
        out = []
        for f in rec.get("pipeline") or []:
            if not out or out[-1] != f:
                out.append(f)
        return ">".join(out) or "(none)"

    same_set = sum(1 for r in runs if fams(gold[r]) == fams(tier1[r]))
    same_col = sum(1 for r in runs if collapse(gold[r]) == collapse(tier1[r]))
    dlen = [len(tier1[r].get("pipeline") or []) - len(gold[r].get("pipeline") or [])
            for r in runs]
    print(f"\n  same signature, exactly           {len(same_sig)/len(runs):6.1%}")
    print(f"  same after collapsing repeats     {same_col/len(runs):6.1%}   "
          f"(`sft>sft` == `sft`)")
    print(f"  same set of families, any order   {same_set/len(runs):6.1%}")
    print(f"  pipeline length: tier-1 shorter in {sum(1 for d in dlen if d < 0)}, "
          f"longer in {sum(1 for d in dlen if d > 0)}, equal in "
          f"{sum(1 for d in dlen if d == 0)}")

    print("\n=== 2. the fine detail this re-extraction exists to get ===")
    jac = [jaccard(ids(gold[r]), ids(tier1[r])) for r in runs]
    nd = [len(tier1[r].get("datasets") or []) - len(gold[r].get("datasets") or [])
          for r in runs]
    print(f"  dataset-id Jaccard   mean {st.mean(jac):.3f}  median "
          f"{st.median(jac):.3f}  exact {sum(1 for j in jac if j == 1)/len(jac):.1%}"
          f"  disjoint {sum(1 for j in jac if j == 0)/len(jac):.1%}")
    print(f"  dataset count        identical {sum(1 for d in nd if d == 0)/len(nd):.1%}, "
          f"tier-1 lists more in {sum(1 for d in nd if d > 0)}, fewer in "
          f"{sum(1 for d in nd if d < 0)}")

    print(f'\n{"hyperparameter":>16} {"agree":>7} {"differ":>7} {"one-null":>9} '
          f'{"both-null":>10}   both-present agreement')
    print("  " + "-" * 76)
    for k in HP_NUM + ("scheduler", "precision"):
        c = collections.Counter(
            close((gold[r].get("hyperparams") or {}).get(k),
                  (tier1[r].get("hyperparams") or {}).get(k))
            for r in runs)
        both = c["agree"] + c["differ"]
        rate = c["agree"] / both if both else float("nan")
        print(f'{k:>16} {c["agree"]:7d} {c["differ"]:7d} {c["one-null"]:9d} '
              f'{c["both-null"]:10d}   {rate:6.1%} {bar(rate if both else 0, 16)}')
    tot = collections.Counter(
        close(gold[r].get("total_train_examples"),
              tier1[r].get("total_train_examples"), rel=0.05) for r in runs)
    print(f'\n  total_train_examples (5% tol): ' +
          "  ".join(f"{k}={v}" for k, v in sorted(tot.items())))

    print("\n=== 3. what the cheap tier cost in coverage and self-report ===")
    for name, src in (("gold", gold), ("tier-1", tier1)):
        sub = {r: src[r] for r in runs}
        conf = collections.Counter(v.get("confidence") for v in sub.values())
        nn = sum(1 for v in sub.values()
                 for k in HP_NUM if (v.get("hyperparams") or {}).get(k) is not None)
        ne = sum(1 for v in sub.values() for d in (v.get("datasets") or [])
                 if d.get("n_examples") is not None)
        nsh = sum(1 for v in sub.values() for d in (v.get("datasets") or [])
                  if d.get("share") is not None)
        nds = sum(len(v.get("datasets") or []) for v in sub.values())
        print(f"  {name:>6}  confidence " +
              " ".join(f"{k}={conf.get(k,0)}" for k in ("high", "medium", "low")) +
              f"   hyperparams filled {nn}/{len(sub)*len(HP_NUM)} "
              f"({nn/(len(sub)*len(HP_NUM)):.0%})"
              f"   n_examples {ne}/{nds} ({ne/max(nds,1):.0%})"
              f"   share {nsh}/{nds} ({nsh/max(nds,1):.0%})")

    st1 = collections.Counter(tier1[r]["extraction"].get("status") for r in runs)
    rep = collections.Counter(tier1[r]["extraction"].get("repair_round") for r in runs)
    print(f"\n  tier-1 on these runs: " + " ".join(f"{k}={v}" for k, v in st1.items())
          + " | repair rounds " + " ".join(f"{k}:{v}" for k, v in sorted(rep.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
