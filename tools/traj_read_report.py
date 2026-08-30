"""Score the pairwise arms, and separate reading the recipe from reading the answer.

`traj_read.py` produces the picks. This scores them, and it exists as a separate
file because the interesting part is not the headline accuracy -- it is the
stratification that says whether an arm is doing what its name claims.

Three cuts, in increasing order of how hard they are to argue with:

1. **Redaction.** `raw` minus `redact` is leakage the regex could remove.
   It is a lower bound, because redaction is not complete (it leaves a residue
   that still decides ~6 % of pairs on its own).
2. **Where the regex abstains.** On pairs where no run printed a readable
   score, there is nothing to OCR. An arm that keeps its accuracy here is
   reading the recipe. This is stronger than (1) because it does not depend on
   redaction being thorough -- it only depends on the numbers not being there.
3. **Agreement with the regex.** If an arm's answers track the self-report rule
   pair-for-pair, it is that rule with extra steps, whatever it says in `why`.

Also reported: McNemar between arms on the pairs both answered (the pairs are
shared, so an unpaired CI comparison overstates the uncertainty), leakage
residue per arm, and the per-family breakdown that says whether one family
carries the result.

    python3 tools/traj_read_report.py
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]

import traj_read as T  # noqa: E402


def load_arm(name):
    p = T.OUTDIR / f"{name}.jsonl"
    if not p.exists():
        return None
    got = {}
    for line in p.open():
        try:
            d = json.loads(line)
        except Exception:  # noqa: BLE001
            continue
        got[(d["id"], d["order"])] = d
    return got


def picks_of(recs, pairs):
    out = {}
    for p in pairs:
        out[p["id"]] = (recs.get((p["id"], 0), {}).get("winner"),
                        recs.get((p["id"], 1), {}).get("winner"))
    return out


def verdict(p, picks):
    """1 correct, 0 wrong, 0.5 inconsistent or unanswered."""
    v = picks.get(p["id"])
    if not v or v[0] is None or v[1] is None:
        return 0.5
    if v[0] != v[1]:
        return 0.5
    return 1.0 if v[0] == p["truth"] else 0.0


def acc(pairs, picks):
    if not pairs:
        return float("nan"), 0
    s = [verdict(p, picks) for p in pairs]
    return float(np.mean(s)), len(s)


def boot_ci(pairs, picks, n=4000, seed=1):
    """Bootstrap over pairs. Not Wilson: the scores include 0.5s, and pairs
    from one cell are not independent, so a binomial CI is the wrong shape."""
    if not pairs:
        return float("nan"), float("nan")
    s = np.array([verdict(p, picks) for p in pairs])
    rng = np.random.default_rng(seed)
    d = s[rng.integers(0, len(s), (n, len(s)))].mean(1)
    return float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))


def mcnemar(pairs, pa, pb):
    """Paired sign test on pairs where the two arms differ. Same pairs, so the
    comparison that matters is who wins the disagreements, not two CIs."""
    ab = ba = 0
    for p in pairs:
        x, y = verdict(p, pa), verdict(p, pb)
        if x > y:
            ab += 1
        elif y > x:
            ba += 1
    n = ab + ba
    if n == 0:
        return ab, ba, float("nan")
    # two-sided exact binomial
    k = min(ab, ba)
    pv = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return ab, ba, min(1.0, pv)


def main() -> int:
    rows = T.load_rows()
    pairs = T.build_pairs(rows)
    byid = {p["id"]: p for p in pairs}

    print(f'{len(pairs)} pairs, {len({p["fam"] for p in pairs})} families, '
          f'{len({p["cell"] for p in pairs})} cells\n')

    # --- the free arms, recomputed so everything is scored the same way ------
    arms = {}
    arms["selfreport"] = T.arm_selfreport(pairs, rows)
    arms["features"] = T.arm_features(pairs, rows, "recipe")
    arms["effort"] = T.arm_features(pairs, rows, "effort")
    arms["feat+effort"] = T.arm_features(pairs, rows, "both")
    for name in ("summary", "recipe", "redact", "raw"):
        recs = load_arm(name)
        if recs is None:
            print(f"  (no {name}.jsonl yet)")
            continue
        arms[name] = picks_of(recs, pairs)

    # --- 1. headline ----------------------------------------------------------
    print("\n=== 1. all 540 pairs (abstain and self-contradiction score 0.5) ===")
    print(f'{"arm":>12} {"acc":>7}  {"95% CI":>16}  {"answered":>8} {"consist":>8} '
          f'{"says-A":>7}')
    for name, pk in arms.items():
        a, n = acc(pairs, pk)
        lo, hi = boot_ci(pairs, pk)
        ans = sum(1 for p in pairs
                  if pk.get(p["id"], (None,))[0] is not None
                  and pk[p["id"]][1] is not None)
        con = sum(1 for p in pairs
                  if (v := pk.get(p["id"])) and v[0] and v[1] and v[0] == v[1])
        sa = [w for p in pairs for w in (pk.get(p["id"]) or ()) if w]
        posA = np.mean([w == "A" for w in sa]) if sa else float("nan")
        print(f"{name:>12} {a:7.1%}  [{lo:6.1%},{hi:6.1%}]  {ans/len(pairs):8.1%} "
              f"{con/max(ans,1):8.1%} {posA:7.1%}")

    # --- 2. does the leak explain it? ----------------------------------------
    sr = arms["selfreport"]
    decided = [p for p in pairs if p["id"] in sr]
    abstain = [p for p in pairs if p["id"] not in sr]
    print(f"\n=== 2. split by whether a score is readable off the page ===")
    print(f"    the regex finds a comparable number in {len(decided)} pairs "
          f"and nothing in {len(abstain)}")
    print(f'{"arm":>12} {"regex-decidable":>17} {"regex-blind":>17}   difference')
    for name, pk in arms.items():
        a1, n1 = acc(decided, pk)
        l1, h1 = boot_ci(decided, pk)
        a2, n2 = acc(abstain, pk)
        l2, h2 = boot_ci(abstain, pk)
        print(f"{name:>12} {a1:6.1%} [{l1:5.1%},{h1:5.1%}] "
              f"{a2:6.1%} [{l2:5.1%},{h2:5.1%}]   {100*(a1-a2):+5.1f}pp")
    print(f"    (n = {len(decided)} and {len(abstain)})")

    # --- 3. is an arm just the regex? ----------------------------------------
    print("\n=== 3. agreement with the self-report rule, on pairs it decides ===")
    for name, pk in arms.items():
        if name == "selfreport":
            continue
        same = tot = 0
        for p in decided:
            v = pk.get(p["id"])
            if not v or v[0] is None or v[0] != v[1]:
                continue
            tot += 1
            same += v[0] == sr[p["id"]][0]
        if tot:
            print(f"{name:>12} agrees {same/tot:6.1%} of {tot} pairs "
                  f"(chance is 50%, and both are ~62% correct there)")

    # --- 4. head to head ------------------------------------------------------
    print("\n=== 4. paired sign test, same pairs ===")
    names = list(arms)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            w1, w2, pv = mcnemar(pairs, arms[a], arms[b])
            star = "*" if pv < 0.05 else " "
            print(f"  {a:>12} vs {b:<12} {w1:4d} - {w2:<4d} p={pv:.4f} {star}")

    # --- 5. how much is one family? ------------------------------------------
    print("\n=== 5. per family (arms as columns) ===")
    fams = sorted({p["fam"] for p in pairs},
                  key=lambda f: -sum(p["fam"] == f for p in pairs))
    show = [n for n in ("selfreport", "features", "effort", "summary",
                        "recipe", "redact", "raw") if n in arms]
    print(f'{"family":>22} {"n":>4} ' + " ".join(f"{n:>10}" for n in show))
    for f in fams:
        sub = [p for p in pairs if p["fam"] == f]
        cells = " ".join(f"{acc(sub, arms[n])[0]:10.1%}" for n in show)
        print(f"{f:>22} {len(sub):4d} {cells}")

    # --- 6. by gap ------------------------------------------------------------
    print("\n=== 6. by accuracy gap ===")
    edges = [0.05, 0.10, 0.20, 0.40, 1.01]
    print(f'{"gap":>22} {"n":>4} ' + " ".join(f"{n:>10}" for n in show))
    for lo, hi in zip(edges, edges[1:]):
        sub = [p for p in pairs if lo <= p["gap"] < hi]
        if not sub:
            continue
        cells = " ".join(f"{acc(sub, arms[n])[0]:10.1%}" for n in show)
        print(f"{f'{lo:.2f}-{hi:.2f}':>22} {len(sub):4d} {cells}")

    # --- 7. confidence, and what the model says it used ----------------------
    print("\n=== 7. is the model's own confidence worth anything? ===")
    for name in ("summary", "recipe", "redact", "raw"):
        recs = load_arm(name)
        if not recs:
            continue
        g = collections.defaultdict(lambda: [0, 0])
        for p in pairs:
            f, s = recs.get((p["id"], 0)), recs.get((p["id"], 1))
            if not f or not s or f["winner"] is None or s["winner"] is None:
                continue
            if f["winner"] != s["winner"]:
                continue
            c = min(f.get("confidence") or 0, s.get("confidence") or 0)
            g[c][0] += f["winner"] == p["truth"]
            g[c][1] += 1
        line = "  ".join(f"{c}:{k}/{n}={k/n:.0%}"
                         for c, (k, n) in sorted(g.items()) if n)
        print(f"{name:>12} {line}")

    # --- 8. leakage residue ---------------------------------------------------
    print("\n=== 8. how much score text survived redaction, per arm ===")
    print("    (the exact text each arm was shown, re-run through the same "
          "regex; 200 runs)")
    featkeys = set(T.RS.feat(rows[0]))
    sample = rows[:200]
    for kind in ("recipe", "redact", "raw"):
        if kind not in arms:
            continue
        jobs = []
        for r in sample:
            if kind == "recipe":
                meta = {k: v for k, v in r.items()
                        if k not in T.DROP and k not in featkeys}
            else:
                meta = {k: r.get(k) for k in T.RC.HEADER_KEYS}
            jobs.append((kind, r["experiment"], r["run"], meta))
        tot = hits = 0
        for _, txt in map(T._text_worker, jobs):
            v = T.self_scores(txt)
            tot += len(v)
            hits += bool(v)
        print(f"{kind:>12} {tot:5d} score-shaped numbers, "
              f"{hits}/{len(sample)} runs with at least one")

    # --- 9. what "regex-blind" actually means --------------------------------
    print('\n=== 9. why the regex abstains on those 132 pairs ===')
    worker = [(r["experiment"], r["run"], False) for r in rows]
    best = {}
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(32) as ex:
        for run, v, n in ex.map(T._sr_worker, worker, chunksize=8):
            best[run] = (v, n)
    kinds = collections.Counter()
    for p in abstain:
        x, y = best.get(p["a"]["run"], (None, 0)), best.get(p["b"]["run"], (None, 0))
        if x[0] is None and y[0] is None:
            kinds["neither run printed a score"] += 1
        elif x[0] is None or y[0] is None:
            kinds["only one run printed a score"] += 1
        else:
            kinds["both printed, and the maxima tie"] += 1
    for k, v in kinds.most_common():
        print(f"    {v:4d}  {k}")
    hard = [p for p in abstain
            if best.get(p["a"]["run"], (None,))[0] is None
            and best.get(p["b"]["run"], (None,))[0] is None]
    print(f"\n    on the {len(hard)} pairs where NEITHER side printed a "
          f"readable score:")
    for name in show:
        a, _ = acc(hard, arms[name])
        lo, hi = boot_ci(hard, arms[name])
        print(f"{name:>16} {a:6.1%} [{lo:5.1%},{hi:5.1%}]")

    # --- 10. does the model say it used a number? ----------------------------
    print('\n=== 10. does the stated reason quote a measured SCORE? ===')
    print("    (score-shaped and next to an eval word -- a bare number would "
          "also match every learning rate and warmup fraction)")
    for name in ("summary", "recipe", "redact", "raw"):
        recs = load_arm(name)
        if not recs:
            continue
        cited, plain = [], []
        for p in pairs:
            f, s = recs.get((p["id"], 0)), recs.get((p["id"], 1))
            if not f or not s:
                continue
            txt = f"{f.get('why') or ''} {s.get('why') or ''}"
            (cited if T.self_scores(txt) else plain).append(p)
        ac, _ = acc(cited, arms[name])
        ap, _ = acc(plain, arms[name])
        print(f"{name:>12} quotes a number in {len(cited)/len(pairs):5.1%} of "
              f"pairs: {ac:6.1%} there, {ap:6.1%} on the {len(plain)} where it "
              f"does not")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
