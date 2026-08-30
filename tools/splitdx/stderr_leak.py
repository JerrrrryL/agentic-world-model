"""Which catalogue columns are the label wearing a different hat?

A split decides *which rows* a model may learn from. It says nothing about
*which columns* it may read, and the PostTrainBench catalogue row is written
**after** the run finishes. Six of its fields are post-execution facts:

    stderr  total_cost_usd  num_turns  duration_ms  time_taken  session_count

`stderr` is not merely correlated with the label, it *is* the label:

    stderr = sqrt(p*(1-p)/n)

with `n` the benchmark's item count, fixed per benchmark. Solving n back out of
`accuracy*(1-accuracy)/stderr**2` returns a constant to machine precision. Below
p = 0.5 the map is strictly increasing, so on any benchmark whose accuracies all
sit below 0.5, ranking by stderr is an **exact** oracle.

That is a statement about the column. What matters for the design is the *cost*,
and the honest way to price it is the score an attacker actually gets under the
proposed 5-fold split -- choosing per benchmark, on train only, whichever ranker
looks best. This script measures that, for stderr alone and for all six columns
at once, against the same bar every other design in `doc/split-redesign.md` is
scored on: pooled top-3 regret versus the best parameter-free lookup.

Run: OMP_NUM_THREADS=4 python3 tools/splitdx/stderr_leak.py   (from the repo root)
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import battery as B  # noqa: E402
import run as R  # noqa: E402

K = 5
#: from ceiling.regret_floor over the same 5 folds; see doc/split-redesign.md
FLOOR = 0.0018


def folds_by_family(pop):
    """The recommended split: 26 agent families greedily packed into 5 folds."""
    counts = collections.Counter(B.agent_family(r["agent_model"]) for r in pop)
    folds, load = [[] for _ in range(K)], [0] * K
    for fam, n in counts.most_common():
        i = int(np.argmin(load))
        folds[i].append(fam)
        load[i] += n
    return folds


def _hhmmss(v):
    if not isinstance(v, str) or v.count(":") != 2:
        return None
    try:
        h, m, s = (int(x) for x in v.split(":"))
    except ValueError:
        return None
    return h * 3600 + m * 60 + s


#: every catalogue column that is only knowable after the run has finished
POST_HOC = {
    "stderr": lambda r: r.get("stderr"),
    "total_cost_usd": lambda r: r.get("total_cost_usd"),
    "num_turns": lambda r: r.get("num_turns"),
    "duration_ms": lambda r: r.get("duration_ms"),
    "time_taken": lambda r: _hhmmss(r.get("time_taken")),
    "session_count": lambda r: r.get("session_count"),
}


def groups(rows):
    """Choice sets: the cell a top-k pick is made within."""
    g = collections.defaultdict(list)
    for j, r in enumerate(rows):
        g[(r["benchmark"], r["trained_model"])].append(j)
    return [np.array(v) for v in g.values() if len(v) > 1]


def regret(pred, acc, gs, k=3):
    return float(np.mean([B.topk_regret(pred[ix], acc[ix], k)["expected"] for ix in gs]))


def signs(rows, col):
    """Ranking direction per benchmark, fitted on train only.

    An attacker does not need to know that stderr rises with accuracy below
    p = 0.5 and falls above it -- one Spearman on the training half tells them.
    """
    get, out = POST_HOC[col], {}
    for b in {r["benchmark"] for r in rows}:
        sub = [r for r in rows if r["benchmark"] == b and get(r) is not None]
        rho = (B.spearman(np.array([float(get(r)) for r in sub]),
                          np.array([r["accuracy"] for r in sub]))
               if len(sub) > 2 else float("nan"))
        out[b] = 1.0 if (rho == rho and rho >= 0) else -1.0
    return out


def score(rows, col, sgn):
    get = POST_HOC[col]
    return np.array([sgn.get(r["benchmark"], 0.0) * float(get(r) or 0.0) for r in rows])


def fold_design(held):
    def split(rows, held=held):
        return ([r for r in rows if B.agent_family(r["agent_model"]) not in held],
                [r for r in rows if B.agent_family(r["agent_model"]) in held])
    return B.Design(name=f'fold {"+".join(sorted(held))[:30]}', split=split,
                    target=B.DELTA_CELL,
                    choice=lambda r: (r["benchmark"], r["trained_model"]))


def honest(design, pop):
    """The best parameter-free lookup -- the bar every attack has to beat."""
    res = B.evaluate(design, pop)
    best = min((b for b in res["baselines"] if b["regret@3"] == b["regret@3"]),
               key=lambda b: b["regret@3"])
    return next(b for b in B.BASELINES if b.name == best["baseline"])


def pooled(rows):
    w = np.array([n for n, *_ in rows], dtype=float)
    w /= w.sum()
    return [np.array([r[i] for r in rows]) @ w for i in range(1, len(rows[0]))]


def main() -> int:  # noqa: C901 -- one report, read top to bottom
    pop = list(R.POP)
    R.check_control(B.evaluate(R.CONTROL, pop))
    print("control reproduces; the numbers below are comparable to the rest of the battery\n")

    # ---- 1. stderr is an algebraic re-encoding of the label
    print("=== 1. stderr = sqrt(accuracy*(1-accuracy)/n): solving for the item count n ===")
    print(f'{"benchmark":>18} {"rows":>6} {"implied n":>10} {"rel. spread":>12} '
          f'{"max acc":>8} {"rho(stderr,acc)":>16}')
    have = [r for r in pop if isinstance(r.get("stderr"), (int, float)) and r["stderr"] > 0]
    exact = []
    for b in sorted({r["benchmark"] for r in have}):
        sub = [r for r in have if r["benchmark"] == b]
        n = np.array([r["accuracy"] * (1 - r["accuracy"]) / r["stderr"] ** 2 for r in sub])
        n = n[n > 0]
        acc = np.array([r["accuracy"] for r in sub], dtype=float)
        rho = B.spearman(np.array([r["stderr"] for r in sub], dtype=float), acc)
        flag = "  <-- all below p=0.5: monotone, so ranking by stderr is EXACT"
        below = acc.max() < 0.5
        if below:
            exact.append(b)
        print(f"{b:>18} {len(sub):6d} {np.median(n):10.1f} "
              f"{(n.max()-n.min())/np.median(n):12.2e} {acc.max():8.4f} {rho:+16.4f}"
              f"{flag if below else ''}")
    print(f"\n  {len(have)}/{len(pop)} rows carry a stderr. A constant implied n means the "
          f"column is\n  a deterministic function of the label, not an independent "
          f"measurement of it.")

    # ---- 2. what it costs the recommended design
    folds = folds_by_family(pop)
    print(f"\n=== 2. cost under the recommended {K}-fold split "
          f"(top-3 regret, lower is better) ===")
    print(f'{"fold":>5} {"n_test":>7} {"cover":>6} {"honest@3":>9} {"stderr@3":>9} '
          f'{"hybrid@3":>9}  benchmarks where stderr won on train')
    print("-" * 108)
    rows_out, per_bench = [], collections.defaultdict(list)
    for i, fs in enumerate(folds):
        d = fold_design(frozenset(fs))
        train, test = d.split(pop)
        hb = honest(d, pop)
        y_tr = d.target.labels(train, d.target.fit_reference(train))
        sgn = signs(train, "stderr")

        # pick per benchmark on TRAIN only -- no test label is touched
        tr_acc = np.array([r["accuracy"] for r in train], dtype=float)
        tg = collections.defaultdict(list)
        for j, r in enumerate(train):
            tg[(r["benchmark"], r["trained_model"])].append(j)
        p_hon_tr = hb.predict(train, y_tr, train)
        s_tr = score(train, "stderr", sgn)
        won = set()
        for b in sorted({r["benchmark"] for r in train}):
            gs = [np.array(v) for k, v in tg.items() if k[0] == b and len(v) > 1]
            if gs and regret(s_tr, tr_acc, gs) < regret(p_hon_tr, tr_acc, gs):
                won.add(b)

        acc = np.array([r["accuracy"] for r in test], dtype=float)
        gs = groups(test)
        p_hon = hb.predict(train, y_tr, test)
        p_std = score(test, "stderr", sgn)
        p_hyb = np.array([p_std[j] if test[j]["benchmark"] in won else p_hon[j]
                          for j in range(len(test))])
        cover = np.mean([r.get("stderr") is not None for r in test])
        rows_out.append((len(test), regret(p_hon, acc, gs), regret(p_std, acc, gs),
                         regret(p_hyb, acc, gs)))
        print(f"{i:5d} {len(test):7d} {cover:5.0%} {rows_out[-1][1]:9.4f} "
              f"{rows_out[-1][2]:9.4f} {rows_out[-1][3]:9.4f}  {', '.join(sorted(won))}")

        for b in sorted({r["benchmark"] for r in test}):
            bg = [ix for ix in gs if test[ix[0]]["benchmark"] == b]
            if bg:
                per_bench[b].append((len(test), regret(p_std, acc, bg)))
    hon, std, hyb = pooled(rows_out)
    print(f'{"POOL":>5} {sum(r[0] for r in rows_out):7d} {"":6} '
          f"{hon:9.4f} {std:9.4f} {hyb:9.4f}")

    print("\n  per benchmark, ranking by stderr alone, pooled over folds:")
    for b, vs in sorted(per_bench.items()):
        v = pooled([(n, x) for n, x in vs])[0]
        print(f"    {b:<18} regret@3 = {v:.4f}"
              + ("   <-- EXACT, the benchmark is solved" if b in exact else ""))

    # ---- 3. every post-hoc column, alone and together
    print(f"\n=== 3. every post-hoc column as a ranker, pooled over the {K} folds ===")
    print(f'{"column":>16} {"cover":>6} {"regret@3":>9} {"vs honest":>10} {"verdict":>10}')
    print("-" * 60)
    print(f'{"(honest best)":>16} {"":6} {hon:9.4f} {"":10}  the bar to beat')
    alone = {}
    for col in POST_HOC:
        acc_rows, cov = [], []
        for fs in folds:
            d = fold_design(frozenset(fs))
            train, test = d.split(pop)
            sgn = signs(train, col)
            acc = np.array([r["accuracy"] for r in test], dtype=float)
            acc_rows.append((len(test), regret(score(test, col, sgn), acc, groups(test))))
            cov.append(np.mean([POST_HOC[col](r) is not None for r in test]))
        alone[col] = pooled(acc_rows)[0]
        print(f"{col:>16} {np.mean(cov):5.0%} {alone[col]:9.4f} {alone[col]-hon:+10.4f} "
              f'{"LEAKS" if alone[col] < hon else "no help":>10}')

    print(f"\n=== 4. all six at once: best of {{honest, 6 columns}} chosen per benchmark "
          f"on train ===")
    print(f'{"fold":>5} {"honest@3":>9} {"attacked@3":>11}   column chosen per benchmark')
    print("-" * 108)
    full = []
    for i, fs in enumerate(folds):
        d = fold_design(frozenset(fs))
        train, test = d.split(pop)
        hb = honest(d, pop)
        y_tr = d.target.labels(train, d.target.fit_reference(train))
        sgn = {c: signs(train, c) for c in POST_HOC}

        tr_acc = np.array([r["accuracy"] for r in train], dtype=float)
        tg = collections.defaultdict(list)
        for j, r in enumerate(train):
            tg[(r["benchmark"], r["trained_model"])].append(j)
        p_hon_tr = hb.predict(train, y_tr, train)
        pick = {}
        for b in sorted({r["benchmark"] for r in train}):
            gs = [np.array(v) for k, v in tg.items() if k[0] == b and len(v) > 1]
            if not gs:
                continue
            best, bs = "honest", regret(p_hon_tr, tr_acc, gs)
            for c in POST_HOC:
                v = regret(score(train, c, sgn[c]), tr_acc, gs)
                if v < bs:
                    best, bs = c, v
            pick[b] = best

        acc = np.array([r["accuracy"] for r in test], dtype=float)
        p_hon = hb.predict(train, y_tr, test)
        cache = {c: score(test, c, sgn[c]) for c in POST_HOC}
        p = np.array([p_hon[j] if pick.get(test[j]["benchmark"], "honest") == "honest"
                      else cache[pick[test[j]["benchmark"]]][j] for j in range(len(test))])
        gs = groups(test)
        full.append((len(test), regret(p_hon, acc, gs), regret(p, acc, gs)))
        print(f"{i:5d} {full[-1][1]:9.4f} {full[-1][2]:11.4f}   "
              + ", ".join(f"{b[:6]}={c}" for b, c in sorted(pick.items()) if c != "honest"))
    _, att = pooled(full)
    print(f'{"POOL":>5} {hon:9.4f} {att:11.4f}')

    print(f"\n  honest best baseline    regret@3 = {hon:.4f}   headroom {hon-FLOOR:+.4f}")
    print(f"  stderr alone                     = {std:.4f}")
    print(f"  stderr, chosen per benchmark     = {hyb:.4f}   headroom {hyb-FLOOR:+.4f}"
          f"   ({100*(1-(hyb-FLOOR)/(hon-FLOOR)):.0f}% destroyed)")
    print(f"  every post-hoc column            = {att:.4f}   headroom {att-FLOOR:+.4f}"
          f"   ({100*(1-(att-FLOOR)/(hon-FLOOR)):.0f}% destroyed)")
    print(f"\n  A split spec that does not name an excluded-columns list is measuring this\n"
          f"  attack, not the thesis. The design survives it -- {att-FLOOR:+.4f} of headroom\n"
          f"  is still positive -- but the honest headline drops from {hon-FLOOR:+.4f} to "
          f"{att-FLOOR:+.4f},\n  and on {len(exact)} of {len(per_bench)} benchmarks stderr "
          f"alone is an exact oracle.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
