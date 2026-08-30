"""The one table the decision is made from.

``run.py`` prints a full page per design, which is the right shape for auditing a
single design and the wrong shape for choosing between sixteen. This collapses
each design to the four numbers that actually separate them:

  * **leakage** — the share of test rows whose agent / family / experiment was
    also seen in train. A design that scores well with leakage above zero scored
    well by remembering.
  * **dumb** — the best parameter-free lookup's top-3 regret. This is the number
    a learned predictor has to beat, so it is the floor of the useful range.
  * **floor** — the top-3 regret a *perfect* predictor still pays because the
    labels carry re-run noise. Nothing can go below it.
  * **winnable** — ``dumb − floor``, and the same quantity as a share of the mean
    accuracy spread inside a choice set. That share is the comparable one: an
    absolute regret gap means different things on a benchmark where runs differ
    by 40 points and one where they differ by 4.

A design whose winnable share is zero or negative is saturated, whatever its
holdout looks like on paper. Everything is re-derived from ``battery.py`` so the
control in ``run.py`` guards these numbers too.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import battery as B  # noqa: E402
import ceiling as C  # noqa: E402
import run as R  # noqa: E402

DESIGNS = [R.CONTROL] + R.load(str(HERE / "designs/owner.py")) \
                      + R.load(str(HERE / "designs/owner2.py"))


NAN = float("nan")


def row(design):
    res = B.evaluate(design, R.POP)
    if design is R.CONTROL:
        R.check_control(res)
    if "verdict" in res:                       # UNDEFINED — nothing to compare
        return {"name": design.name, "verdict": res["verdict"], "win_share": NAN,
                "n_test": res["n_test"], "labelled": res.get("test_labels_defined", 0)}
    best = min((b for b in res["baselines"] if b["regret@3"] == b["regret@3"]),
               key=lambda b: b["regret@3"], default=None)
    fl = C.regret_floor(design, R.POP)
    mc = C.metadata_ceiling(design, R.POP)
    dumb = best["regret@3"] if best else NAN
    floor = fl["floor@3"] if fl else NAN
    spread = res["headroom"]["mean_within-choice-set accuracy spread"]
    win = dumb - floor
    return {
        "name": design.name,
        "n_test": res["n_test"],
        "labelled": res["test_labels_defined"],
        "sets": res["choice_sets"]["n"],
        "median_set": res["choice_sets"]["median_size"],
        "singletons": res["choice_sets"]["singletons"],
        "leak_agent": res["seen_agent_model"],
        "leak_family": res["seen_agent_family"],
        "leak_exp": res["leak_share_experiment"],
        "dumb": dumb,
        "dumb_by": best["baseline"] if best else "-",
        "floor": floor,
        "win": win,
        "win_share": win / spread if spread and spread == spread else NAN,
        "gbdt": mc["regret@3"] if mc else NAN,
        "gbdt_spearman": mc["spearman"] if mc else NAN,
        "spread": spread,
        "explained": res["noise"]["explainable_variance_share"],
        "replicate_groups": res["noise"]["groups_with_replicates"],
    }


def main() -> int:
    rows = []
    for d in DESIGNS:
        try:
            rows.append(row(d))
        except SystemExit:
            raise
        except Exception as exc:
            print(f"!! {d.name}: {type(exc).__name__}: {exc}", file=sys.stderr)
    rows.sort(key=lambda r: (-(r["win_share"] if r["win_share"] == r["win_share"] else -9)))

    head = (f'{"win%":>6} {"winnable":>9} {"dumb@3":>7} {"floor@3":>8} {"gbdt@3":>7} '
            f'{"n_test":>7} {"lab":>5} {"sets":>5} {"med":>4} {"1s":>3} '
            f'{"lk_ag":>6} {"lk_fam":>7} {"lk_exp":>7}  design')
    print(head)
    print("-" * len(head))
    for r in rows:
        if "verdict" in r:
            print(f'{"  --":>6} {"":>9} {"":>7} {"":>8} {"":>7} '
                  f'{r["n_test"]:7d} {r["labelled"]:5d}'
                  f'{"":>25}  {r["name"]}')
            print(f'{"":>6} !! {r["verdict"]}')
            continue
        ws = f'{100*r["win_share"]:5.1f}%' if r["win_share"] == r["win_share"] else "    --"
        print(f'{ws:>6} {r["win"]:+9.4f} {r["dumb"]:7.4f} {r["floor"]:8.4f} {r["gbdt"]:7.4f} '
              f'{r["n_test"]:7d} {r["labelled"]:5d} {r["sets"]:5d} {r["median_set"]:4d} '
              f'{r["singletons"]:3d} '
              f'{100*r["leak_agent"]:5.0f}% {100*r["leak_family"]:6.0f}% {100*r["leak_exp"]:6.0f}%'
              f'  {r["name"]}')
        print(f'{"":>6} dumb baseline = {r["dumb_by"]:<18s} within-set spread {r["spread"]:.4f}'
              f'   gbdt rho {r["gbdt_spearman"]:+.3f}'
              f'   explainable {100*r["explained"]:.0f}% ({r["replicate_groups"]} replicate groups)')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
