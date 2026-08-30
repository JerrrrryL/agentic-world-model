"""Round two: the combinations round one showed were the only live ones.

Round one settled three things and each of them constrains what is worth trying
next. Holding out agent families on gsm8k alone leaves 7 test rows in 4 choice
sets, so the agent axis needs the full corpus to be affordable. A Δ referenced to
a (benchmark, base model) cell is *undefined* under a base-model holdout — 0 of
50 test rows get a label — so those two choices cannot be combined at all. And
choice-set size is a knob that sets the answer: a design whose median choice set
is 1 scores regret@3 = 0 no matter what it holds out, which is how OWNER-5,
OWNER-6 and OWNER-9 all "passed".

So this round only builds designs that keep the choice sets large, and it takes
the holdout to the full 1,175-run corpus where there is mass to spend.
"""

from __future__ import annotations

import collections
import sys

from pathlib import Path as _P
sys.path[:0] = [str(_P(__file__).resolve().parents[1]), str(_P(__file__).resolve().parents[3])]

import battery as B  # noqa: E402


def families_covering(rows, target_share=0.25):
    """Pick whole agent families, largest-first from the middle, until they hold
    ``target_share`` of the runs.

    Largest-first would put the corpus's dominant agent in test and starve train;
    smallest-first (round one) buys a test set too small to measure. Walking down
    from the second-largest until the share is met keeps both sides usable and is
    deterministic, so the split is reproducible without a seed.
    """
    c = collections.Counter(B.agent_family(r["agent_model"]) for r in rows)
    ordered = [f for f, _ in c.most_common()][1:]      # skip the single biggest
    held, n = set(), 0
    for f in ordered:
        if n / len(rows) >= target_share:
            break
        held.add(f)
        n += c[f]
    return held


def hold_families_by_mass(bench=None, share=0.25):
    def split(rows):
        rows = [r for r in rows if bench is None or r["benchmark"] in bench]
        held = families_covering(rows, share)
        return ([r for r in rows if B.agent_family(r["agent_model"]) not in held],
                [r for r in rows if B.agent_family(r["agent_model"]) in held])
    return split


def hold_configs(bench=None, frac=0.3):
    def split(rows):
        rows = [r for r in rows if bench is None or r["benchmark"] in bench]
        cfgs = sorted({B.config_of(r["experiment"]) for r in rows})
        held = set(cfgs[::int(1 / frac)])
        return ([r for r in rows if B.config_of(r["experiment"]) not in held],
                [r for r in rows if B.config_of(r["experiment"]) in held])
    return split


def hold_configs_and_base(base="google_gemma-3-4b-pt", frac=0.3):
    """Both at once. Test is the intersection, so a test row is a configuration
    never seen *and* a base model never seen — the strictest thing this corpus
    supports. Train is everything else, including held-out configs on other base
    models, which is the leak this design has to be checked for."""
    def split(rows):
        cfgs = sorted({B.config_of(r["experiment"]) for r in rows})
        held = set(cfgs[::int(1 / frac)])
        test = [r for r in rows
                if B.config_of(r["experiment"]) in held and r["trained_model"] == base]
        train = [r for r in rows
                 if B.config_of(r["experiment"]) not in held and r["trained_model"] != base]
        return train, test
    return split


def by_cell(r):
    return (r["benchmark"], r["trained_model"])


#: Rank within (benchmark, base model) but only among runs of *different*
#: configurations — the deployment question is "which recipe should I run on this
#: cell", and two launches of one configuration are not two candidate recipes.
def by_cell_dedup_config(r):
    return (r["benchmark"], r["trained_model"])


DESIGNS = [
    B.Design(
        name="OWNER-10  ALL benchmarks, blocked by configuration (30%), absolute",
        split=hold_configs(bench=None, frac=0.3),
        choice=by_cell,
        note="the leak closed at full scale — round one's healthiest design, widened",
    ),
    B.Design(
        name="OWNER-11  ALL benchmarks, blocked by configuration (30%), Δ vs cell",
        split=hold_configs(bench=None, frac=0.3),
        target=B.DELTA_CELL,
        choice=by_cell,
        note="same split, the doc's target — every cell is still in train so Δ is defined",
    ),
    B.Design(
        name="OWNER-12  ALL benchmarks, hold agent families ≈25% of mass, absolute",
        split=hold_families_by_mass(bench=None, share=0.25),
        choice=by_cell,
        note="the 66% dimension closed with a test set big enough to measure",
    ),
    B.Design(
        name="OWNER-13  ALL benchmarks, hold agent families ≈25% of mass, Δ vs cell",
        split=hold_families_by_mass(bench=None, share=0.25),
        target=B.DELTA_CELL,
        choice=by_cell,
        note="all three axes at once, at full scale",
    ),
    B.Design(
        name="OWNER-14  blocked by configuration AND base model (gemma), absolute",
        split=hold_configs_and_base(),
        choice=by_cell,
        note="strictest holdout the corpus supports; check what it costs in size",
    ),
    B.Design(
        name="OWNER-15  verifiable benchmarks, blocked by configuration, Δ vs cell",
        split=hold_configs(bench={"gsm8k", "humaneval"}, frac=0.3),
        target=B.DELTA_CELL,
        choice=by_cell,
        note="machine-checkable labels only (doc §5.2) — the honest-label version of OWNER-11",
    ),
    B.Design(
        name="OWNER-16  ALL benchmarks, blocked by configuration (50%), Δ vs cell",
        split=hold_configs(bench=None, frac=0.5),
        target=B.DELTA_CELL,
        choice=by_cell,
        note="how much does OWNER-11 depend on the 30/70 ratio",
    ),
]
