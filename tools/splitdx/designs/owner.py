"""My own read, computed independently of the proposing agents.

Six designs along the three axes that can each fix the saturation on their own —
what is held out, what is predicted, and what the pick is made within — so the
comparison shows which axis is actually load-bearing rather than which bundle
happened to win. Deliberately includes two designs I expect to fail, because a
set of designs that all work is a set that was not measured.
"""

from __future__ import annotations

import collections
import sys

from pathlib import Path as _P
sys.path[:0] = [str(_P(__file__).resolve().parents[1]), str(_P(__file__).resolve().parents[3])]

import battery as B  # noqa: E402

GSM8K = "gsm8k"
VERIFIABLE = ("gsm8k", "humaneval")          # exact-match / unit tests, per idea doc §5.2


def _fams(rows):
    """Agent families ordered by run count — the holdout picks from the tail so
    train keeps the mass, which is the realistic deployment direction: predict a
    new agent from many old ones, not the reverse."""
    c = collections.Counter(B.agent_family(r["agent_model"]) for r in rows)
    return [f for f, _ in c.most_common()]


def hold_agent_families(bench=None, n_test_families=6):
    """Hold out whole agent families — the 66% dimension the shipped split leaves visible."""
    def split(rows):
        rows = [r for r in rows if bench is None or r["benchmark"] in bench]
        fams = _fams(rows)
        held = set(fams[-n_test_families:])
        return ([r for r in rows if B.agent_family(r["agent_model"]) not in held],
                [r for r in rows if B.agent_family(r["agent_model"]) in held])
    return split


def hold_big_agent_families(bench=None, n=3):
    """Hold out the *largest* families instead. If the tail-holdout looks healthy
    only because the tail is small and weird, this is where that shows."""
    def split(rows):
        rows = [r for r in rows if bench is None or r["benchmark"] in bench]
        held = set(_fams(rows)[:n])
        return ([r for r in rows if B.agent_family(r["agent_model"]) not in held],
                [r for r in rows if B.agent_family(r["agent_model"]) in held])
    return split


def hold_configurations(bench=None, frac=0.3):
    """Blocked by configuration: every launch of a held-out config goes to test.

    Closes the 96% experiment leak without touching the agent axis, so the
    contrast with the family holdout isolates how much of the leak was
    *configuration* memorisation versus *agent* memorisation.
    """
    def split(rows):
        rows = [r for r in rows if bench is None or r["benchmark"] in bench]
        cfgs = sorted({B.config_of(r["experiment"]) for r in rows})
        held = set(cfgs[::int(1 / frac)])          # deterministic stride, no rng
        return ([r for r in rows if B.config_of(r["experiment"]) not in held],
                [r for r in rows if B.config_of(r["experiment"]) in held])
    return split


def hold_benchmark(held):
    def split(rows):
        return ([r for r in rows if r["benchmark"] != held],
                [r for r in rows if r["benchmark"] == held])
    return split


def hold_base_model_all_benchmarks(held="google_gemma-3-4b-pt"):
    def split(rows):
        return ([r for r in rows if r["trained_model"] != held],
                [r for r in rows if r["trained_model"] == held])
    return split


def _shipped():
    from awm import splits
    s = splits.load("posttrainbench/gsm8k-gemma-holdout-v1")
    tr, te = set(s.train), set(s.test)
    def split(rows):
        p = lambda r: f'{r["experiment"]}/{r["run_name"]}'
        return ([r for r in rows if p(r) in tr], [r for r in rows if p(r) in te])
    return split


#: The pick is made among the runs of one configuration across base models —
#: "given this recipe, which base model does it help most" — rather than among
#: agents. Included to show that the choice set alone can change the verdict.
def by_config_benchmark(r):
    return (B.config_of(r["experiment"]), r["benchmark"])


DESIGNS = [
    B.Design(
        name="OWNER-1  gsm8k, hold out 6 smallest agent families",
        split=hold_agent_families(bench={GSM8K}, n_test_families=6),
        note="closes the 66% dimension; keeps the shipped split's benchmark and target",
    ),
    B.Design(
        name="OWNER-2  gsm8k, hold out 3 LARGEST agent families",
        split=hold_big_agent_families(bench={GSM8K}, n=3),
        note="the same axis in the direction that costs train mass — a robustness check on OWNER-1",
    ),
    B.Design(
        name="OWNER-3  gsm8k, blocked by configuration (30%)",
        split=hold_configurations(bench={GSM8K}, frac=0.3),
        note="closes the 96% experiment leak only; agent axis left open on purpose",
    ),
    B.Design(
        name="OWNER-4  SHIPPED split, but target = Δ vs cell median",
        split=_shipped(),
        target=B.DELTA_CELL,
        note="expected UNDEFINED: holding out the base model removes every cell reference",
    ),
    B.Design(
        name="OWNER-5  all 7 benchmarks, hold out 6 smallest agent families, Δ vs cell",
        split=hold_agent_families(bench=None, n_test_families=6),
        target=B.DELTA_CELL,
        note="all three axes moved at once — the doc's target, the leaky dimension closed, full corpus",
    ),
    B.Design(
        name="OWNER-6  verifiable benchmarks only, hold agent families, Δ vs cell",
        split=hold_agent_families(bench=set(VERIFIABLE), n_test_families=6),
        target=B.DELTA_CELL,
        note="idea doc §5.2: gsm8k and humaneval are the two with machine-checkable labels",
    ),
    B.Design(
        name="OWNER-7  all benchmarks, hold out gemma (the shipped rule, widened)",
        split=hold_base_model_all_benchmarks(),
        note="what the shipped rule buys if it simply stops being gsm8k-only",
    ),
    B.Design(
        name="OWNER-8  hold out humaneval entirely (cross-benchmark transfer)",
        split=hold_benchmark("humaneval"),
        note="doc §7: does the predictor transfer, or are we training two models",
    ),
    B.Design(
        name="OWNER-9  gsm8k, hold agent families, pick within configuration",
        split=hold_agent_families(bench={GSM8K}, n_test_families=6),
        choice=by_config_benchmark,
        note="same split as OWNER-1, choice set changed — isolates the choice axis",
    ),
]
