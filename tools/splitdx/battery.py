"""Does a candidate split leave any headroom, or does a lookup table already win?

One question, asked the same way of every design so the answers are comparable:
**before fitting anything, could the dumbest baseline already score what the
paper wants to report?** For ``gsm8k-gemma-holdout-v1`` the answer was yes — a
three-line per-agent lookup hits top-3 regret 0.0000 on the primary metric —
and that was invisible until someone ran the baseline.

A design is three choices, and this module makes all three explicit because
each of them can be the thing that saturates:

``split``     which runs are test. Leaks arrive here.
``target``    what number is predicted. ``accuracy`` is dominated by "how hard
              is this task" and "how strong is this base model"; a Δ against a
              cell reference removes that free win, which is what §3 of the
              idea doc asks for and what the shipped split does not do.
``choice``    what set the top-k pick is made *within*. Regret is meaningless
              without it, and a design that scores well only because its choice
              sets have two elements has not been measured.

Every number here is computed on train and applied to test, never fitted on
test. The baselines are deliberately stupid: if one of them saturates, the
split is the finding and no model needs writing.
"""

from __future__ import annotations

import collections
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

import numpy as np

Row = dict[str, Any]

#: Metadata a predictor could read off the catalogue without opening a
#: trajectory. The whole question is whether anything beats these.
META_FEATURES = ("agent_model", "benchmark", "trained_model", "trace_format", "experiment")


# --------------------------------------------------------------------------
# population


def scoreable(runs: Iterable[Row]) -> list[Row]:
    """The catalogue rows a split may draw from.

    Same three gates the shipped split's ``rule.require`` applies: a run must
    have been scored, and must not be flagged for contamination or for using a
    disallowed model. ``r.get(k) or {}`` rather than ``r.get(k, {})`` because
    the catalogue spells "not flagged" as a literal ``null`` on some rows and
    as a missing key on others.
    """
    out = []
    for r in runs:
        if r.get("accuracy") is None:
            continue
        if (r.get("contamination") or {}).get("flagged", False):
            continue
        if (r.get("disallowed_model") or {}).get("flagged", False):
            continue
        out.append(r)
    return out


def agent_family(agent_model: str) -> str:
    """Collapse ``claude-opus-4-6[1m]`` and ``claude-opus-4-6`` to one label.

    A design that holds out ``claude-opus-4-6[1m]`` while keeping
    ``claude-opus-4-6`` in train has held out a context-window flag, not an
    agent. 28 agent ids collapse to far fewer actual models, and the
    difference decides whether an agent-holdout is a real holdout.
    """
    a = agent_model.split("/")[-1]          # opencode/kimi-k2.5 -> kimi-k2.5
    a = a.split("[")[0].strip()             # claude-opus-4-6[1m] -> claude-opus-4-6
    return a


# --------------------------------------------------------------------------
# targets


@dataclass(frozen=True)
class Target:
    """What number the predictor is asked for, and what it is measured against.

    ``value`` maps a row to the label. ``rank_by`` maps a row to the quantity
    the *choice* is made on — usually the same thing, but for a Δ target the
    pick is still ultimately about which run ends up highest, so regret is
    always computed on raw accuracy. Keeping the two separate stops a Δ design
    from scoring itself on a scale nobody deploys.
    """

    name: str
    value: Callable[[Row, dict], float]
    #: Fitted on train only. ``None`` means the target needs no reference.
    fit_reference: Callable[[Sequence[Row]], dict] | None = None
    note: str = ""

    def labels(self, rows: Sequence[Row], ref: dict) -> np.ndarray:
        return np.array([self.value(r, ref) for r in rows], dtype=float)


def _cell(r: Row) -> tuple[str, str]:
    return (r["benchmark"], r["trained_model"])


def _fit_cell_median(train: Sequence[Row]) -> dict:
    by = collections.defaultdict(list)
    for r in train:
        by[_cell(r)].append(r["accuracy"])
    return {k: float(np.median(v)) for k, v in by.items()}


def _delta_vs_cell(r: Row, ref: dict) -> float:
    """Accuracy minus the train median for this (benchmark, base model) cell.

    ``nan`` when the cell is unseen in train. That is not a defect to paper
    over: a base-model holdout combined with a cell-referenced Δ target has no
    reference for any test row, and the right output is a design that reports
    itself undefined rather than one that silently falls back to a global mean.
    """
    return r["accuracy"] - ref[_cell(r)] if _cell(r) in ref else math.nan


ABSOLUTE = Target(
    name="accuracy",
    value=lambda r, ref: float(r["accuracy"]),
    note="what the shipped split predicts",
)

DELTA_CELL = Target(
    name="delta-vs-cell-median",
    value=_delta_vs_cell,
    fit_reference=_fit_cell_median,
    note="removes task hardness and base-model strength, per idea doc §3",
)


# --------------------------------------------------------------------------
# baselines


@dataclass(frozen=True)
class Baseline:
    """A predictor with no parameters worth the name.

    ``keys`` is the grouping it memorises on train; an unseen key falls back to
    the global train mean, which is the honest thing a lookup does and is also
    how the leak shows up — a design where every test key was seen in train is
    a design where the lookup never has to fall back.
    """

    name: str
    keys: tuple[str, ...]

    def predict(self, train: Sequence[Row], y_train: np.ndarray,
                test: Sequence[Row]) -> np.ndarray:
        if not self.keys:                       # the global mean
            return np.full(len(test), float(np.nanmean(y_train)))
        table: dict[tuple, list[float]] = collections.defaultdict(list)
        for r, y in zip(train, y_train):
            if not math.isnan(y):
                table[tuple(_key(r, k) for k in self.keys)].append(y)
        means = {k: float(np.mean(v)) for k, v in table.items()}
        fallback = float(np.nanmean(y_train))
        return np.array([means.get(tuple(_key(r, k) for k in self.keys), fallback)
                         for r in test])

    def coverage(self, train: Sequence[Row], test: Sequence[Row]) -> float:
        """Share of test rows whose key the lookup actually has an entry for."""
        if not self.keys:
            return 1.0
        seen = {tuple(_key(r, k) for k in self.keys) for r in train}
        hit = sum(tuple(_key(r, k) for k in self.keys) in seen for r in test)
        return hit / len(test) if test else math.nan


def _key(r: Row, k: str):
    return agent_family(r["agent_model"]) if k == "agent_family" else r[k]


BASELINES = (
    Baseline("global-mean", ()),
    Baseline("per-agent", ("agent_model",)),
    Baseline("per-agent-family", ("agent_family",)),
    Baseline("per-agent×benchmark", ("agent_model", "benchmark")),
    Baseline("per-agent×cell", ("agent_model", "benchmark", "trained_model")),
    Baseline("per-experiment", ("experiment",)),
    Baseline("per-format", ("trace_format",)),
)


# --------------------------------------------------------------------------
# metrics


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation, ties averaged. Written out rather than imported so the
    tie handling is visible — ties are the whole story on this corpus."""
    ok = ~(np.isnan(a) | np.isnan(b))
    a, b = a[ok], b[ok]
    if len(a) < 3:
        return math.nan
    ra, rb = _rankdata(a), _rankdata(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = math.sqrt(float((ra ** 2).sum() * (rb ** 2).sum()))
    return float((ra * rb).sum() / den) if den else math.nan


def _rankdata(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(len(x), dtype=float)
    # average ties
    i = 0
    xs = x[order]
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return ranks


def topk_regret(pred: np.ndarray, actual: np.ndarray, k: int, *,
                trials: int = 200, seed: int = 0) -> dict[str, float]:
    """How much worse than the oracle you do by executing your top ``k`` picks.

    Ties are broken at random, ``trials`` times, and the spread is reported.
    A single deterministic tie-break is a free point for whichever side the
    input happened to be sorted for — on this corpus 50 runs share 21 distinct
    predictions, so the tie-break is worth more than most model changes.
    """
    n = len(actual)
    if n == 0:
        return {"expected": math.nan, "best": math.nan, "worst": math.nan}
    k = min(k, n)
    oracle = float(np.max(actual))
    rng = np.random.default_rng(seed)
    got = np.empty(trials)
    for t in range(trials):
        jitter = rng.random(n)
        order = np.lexsort((jitter, -pred))       # -pred primary, jitter breaks ties
        got[t] = float(np.max(actual[order[:k]]))
    reg = oracle - got
    return {"expected": float(reg.mean()), "best": float(reg.min()), "worst": float(reg.max())}


def one_way_r2(rows: Sequence[Row], y: np.ndarray, feature: str) -> float:
    """Share of variance a per-level mean of one categorical feature explains.

    Uncorrected for the number of levels on purpose — the point is not an
    unbiased effect size, it is "could a lookup on this column alone do the
    job". ``experiment`` has 58 levels and will look strong for that reason;
    that is exactly the warning the number is there to give.
    """
    ok = ~np.isnan(y)
    rows = [r for r, m in zip(rows, ok) if m]
    y = y[ok]
    if len(y) < 3 or float(np.var(y)) == 0:
        return math.nan
    by = collections.defaultdict(list)
    for r, v in zip(rows, y):
        by[_key(r, feature)].append(v)
    fitted = np.concatenate([np.full(len(v), np.mean(v)) for v in by.values()])
    truth = np.concatenate([np.array(v) for v in by.values()])
    return float(1 - np.var(truth - fitted) / np.var(y))


#: ``claude_claude-opus-4-6_10h_run2`` and ``..._run3`` are the same configuration
#: launched again, and ``_old_container`` marks a re-run on different infrastructure.
#: Grouping replicates by raw ``experiment`` finds none — every (experiment,
#: benchmark, base model) triple on this corpus is unique — and a noise ceiling of
#: ``nan`` then reads as "no seed noise" when it means "asked the wrong question".
_REPLICATE_SUFFIX = __import__("re").compile(r"_run\d+(_old_container)?$|_old_container$")


def config_of(experiment: str) -> str:
    return _REPLICATE_SUFFIX.sub("", experiment)


def noise_ceiling(rows: Sequence[Row], y: np.ndarray) -> dict[str, float]:
    """How much of the spread is re-run noise, i.e. unexplainable by any model.

    Replicates are runs sharing (configuration, benchmark, base model) and
    differing only in which launch of that configuration they came from. Groups
    of one contribute nothing and are counted, so a ceiling computed from three
    replicates is not read as a measurement.
    """
    by = collections.defaultdict(list)
    for r, v in zip(rows, y):
        if not math.isnan(v):
            by[(config_of(r["experiment"]), r["benchmark"], r["trained_model"])].append(v)
    reps = [v for v in by.values() if len(v) > 1]
    within = [float(np.std(v, ddof=1)) for v in reps]
    total = float(np.nanstd(y))
    return {
        "groups_with_replicates": len(reps),
        "runs_in_those_groups": sum(len(v) for v in reps),
        "median_within_sd": float(np.median(within)) if within else math.nan,
        "total_sd": total,
        "explainable_variance_share": (
            1 - (float(np.median(within)) ** 2) / total ** 2 if within and total else math.nan
        ),
    }


# --------------------------------------------------------------------------
# a design, and its verdict


@dataclass
class Design:
    """One candidate split, complete enough to be measured end to end."""

    name: str
    #: rows -> (train, test). Must not look at accuracy.
    split: Callable[[Sequence[Row]], tuple[list[Row], list[Row]]]
    target: Target = ABSOLUTE
    #: How test rows are grouped into competing candidate sets for top-k.
    choice: Callable[[Row], tuple] = _cell
    ks: tuple[int, ...] = (1, 3, 5)
    note: str = ""
    extra_features: tuple[str, ...] = ()


def evaluate(design: Design, pop: Sequence[Row]) -> dict:
    """Run the whole battery on one design and return a flat, printable dict."""
    train, test = design.split(list(pop))
    ref = design.target.fit_reference(train) if design.target.fit_reference else {}
    y_tr = design.target.labels(train, ref)
    y_te = design.target.labels(test, ref)

    out: dict[str, Any] = {
        "design": design.name,
        "note": design.note,
        "target": design.target.name,
        "n_train": len(train),
        "n_test": len(test),
        "test_labels_defined": int((~np.isnan(y_te)).sum()),
    }
    if len(test) == 0 or not (~np.isnan(y_te)).any():
        out["verdict"] = "UNDEFINED — no test row has a label under this target"
        return out

    # --- leakage: what does test share with train?
    tr_exp = {r["experiment"] for r in train}
    out["leak_share_experiment"] = sum(r["experiment"] in tr_exp for r in test) / len(test)
    for f in ("agent_model", "trained_model", "benchmark", "trace_format"):
        seen = {r[f] for r in train}
        out[f"seen_{f}"] = sum(r[f] in seen for r in test) / len(test)
    fam = {agent_family(r["agent_model"]) for r in train}
    out["seen_agent_family"] = sum(agent_family(r["agent_model"]) in fam for r in test) / len(test)

    # --- what a single column already explains, on train
    out["r2_train"] = {f: round(one_way_r2(train, y_tr, f), 4)
                       for f in META_FEATURES + ("agent_family",) + design.extra_features}

    # --- the noise floor, on test
    out["noise"] = {k: (round(v, 4) if isinstance(v, float) else v)
                    for k, v in noise_ceiling(test, y_te).items()}

    # --- choice sets
    groups: dict[tuple, list[int]] = collections.defaultdict(list)
    for i, r in enumerate(test):
        groups[design.choice(r)].append(i)
    sizes = sorted(len(v) for v in groups.values())
    out["choice_sets"] = {"n": len(groups), "median_size": int(np.median(sizes)),
                          "min": sizes[0], "max": sizes[-1],
                          "singletons": sum(s == 1 for s in sizes)}

    # --- the baselines
    acc_te = np.array([r["accuracy"] for r in test], dtype=float)
    rows_out = []
    for b in BASELINES:
        p = b.predict(train, y_tr, test)
        row = {
            "baseline": b.name,
            "key_coverage": round(b.coverage(train, test), 3),
            "spearman": round(spearman(p, y_te), 4),
            "rmse": round(float(np.sqrt(np.nanmean((p - y_te) ** 2))), 4),
        }
        for k in design.ks:
            per_group = [topk_regret(p[idx], acc_te[idx], k) for idx in groups.values()
                         if len(idx) > 1]
            row[f"regret@{k}"] = round(float(np.mean([g["expected"] for g in per_group])), 4) \
                if per_group else math.nan
            row[f"regret@{k}_worst"] = round(float(np.mean([g["worst"] for g in per_group])), 4) \
                if per_group else math.nan
        rows_out.append(row)
    out["baselines"] = rows_out

    # --- the verdict: is there room for a model?
    best = min(rows_out, key=lambda r: (r.get("regret@3") if not math.isnan(
        r.get("regret@3", math.nan)) else math.inf))
    oracle_spread = float(np.mean([np.ptp(acc_te[idx]) for idx in groups.values()
                                   if len(idx) > 1])) if len(groups) else math.nan
    out["headroom"] = {
        "best_dumb_baseline": best["baseline"],
        "its_regret@3": best.get("regret@3"),
        "mean_within-choice-set accuracy spread": round(oracle_spread, 4),
        "regret@3 as share of that spread": round(best.get("regret@3", math.nan) / oracle_spread, 4)
        if oracle_spread else math.nan,
    }
    return out


def render(res: dict) -> str:
    """A compact block per design — the point is to put designs side by side."""
    L = [f"### {res['design']}   [target: {res['target']}]"]
    if res.get("note"):
        L.append(f"    {res['note']}")
    L.append(f"    n_train={res['n_train']}  n_test={res['n_test']}"
             f"  labelled_test={res.get('test_labels_defined')}")
    if "verdict" in res:
        L.append(f"    !! {res['verdict']}")
        return "\n".join(L)
    L.append("    leakage: " + "  ".join(
        f"{k.replace('seen_', '')}={res[k]:.0%}" for k in res if k.startswith("seen_")))
    L.append(f"    test rows sharing an experiment with train: {res['leak_share_experiment']:.0%}")
    L.append("    one-way R² on train: " + "  ".join(
        f"{k}={v}" for k, v in sorted(res["r2_train"].items(), key=lambda kv: -(kv[1] or 0))))
    cs = res["choice_sets"]
    L.append(f"    choice sets: {cs['n']} (median {cs['median_size']},"
             f" {cs['min']}–{cs['max']}, {cs['singletons']} singleton)")
    n = res["noise"]
    L.append(f"    noise floor: within-seed sd {n['median_within_sd']} vs total {n['total_sd']}"
             f"  → {n['explainable_variance_share']:.0%} explainable"
             f"  (from {n['groups_with_replicates']} replicate groups)")
    hdr = ["baseline", "cover", "spearman", "rmse"] + \
          [k for k in res["baselines"][0] if k.startswith("regret@") and not k.endswith("worst")]
    L.append("    " + "".join(h.ljust(22 if h == "baseline" else 12) for h in hdr))
    for b in res["baselines"]:
        cells = [b["baseline"].ljust(22), f"{b['key_coverage']:.0%}".ljust(12),
                 f"{b['spearman']}".ljust(12), f"{b['rmse']}".ljust(12)]
        cells += [f"{b[h]}".ljust(12) for h in hdr[4:]]
        L.append("    " + "".join(cells))
    h = res["headroom"]
    L.append(f"    → best dumb baseline: {h['best_dumb_baseline']}, regret@3 = {h['its_regret@3']}"
             f"  ({h['regret@3 as share of that spread']:.1%} of the"
             f" {h['mean_within-choice-set accuracy spread']} spread it could have lost)")
    return "\n".join(L)
