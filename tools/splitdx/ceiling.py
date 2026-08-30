"""The number that decides whether the thesis is testable on this corpus at all.

The parameter-free ladder in ``battery.py`` answers "does a lookup already win".
This answers the harder version: **how well can a real model do using nothing but
the catalogue metadata?** If gradient boosting on one-hot metadata already reaches
the seed-noise floor, then a model that reads trajectories has nothing left to
win, and no choice of split fixes that — the finding would be that post-training
outcomes on this corpus are predictable from configuration alone, which is the
"different paper" the idea doc §6 step 6 warns about.

Two things this file is careful about, because the first version got both wrong:

**The seeds have to be real.** ``HistGradientBoostingRegressor`` ignores
``random_state`` unless something in the fit is actually stochastic, so five fits
that differ only in that argument are one fit reported five times — and it
printed a standard deviation of exactly 0.0 across five seeds, which is what that
looks like from outside. Here the variation is a resampled 85% of train, so the
spread is a real answer to "would this number survive a different draw".

**The capacity has to be chosen without looking at test.** One arbitrary GBDT
config losing to a lookup shows that config lost, not that the metadata is
exhausted. So a small grid is selected by grouped cross-validation *inside train*
— grouped by configuration, the coarsest block, so the inner folds are never
easier than the outer split — and only the winner is scored on test.
"""

from __future__ import annotations

import sys

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import OrdinalEncoder

from pathlib import Path as _P
sys.path[:0] = [str(_P(__file__).resolve().parent), str(_P(__file__).resolve().parents[2])]

import battery as B  # noqa: E402

#: Everything a predictor can read without opening a trajectory. ``experiment`` is
#: excluded on purpose: under a blocked split it is never seen in test, and under
#: an unblocked one it is the leak itself, so including it measures the split
#: rather than the features.
META = ("agent_model", "agent_family", "benchmark", "trained_model", "trace_format")

#: Deliberately spans from "barely a model" to "will overfit 300 rows", so the
#: selected point says something about the data rather than about the default.
GRID = (
    {"max_iter": 60, "learning_rate": 0.05, "max_leaf_nodes": 7, "min_samples_leaf": 20},
    {"max_iter": 150, "learning_rate": 0.05, "max_leaf_nodes": 15, "min_samples_leaf": 10},
    {"max_iter": 300, "learning_rate": 0.06, "max_leaf_nodes": 31, "min_samples_leaf": 5},
    {"max_iter": 400, "learning_rate": 0.02, "max_leaf_nodes": 15, "min_samples_leaf": 10,
     "l2_regularization": 1.0},
    {"max_iter": 80, "learning_rate": 0.1, "max_leaf_nodes": 4, "min_samples_leaf": 30,
     "l2_regularization": 3.0},
)


def _X(rows, enc=None):
    raw = np.array([[B._key(r, f) for f in META] for r in rows], dtype=object)
    if enc is None:
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        enc.fit(raw)
    return enc.transform(raw), enc


def _fit(params, X, y, seed):
    return HistGradientBoostingRegressor(
        categorical_features=list(range(len(META))), random_state=seed, **params
    ).fit(X, y)


def _select(train, X, y, folds=4):
    """Pick a capacity by grouped CV inside train. Returns (params, cv_spearman).

    Scored on Spearman rather than RMSE because the outer metric is a ranking and
    because RMSE would reward a near-constant predictor — the exact failure the
    idea doc §6 warns about when most Δ are near zero.
    """
    groups = np.array([B.config_of(r["experiment"]) for r in train])
    n_groups = len(set(groups))
    if n_groups < 2 or len(y) < 40:
        return GRID[0], float("nan")
    cv = GroupKFold(n_splits=min(folds, n_groups))
    best, best_s = GRID[0], -np.inf
    for params in GRID:
        scores = []
        for tr, va in cv.split(X, y, groups):
            if len(set(y[tr])) < 2 or len(va) < 5:
                continue
            p = _fit(params, X[tr], y[tr], 0).predict(X[va])
            s = B.spearman(p, y[va])
            if not np.isnan(s):
                scores.append(s)
        if scores and float(np.mean(scores)) > best_s:
            best, best_s = params, float(np.mean(scores))
    return best, best_s


def metadata_ceiling(design, pop, *, seeds=(0, 1, 2, 3, 4), subsample=0.85):
    """Best metadata-only GBDT this corpus supports, scored on the design's test.

    ``subsample`` resamples train per seed so the reported spread is a real draw
    spread and not a decorated single fit.
    """
    train, test = design.split(list(pop))
    ref = design.target.fit_reference(train) if design.target.fit_reference else {}
    y_tr, y_te = design.target.labels(train, ref), design.target.labels(test, ref)
    ok = ~np.isnan(y_tr)
    train = [r for r, m in zip(train, ok) if m]
    y_tr = y_tr[ok]
    if len(train) < 40 or len(test) < 10:
        return None

    Xtr, enc = _X(train)
    Xte, _ = _X(test, enc)
    acc_te = np.array([r["accuracy"] for r in test], dtype=float)

    groups: dict = {}
    for i, r in enumerate(test):
        groups.setdefault(design.choice(r), []).append(i)
    idxs = [v for v in groups.values() if len(v) > 1]
    if not idxs:
        return None

    params, cv_s = _select(train, Xtr, y_tr)

    rng = np.random.default_rng(0)
    per = {"regret@1": [], "regret@3": [], "spearman": [], "rmse": []}
    n = len(train)
    for s in seeds:
        take = rng.choice(n, size=int(n * subsample), replace=False)
        p = _fit(params, Xtr[take], y_tr[take], s).predict(Xte)
        for k in (1, 3):
            per[f"regret@{k}"].append(float(np.mean(
                [B.topk_regret(p[i], acc_te[i], k)["expected"] for i in idxs])))
        per["spearman"].append(B.spearman(p, y_te))
        per["rmse"].append(float(np.sqrt(np.nanmean((p - y_te) ** 2))))

    noise = B.noise_ceiling(test, y_te)
    out = {k: round(float(np.mean(v)), 4) for k, v in per.items()}
    out["regret@3_sd"] = round(float(np.std(per["regret@3"])), 4)
    out["cv_spearman_in_train"] = round(cv_s, 4) if cv_s == cv_s else float("nan")
    out["chosen_capacity"] = f'{params["max_iter"]}it/{params["max_leaf_nodes"]}leaf'
    out["n_test"] = len(test)
    out["seed_noise_sd"] = round(noise["median_within_sd"], 4) \
        if noise["median_within_sd"] == noise["median_within_sd"] else float("nan")
    out["mean_choice_set_spread"] = round(
        float(np.mean([np.ptp(acc_te[i]) for i in idxs])), 4)
    return out


def regret_floor(design, pop, *, draws=400, seed=0):
    """The top-k regret a **perfect** predictor still pays, because labels are noisy.

    Comparing a baseline's regret to a per-run standard deviation is not a
    comparison — one is a max-order statistic over a choice set, the other is a
    spread around a single run, and they are only in the same units by accident.
    The quantity that actually bounds the headroom is this: if a model predicted
    every run's *expected* accuracy exactly, how far would its picks still fall
    short of the best *realised* score in the set?

    Simulated. Each run's observed accuracy is taken as its expected value, a
    draw adds re-run noise estimated from the replicate groups of this design's
    own test set, the oracle ranks by the noiseless value, and regret is measured
    against the noisy realisation. Anything a model could win lies between this
    floor and the dumb baseline; a design where the two meet has no room in it,
    however clean its holdout looks.
    """
    train, test = design.split(list(pop))
    if len(test) < 10:
        return None
    ref = design.target.fit_reference(train) if design.target.fit_reference else {}
    y_te = design.target.labels(test, ref)
    sd = B.noise_ceiling(test, y_te)["median_within_sd"]
    if sd != sd:                       # no replicate groups: cannot estimate it
        return None
    mu = np.array([r["accuracy"] for r in test], dtype=float)
    groups: dict = {}
    for i, r in enumerate(test):
        groups.setdefault(design.choice(r), []).append(i)
    idxs = [np.array(v) for v in groups.values() if len(v) > 1]
    if not idxs:
        return None
    rng = np.random.default_rng(seed)
    out = {}
    for k in (1, 3):
        per_set = []
        for ix in idxs:
            m = mu[ix]
            order = np.argsort(-m, kind="mergesort")[:k]
            noisy = m[None, :] + rng.normal(0, sd, size=(draws, len(ix)))
            per_set.append(float(np.mean(noisy.max(axis=1) - noisy[:, order].max(axis=1))))
        out[f"floor@{k}"] = round(float(np.mean(per_set)), 4)
    out["noise_sd_used"] = round(sd, 4)
    return out
