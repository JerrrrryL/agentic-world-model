# Redesigning `gsm8k-gemma-holdout-v1`

Status: **proposal**, 2026-08-30. Nothing here is committed to `splits/` yet.
Every number below is reproducible with `tools/splitdx/` (see the last section).

## Why the shipped split has to change

`splits/posttrainbench/gsm8k-gemma-holdout-v1` holds out the *base model*: 50 test
runs, all gsm8k, all `google_gemma-3-4b-pt`. Measured on the catalogue:

| what test shares with train | share |
|---|---|
| `agent_model` seen in train | **98 %** |
| `agent_family` seen in train | **98 %** |
| same `experiment` as a train row | **96 %** |
| `trained_model` seen in train | 0 % |

The one dimension it closes explains **7.5 %** of accuracy variance. The dimension
it leaves 98 % open explains **66.3 %**. So a per-agent lookup table — no
trajectory, no learning, one `dict` — scores Spearman **0.7507** and

> **top-3 regret 0.0000.**

Top-k regret is the primary metric in `doc/iclr-27-idea.md` §6. A learned
predictor cannot beat zero; the best it can do is tie. The split is saturated.

Two smaller problems come with it:

* **The target contradicts §3.** The idea doc says predict *Δ, not the absolute
  score*, because "absolute score is dominated by how hard is this task and how
  strong is this base model". The shipped split predicts absolute accuracy. On
  the full corpus `benchmark` alone explains 47.9 % of absolute-accuracy variance.
* **The choice set is one set of 50.** Picking 3 of 50 candidates on one
  (benchmark, base model) cell is not a decision anyone makes, and a single
  choice set means the metric has no variance to average over.

## Three axes, not one

The saturation is usually described as "the holdout is wrong". It isn't only the
holdout. Three independent choices each set the answer, and each can saturate a
design on its own:

1. **Split** — what is held out.
2. **Target** — what number is predicted (absolute accuracy, or Δ against a
   reference).
3. **Choice set** — the group of candidates the top-k pick is made within.

Sixteen designs were measured across all three (`tools/splitdx/designs/`). Four
structural facts came out, and they constrain what is even worth proposing.

### 1. A base-model holdout and a cell-referenced Δ are mutually exclusive

If Δ is measured against the train median of the row's (benchmark, base model)
cell, and the split holds out a whole base model, then no test row's cell exists
in train. **0 of 50 test rows get a label.** The combination is not "bad", it is
undefined. (Design OWNER-4.)

### 2. When the choice set *is* the cell, a cell-referenced Δ cannot move top-k regret

Δ against the cell median subtracts the same constant from every member of a
cell. A within-set monotone shift leaves the within-set *ranking* untouched, so
every top-k regret is byte-identical to the absolute-target version. Verified on
two pairs that differ only in target — OWNER-10/11 and OWNER-12/13 — identical at
every k for every baseline, and again per-fold in the 5-fold design below
(0.0231 / 0.0469 / 0.0444 / 0.0259 / 0.0272 under both targets).

This does **not** make the Δ target pointless. It changes Spearman, RMSE and
calibration, and that is where it earns its place — see the recommendation.

### 3. A small choice set manufactures a passing score

regret@3 inside a set of 3 is zero by construction. Four designs "passed" this
way before the choice-set sizes were printed:

| design | median choice set | regret@3 |
|---|---|---|
| OWNER-1 | 2 | 0.0000 |
| OWNER-5 | 1 (13 singletons of 24) | 0.0000 |
| OWNER-6 | 2 | 0.0000 |
| OWNER-9 | 1 (5 singletons of 6) | 0.0000 |

Any proposal has to report the choice-set size distribution next to the regret,
or the regret means nothing. One of the five independently-proposed designs
("hack-aware arenas", median 4 candidates) scores regret@3 = 0.0023 and fails for
exactly this reason.

### 4. A tuned metadata model loses to a parameter-free lookup

A `HistGradientBoostingRegressor` over the five metadata columns, with its
capacity chosen by configuration-grouped CV *inside train* (never test) and
averaged over five 85 %-resampled fits, **loses to the best parameter-free lookup
on 14 of 16 designs**, and on 5 of 5 folds of the recommended design. Metadata is
not the bottleneck; there is real room above it for something that reads content.

## The measurement that ranks the designs

Comparing a baseline's regret to a per-run standard deviation is not a
comparison — one is a max-order statistic over a choice set, the other is a
spread around a single run. The quantity that actually bounds the headroom:

* **dumb@3** — the best parameter-free lookup's top-3 regret. A model has to beat
  this, so it is the top of the useful range.
* **floor@3** — the top-3 regret a **perfect** predictor still pays, because the
  labels carry re-run noise. Simulated from the replicate groups of the design's
  own test set (`_run2` / `_run3` / `_old_container` are replicate suffixes; 58
  experiments collapse to 33 configurations). Nothing can go below it.
* **winnable** = dumb@3 − floor@3, reported as a share of the mean accuracy
  spread inside a choice set — because an absolute regret gap means different
  things on a benchmark where runs differ by 40 points and one where they differ
  by 4.

A design whose winnable share is zero or negative is saturated, whatever its
holdout looks like on paper.

## The comparison

Full output in `tools/splitdx/compare.out`. `lk_*` are leakage shares.

| win% | winnable | dumb@3 | floor@3 | gbdt@3 | n_test | sets | med | lk_ag | lk_exp | design |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| **13.8 %** | +0.0598 | 0.0619 | 0.0021 | 0.0865 | 297 | 28 | 11 | **0 %** | **0 %** | OWNER-12 — agent families ≈25 % of mass, absolute |
| **13.8 %** | +0.0598 | 0.0619 | 0.0021 | 0.1457 | 297 | 28 | 11 | **0 %** | **0 %** | OWNER-13 — same split, Δ vs cell |
| 9.9 % | +0.0545 | 0.0593 | 0.0048 | 0.0894 | 614 | 28 | 23 | 35 % | 0 % | OWNER-16 — blocked by configuration (50 %), Δ |
| 8.9 % | +0.0397 | 0.0447 | 0.0050 | 0.0749 | 103 | 7 | 14 | 50 % | 0 % | OWNER-14 — configuration **and** base model |
| 8.8 % | +0.0511 | 0.0669 | 0.0158 | 0.0766 | 63 | 4 | 15 | 0 % | 0 % | OWNER-2 — gsm8k, 3 largest agent families out |
| 7.8 % | +0.0470 | 0.0478 | 0.0008 | 0.0642 | 59 | 4 | 14 | 32 % | 0 % | OWNER-3 — gsm8k, blocked by configuration |
| 6.1 % | +0.0304 | 0.0386 | 0.0082 | 0.0179 | 305 | 7 | 44 | 100 % | 97 % | OWNER-7 — the shipped rule, widened to 7 benchmarks |
| 6.0 % | +0.0373 | 0.0405 | 0.0032 | 0.0774 | 113 | 8 | 14 | 35 % | 0 % | OWNER-15 — verifiable benchmarks only |
| 5.2 % | +0.0280 | 0.0298 | 0.0018 | 0.0599 | 392 | 28 | 14 | 64 % | 0 % | OWNER-10/11 — blocked by configuration (30 %) |
| **−1.3 %** | −0.0083 | 0.0061 | 0.0144 | 0.0083 | 172 | 4 | 43 | 100 % | 100 % | OWNER-8 — hold out a whole benchmark |
| **−3.3 %** | −0.0245 | 0.0000 | 0.0245 | 0.0634 | 50 | **1** | 50 | 98 % | 96 % | **CONTROL — the shipped split** |
| — | — | — | — | — | 50 | — | — | — | — | OWNER-4 — **UNDEFINED**, 0/50 labelled |

Reading the two negative rows: the dumb baseline already scores *below* the noise
floor. That can only happen when the choice-set structure hands it the answer —
which is the sharpest available statement that a split is saturated.

Widening the shipped rule to all seven benchmarks (OWNER-7) does help — 6.1 % —
but it leaves `agent_model` and `experiment` leakage at 100 % / 97 %. It buys
headroom by adding rows, not by closing the leak.

## The knob nobody was reporting: which families you hold out

Two of the five independently-proposed designs are the same shape as OWNER-12/13
and report dumb@3 of **0.0338** and **0.0453** where this one measures **0.0619**.
Same rule, different rosters, and the number moves by a factor of two.

So: fix the rule, vary the roster over every draw the rule allows, and look at
the spread (`tools/splitdx/roster.out`). Thirteen rosters, each ~25 % of runs:

* dumb regret@3: median 0.0380, range **0.0104 – 0.0711**
* winnable share: median 7.1 %, range **1.7 % – 17.3 %**
* **every roster positive**

The *sign* is robust; the *magnitude* is a factor of ten. Quoting 13.8 % as the
headline would be tuning on the metric the split is supposed to report.

## Recommendation: a 5-fold agent-family partition, Δ vs cell

Remove the roster knob rather than choosing a value for it. Partition all 26
agent families into 5 folds (greedy largest-first into the lightest fold — no
seed, no search). Every family is in test exactly once; **every one of the 1,175
runs is scored exactly once out-of-fold**; the fold spread becomes the error bar
instead of a degree of freedom.

* **Split** — hold out whole agent *families*, so `claude-opus-4-6[1m]` never
  trains while `claude-opus-4-6` tests. Leakage: `agent_model` 0 %,
  `agent_family` 0 %, `experiment` 0 %.
* **Target** — Δ against the train median of the (benchmark, base model) cell.
  Every cell is in train by construction, so all labels are defined.
* **Choice set** — the (benchmark, base model) cell. 28 sets, median 8–9, no
  singletons, median 6 distinct configurations per set — the candidates are
  genuinely different recipes, not replicates of one.

`tools/splitdx/kfold.out`:

| fold | n_test | dumb@3 | floor@3 | winnable | win% |
|---|---:|---:|---:|---:|---:|
| 0 | 235 | 0.0231 | 0.0004 | +0.0227 | 5.0 % |
| 1 | 234 | 0.0469 | 0.0012 | +0.0457 | 12.1 % |
| 2 | 237 | 0.0444 | 0.0019 | +0.0425 | 11.2 % |
| 3 | 234 | 0.0259 | 0.0028 | +0.0231 | 6.3 % |
| 4 | 235 | 0.0272 | 0.0028 | +0.0244 | 7.6 % |
| **pooled** | **1175** | **0.0335** | **0.0018** | **+0.0317** | **8.5 %** |

Fold spread 5.0 – 12.1 %, sd 2.8 points, **all five positive**. Identical under
both targets, as fact 2 requires.

**Why Δ, given that it cannot move top-k regret.** It moves everything else, and
it moves it in the direction §3 asks for:

| | absolute | Δ vs cell |
|---|---:|---:|
| one-way R² of `benchmark` on train | 0.4788 | 0.1308 |
| one-way R² of `trained_model` on train | 0.0136 | 0.0065 |
| metadata GBDT Spearman, pooled over folds | **+0.691** | **+0.129** |
| metadata GBDT beats the dumb lookup | 0 / 5 folds | 0 / 5 folds |

Under the absolute target a metadata-only model reaches Spearman +0.69 by
learning which benchmark is easy. Under Δ that shortcut is gone and the same
model drops to +0.13 — while 97 % of the Δ variance is still explainable from the
replicate groups, so the signal has not been removed, only the shortcut. That is
the configuration a trajectory-reading model needs in order to be attributable.

**Cost:** free. No new runs; a re-partition of the 1,175 already in the
catalogue.

**Open, not yet measured:** whether recipe-level features actually vary within a
choice set. `splits/posttrainbench/gsm8k-gemma-holdout-v1.recipes.jsonl` covers
only the 143 gsm8k train rows, so the recipe extraction has to be run over the
full corpus before "the trajectory has something the metadata does not" is a
measured claim rather than a plausible one.

## Reproducing

```bash
pip install scikit-learn                # ceiling.py only; the battery needs nothing extra
cd tools/splitdx
python3 run.py designs/owner.py designs/owner2.py   # per-design detail
python3 compare.py                                  # the ranking table
python3 kfold.py                                    # the recommended design
```

`run.py` evaluates the shipped split first as a positive control and exits
non-zero unless it reproduces per-agent regret@3 = 0.0, Spearman = 0.7507 and
`agent_model` R² = 0.6632. If the control does not reproduce, nothing below it
means anything. Set `OMP_NUM_THREADS=4`; on a many-core box sklearn oversubscribes
badly on data this small.
