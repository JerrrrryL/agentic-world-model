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
* **Excluded columns** — `stderr`, `num_turns`, `total_cost_usd`, `duration_ms`,
  `time_taken`, `session_count`. All six are written after the run finishes and
  two of them beat the honest baseline; see the next section for the measurement.

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

## The clause every design above is missing: excluded columns

A split decides *which rows* a model may learn from. None of the 17 designs
above — including the recommended one — says anything about *which columns* it
may read, and the PostTrainBench catalogue row is written **after** the run
finishes. Six of its fields are post-execution facts, and one of them is the
label in disguise.

`stderr` is not correlated with the label; it *is* the label:

```
stderr = sqrt(accuracy * (1 - accuracy) / n)
```

with `n` the benchmark's item count. Solving `n` back out returns a constant to
machine precision — aime2025 **29**, gpqamain **447**, gsm8k **1318**,
humaneval **163** (relative spread ~1e-15; arenahardwriting and healthbench are
not fixed-`n` and spread 0.75 / 1.58). Below p = 0.5 the map is strictly
increasing, so on any benchmark whose accuracies all sit under 0.5 — aime2025
(max 0.3000) and gpqamain (max 0.3973) — ranking by `stderr` is an **exact**
oracle: Spearman +0.9995 and +0.9999, regret@3 **0.0000**.

That is a fact about the column. The number that matters is what it costs the
recommended design, priced the way an attacker actually gets it — choosing per
benchmark, **on train only**, whichever ranker looks best:

| ranker | regret@3 | headroom over the 0.0018 floor | headroom destroyed |
|---|---:|---:|---:|
| best parameter-free lookup (the honest bar) | 0.0335 | **+0.0317** | — |
| `stderr` alone, pooled | 0.0388 | +0.0370 | none — *it looks harmless* |
| `stderr`, chosen per benchmark on train | 0.0274 | +0.0256 | **19 %** |
| all six post-hoc columns, chosen per benchmark | 0.0269 | +0.0251 | **21 %** |

The second row is why this nearly went unnoticed. Pooled over seven benchmarks
`stderr` scores *worse* than the honest lookup, because the average mixes two
benchmarks it solves exactly with five it is bad at. A single pooled number
declares the column safe. Measuring per benchmark, and then letting the attacker
pick per benchmark the way any real one would, reverses the verdict.

Every column, alone, against the 0.0335 bar (`tools/splitdx/stderr_leak.out`):

| column | coverage | regret@3 | vs honest | |
|---|---:|---:|---:|---|
| `num_turns` | 100 % | **0.0269** | −0.0067 | **leaks on its own** |
| `stderr` | 90 % | 0.0388 | +0.0053 | leaks *per benchmark*, see above |
| `total_cost_usd` | 100 % | 0.0378 | +0.0043 | no help alone |
| `time_taken` | 100 % | 0.0462 | +0.0127 | no help alone |
| `duration_ms` | 100 % | 0.0475 | +0.0139 | no help alone |
| `session_count` | 100 % | 0.0691 | +0.0356 | no help alone |

`num_turns` beats the honest lookup outright, pooled, with no per-benchmark
selection at all — a longer session is a harder task badly handled.

**So any split that ships must carry an excluded-columns list**, and it must
exclude all six, not only the two that leak today. The four that "do not help
alone" are still post-execution facts: they cost nothing to exclude, they are
selected into the full attack in every fold above, and whether they help is a
property of this corpus rather than of the design.

**How bad is this?** Not fatal, and it should not be overstated. The design
survives — +0.0251 of headroom is still positive under the worst attack — but
the honest headline drops from +0.0317 to +0.0251, and on **2 of 7 benchmarks**
the benchmark is simply solved. The correct claim is "the excluded-columns
clause is load-bearing and worth a fifth of the headroom", not "the whole
benchmark is a lookup on `sqrt(p(1-p)/n)`".

### Why masking the agent is not a substitute for holding it out

The obvious cheaper fix for §1 is to leave the split alone and simply delete
`agent_model` from the input. Measured on the shipped split, it changes nothing:

| lookup | keys | regret@3 | |
|---|---|---:|---|
| per-agent | `agent_model` | 0.0000 | masked away |
| per-agent-family | `agent_family` | 0.0000 | masked away |
| **per-experiment** | `experiment` | **0.0000** | **survives masking** |
| per-format | `trace_format` | 0.0844 | survives masking |
| global-mean | — | 0.1543 | |

`experiment` recovers the agent **100 %** of the time — all 58 experiments are
single-agent, as is every one of the 1,175 `run_name`s — and 48 of the 50 test
rows share an experiment with a train row. So the saturating lookup is still
there after masking, at exactly regret@3 = 0.0000, keyed on a different column.

Masking is also the weaker guarantee in principle. Holding out a family makes
the exploit *structurally impossible*: there is no train row with that agent to
look up, so a model that identifies the agent perfectly from the trajectory
gains nothing. Masking only makes it *inconvenient*: the row is still in train,
and the trajectory carries the agent's house style whether or not the name field
is present — re-identification is exactly the sort of thing the model under test
is good at. An honour-system constraint on a benchmark whose whole purpose is to
be attacked is not a constraint.

**The two compose, and both are needed.** Hold out families (structural), *and*
list the columns nobody may read (procedural) — and that list has to include
`experiment` and `run_name` alongside the six post-execution fields, because
they are agent identity under another name.

## The confound no split can fix

Worth stating plainly, because it bounds everything above: **all 58 experiments
are single-agent.** "Which agent produced this run" and "which recipe this run
followed" are the same variable in 1,175 of 1,175 rows. The corpus is fully
nested, not crossed.

No choice of split and no masking separates a nested pair — they are one column
wearing two labels. Only new rollouts that deliberately cross the two can: the
**same recipe executed by different agents**, and **different recipes executed
by the same agent**. Until then, any result of the form "the trajectory predicts
the outcome" is indistinguishable from "the agent predicts the outcome", which
§1 already showed is worth 66 % of the variance.

There is one hint that the crossing is worth paying for. At the coarse level
`recipe_signal.py` reads the recipes at, some pipeline signatures *are* shared
across agents — and they do not behave like one recipe:

| pipeline signature | distinct agent families | accuracy spread |
|---|---:|---:|
| `sft` | 19 | **0.8036** |
| `sft>merge` | 9 | 0.6679 |
| `sft>package` | 7 | 0.6027 |
| `sft>sft` | 7 | 0.4261 |
| `sft>sft>sft` | 4 | 0.1941 |

Nineteen different agents ran plain `sft` and landed 0.80 of accuracy apart. So
the algorithm family carries almost nothing; whatever signal exists is in finer
detail — data mix, dataset choice, LR, epochs — or in execution quality.

Note what is and is not to blame here. The extraction *schema* already records
that detail: `datasets[].dataset_id`, `n_examples`, `share`, `filtering`, and a
`hyperparams` block with `lr`, `epochs`, `batch_size`, `grad_accum`,
`max_seq_len`, `scheduler`, `warmup`, `weight_decay`, `precision`. What throws
it away is `feat()` in `tools/splitdx/recipe_signal.py`, which reduces each
record to nine coarse summaries before the probe ever sees it. Reading the
recipes at a finer grain costs no new extraction — only a different `feat()`.
Coverage does: 143 of 1175 rows. That is the reason the extraction has to be
re-run wide before a split is committed, and a first reason to believe a crossed
rollout would resolve something.

## The blocker this exposed, which is bigger than the split

A better split is necessary and not sufficient. The split diagnostics show
metadata is not *sufficient* to predict the outcome; the thesis needs the
stronger claim that trajectory content is *available* — that a model reading the
recipe knows something the catalogue row does not.

Measured on the only rows where recipes exist today (`recipe_signal.out`, the 143
gsm8k train rows of the shipped split, 3 cells, 24 agent families):

**Pass 1** ranked the recipe top: `dataset_names` R² 0.7917 against the best
metadata feature's 0.7029.

**Pass 2** killed that. `dataset_names` has 107 levels over 143 rows — it is
nearly a row identifier, and a one-way R² on a near-unique categorical is high
whatever the labels are. Against a within-cell permutation null it scores
**+0.0409 excess, z = +0.9** — nothing. Ranking features of different cardinality
by raw R² measures cardinality. Corrected, the honest ordering is `agent_family`
+0.5392 (z +13.1), then `n_inference_tricks` +0.4555 (z +23.8) and `n_datasets`
+0.4409 (z +15.8).

**Pass 3** is the one that matters, because the recommended split removes agent
identity from train — so the question is not "does the agent predict the score"
but "does the recipe predict it *once the agent is known*". Residualising the
within-cell Δ on `agent_family` and re-testing against a within-family
permutation null:

| feature | levels | R² | null | excess | z |
|---|---:|---:|---:|---:|---:|
| `n_datasets` | 9 | 0.0773 | 0.0396 | +0.0377 | +1.8 |
| `n_stages` | 9 | 0.0521 | 0.0341 | +0.0180 | +1.1 |
| `dataset_kinds` | 9 | 0.0599 | 0.0430 | +0.0170 | +0.9 |
| `uses_rl` | 2 | 0.0048 | 0.0016 | +0.0032 | +1.4 |
| `pipeline_signature` | 49 | 0.2090 | 0.1959 | +0.0131 | +0.3 |
| `n_discarded` | 14 | 0.0978 | 0.0889 | +0.0088 | +0.3 |
| `n_inference_tricks` | 6 | 0.0172 | 0.0204 | −0.0032 | −0.2 |

**0 of 7 survive.** On this slice every recipe feature collapses to the agent's
house style: which agent wrote the recipe explains what the recipe contains
explains the score, and the middle term adds nothing of its own.

This is not yet a verdict on the thesis, for three reasons, and each is a
concrete next step rather than a hedge:

1. **Power.** Conditioning on 24 families inside 143 rows leaves a residual
   holding 30 % of the within-cell variance. `n_datasets` at z = +1.8 is
   suggestive and unresolvable at this n; reaching z = 3 needs roughly (3/1.8)²
   ≈ 2.8× the rows, i.e. ~400. The full corpus has **1,175**, so this *is*
   resolvable — but only after the extraction is run beyond gsm8k.
2. **Coverage.** One benchmark, and the train side of the split being replaced.
3. **Feature coarseness.** These are seven summary counts derived from the
   extraction, not the recipe text. A model reading the trajectory has more to
   work with — but "more to work with" has to be demonstrated, and right now the
   coarse version shows nothing.

**So the order of work is:** run the recipe extraction over all 1,175 runs, redo
pass 3 at full power, and only then commit a split — with the excluded-columns
clause in the spec file, not in this document. Committing the 5-fold design
first would be committing a well-built measuring instrument before knowing there
is anything to measure.

## The answer, at full power

The extraction now covers all **1,175 runs** (`tools/extract_recipes.py`,
15.4 minutes, 1,279 model calls, ~$244). Every record passes a mechanical anchor
check: each `evidence_i` is an event index that exists in that run's digest and
each `evidence_quote` is a literal span of that event. 1,171 passed on the first
extraction or its one repair; the remaining 4 passed after the checker was
hardened against a non-object list element.

Two things had to be established before the answer means anything.

### Is the cheap extraction the same measurement as the expensive one?

`tools/extract_agree.py` re-scores the 143 runs both pipelines saw. Neither side
is ground truth; disagreement bounds how much of an extraction is the extractor
rather than the run.

| | agreement |
|---|---:|
| pipeline signature, exactly | 62.9 % |
| same after collapsing repeats (`sft>sft` ≡ `sft`) | 80.4 % |
| same set of families, any order | 86.7 % |
| dataset-id Jaccard (mean / median / exact / disjoint) | 0.787 / 1.000 / 64.3 % / 2.1 % |
| `lr`, where both report one | 85.8 % |
| `epochs` / `batch_size` / `grad_accum` / `max_seq_len` | 92.6 / 93.0 / 91.4 / 91.0 % |
| `precision` | 70.0 % |

The cheap tier is not worse on coverage — it fills `n_examples` on 54 % of
datasets against gold's 45 %, and the same 76 % of hyperparameter slots — and it
reports "low confidence" far less often (6 vs 22 of 143), which is the one place
to distrust it.

The interesting result is the ordering. **The coarse skeleton is the least
reproducible part of the extraction and the fine detail is the most.** Two
extractions of the same trajectory disagree about the pipeline signature 37 % of
the time, and 17 of those 37 points are purely *how many re-runs of the same
family count as shipped stages rather than as discarded attempts* — a definition
question, not an observation. They agree about the learning rate 86 % of the
time. So the feature the earlier passes leaned on is the noisy one.

### What the recipe adds once the agent is known

`tools/splitdx/recipe_signal2.py`, 1,175 rows, 28 cells, 26 agent families,
21 features at the grain the recipe is written at, every number bucketed, each
scored as R² minus its own within-family permutation null (500 draws), bar set
at z > 3.5 for Bonferroni over 21 tests.

Before conditioning, the recipe looks strong: `peft` scores +0.1749 excess at
z = +93. After residualising on `agent_family` it scores **−0.0021**. Whether a
run used LoRA or a full finetune is a fact about which agent it was.

Three features survive:

| feature | z | excess R² | share of residual |
|---|---:|---:|---:|
| `uses_rl` | +7.3 | +0.0049 | 0.49 % |
| `n_datasets` | +4.4 | +0.0173 | 1.73 % |
| `lr` | +3.8 | +0.0116 | 1.16 % |

`top_dataset` has the largest excess of anything tested (+0.1183) and does not
clear the bar — 394 levels over 1,175 rows is a near row identifier, and pass 2
of the original probe exists precisely to stop that from counting.

Three hits out of 21 tests is the regime where a result is most likely to be the
search. Split-half, holding agent families intact so a family never straddles
the two halves, 12 halves of ~587 rows. Scored against the *attenuated*
expectation — halving the rows lowers the z a real effect can reach by about
√2, so re-imposing z > 3.5 on half the data would be demanding a bigger effect
than the one that was found:

| feature | full z | expected on a half | observed median | ratio | worst half |
|---|---:|---:|---:|---:|---:|
| `uses_rl` | +7.3 | +5.2 | +3.4 | 0.65 | +0.3 |
| `n_datasets` | +4.4 | +3.1 | +2.8 | 0.90 | −0.0 |
| `lr` | +3.8 | +2.7 | +1.7 | 0.62 | +0.1 |

Only `n_datasets` attenuates no faster than the halving explains, and all three
vanish on at least one half.

**The answer, in three sentences.** Something in the recipe is real — three
features clear a corrected bar against a permutation null after the agent is
conditioned out. It is small: 0.5–1.7 % of the residual variance each, against
the 66 % `agent_model` takes on its own. It is not sturdy, and 1,175 runs is
about the smallest corpus that could see it at all, with no more runs to add.

This closes the power question that §"The blocker" left open. It was not a power
problem. A choice-set predictor built on these features would be predicting the
agent, which is exactly what the 5-fold agent-family split exists to forbid — so
the split is still the right instrument, and there is still nothing here for it
to measure. What remains untested is the claim one step up: that a model reading
the *trajectory* beats these 21 summaries of it. That is the next measurement,
and it is the last one this corpus can support before the crossed rollout in
§"The confound no split can fix" becomes the only way forward.

## Reproducing

```bash
pip install scikit-learn                # ceiling.py only; the battery needs nothing extra
cd tools/splitdx
python3 run.py designs/owner.py designs/owner2.py   # per-design detail
python3 compare.py                                  # the ranking table
python3 kfold.py                                    # the recommended design
python3 recipe_signal.py                            # is there anything in the recipe? (143 rows)
python3 recipe_signal2.py                           # the same question at 1175 rows, finer features
python3 stderr_leak.py                              # which columns are the label in disguise
python3 mask_probe.py                               # would masking the agent do instead of holding it out?

cd ../..                                            # extraction, from the repo root
python3 tools/extract_recipes.py                    # all 1175, resumable, ~15 min, ~$244
python3 tools/extract_agree.py                      # cheap tier vs the 143 heavy records
```

`run.py` evaluates the shipped split first as a positive control and exits
non-zero unless it reproduces per-agent regret@3 = 0.0, Spearman = 0.7507 and
`agent_model` R² = 0.6632. If the control does not reproduce, nothing below it
means anything. Set `OMP_NUM_THREADS=4`; on a many-core box sklearn oversubscribes
badly on data this small.
