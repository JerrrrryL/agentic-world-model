# What these 143 agents actually shipped

One line per **train** run of `gsm8k-gemma-holdout-v1`: the post-training recipe the agent
actually shipped — which data, in what mixture, trained with which algorithm at which
hyper-parameters — read out of that run's own trajectory. The algorithms and the data
mixture are anchored to it quote by quote; the rest of the row is not, and the share is
measured below.

The split itself is `gsm8k-gemma-holdout-v1.yaml`; this file adds nothing to its membership and
changes nothing about it. It is a *reading* of the 143 train runs, and it is only as good as
the audit reported below — which is why the audit is here and not in a commit message.

`gsm8k-gemma-holdout-v1.recipes.jsonl` — one JSON object per line, `run` is the key, same order as the YAML's
`splits.train`.

## Where it comes from

| | |
|---|---|
| dataset | `aisa-group/PostTrainBench-Trajectories` (dataset) |
| revision | `39d3fcd794df51c062c8bd3b7f8523ba707aaeb3` |
| catalogue | `viewer_data/index.json`, sha256 `35d54c47ecdbb9cb38b07e80f3e91f0ea3712998ef9f120894ce88b7be90bb12` |
| benchmark | `gsm8k` |
| population | the 143 runs in `splits.train`, unchanged |

Both pins are the split's, copied here so this file can be checked on its own. The rule every
model in the chain was given: the only admissible evidence for a claim in a row is that run's
own event stream at that revision. That is the instruction, not a proven property of the
output — one row's `unresolved` ends "The harness records 0.818", and 0.818 is the
catalogue's accuracy for that row and appears nowhere in its digest. Treat the rule as what
was asked for, and the anchors below as what can be checked.

## One row

Catalogue facts, copied — `run`, `experiment`, `benchmark`, `trained_model`, `agent_model`,
`trace_format`, `seed`, `time_budget_h`, `time_taken`.

The recipe, extracted — `pipeline` (the normalised stage order, e.g. `sft→rft→grpo`),
`algorithms[]`, `datasets[]`, `hyperparams`, `total_train_examples`, `inference_tricks[]`,
`discarded[]` (what the agent tried and abandoned), `unresolved[]` (what the trajectory does
not settle), `confidence`.

The outcome, joined on afterwards — `accuracy`, `stderr`, `total_cost_usd`, `num_turns`,
`duration_ms`. **Afterwards** means the join, and nothing stronger. No extracted field was
conditioned on the catalogue's number, because the catalogue was not read until every row was
written. It does **not** mean the extractor worked blind: agents evaluate their own models
inside the run and say so, the filter deliberately keeps the tail of a result that follows a
training command, and so most digests do state a score. On a small number of rows one of those
stated scores rounds to the same value the catalogue later joined on. If you need a recipe that
could not have been written by a model that knew roughly how the run turned out, this file does
not give you one — and because the digests are not shipped, that is not something you can
re-check from the release.

The audit, per row — `extraction.{status, problems, evidence_anchors, repair_round, ...}`.

Every `algorithms[]` and `datasets[]` entry carries `evidence_i` (an event index in the full
stream) and `evidence_quote` (text from that event). That pair is what makes a row checkable
rather than merely plausible — and it is the *only* field pair that carries one.
`hyperparams`, `inference_tricks[]`, `discarded[]` and `unresolved[]` have no anchor, so
"anchored quote by quote" is true of the algorithms and the data mixture and of nothing else
in the row. The share is in the table below.

## How it was built

1. **Filter.** Each run's event stream is cut to the events that can carry a recipe —
   training scripts, launch commands, the agent's own statements about mixture and method,
   and the tail of any result that directly follows one. Four scaffolds name the same action
   four ways (`Bash` / `command_execution` / `shell` / `bash`), so the vocabularies are
   normalised first; a filter written against one of them keeps nothing for the other three.
   The budget is spent from the end backwards, because an agent's last hour is the run it
   submits. Median 561 source events → 54 kept, 37,617 characters; worst case 125,306. No run was reduced to nothing.
2. **Extract.** One model per run, reading only that digest, told that absent is null and
   that every claim needs a verbatim quote.
3. **Review.** Two adversarial lenses per recipe, each told to refute it. *Evidence fidelity*
   checks that every quote is really in the block it cites and that every non-null number is
   stated in the digest rather than defaulted. *Shipped, not tried* reads the end of the
   trajectory independently and asks whether the row describes the run that was submitted.
   A recipe is faithful only if both lenses agree.
4. **Repair, then re-review.** Anything faulted major or fatal was repaired against the digest
   and read again by both lenses.

**The stopping rule:** no row was repaired more than 2 times — 12 reached that bound — and no row was repaired again once the two-lens pair had read a text produced by that pair's own objection. Whatever it still faults there is left `flagged` and named below, not repaired a further time. Iterating until the reviewers stop objecting would be fitting the data to the reviewer, and the number that came out of it would mean nothing.

## What the audit measured

| status | runs | share | lenses | what it licenses |
|---|---|---|---|---|
| `reviewed-with-notes` | 62 | 43% | 3 | reviewed, only minor notes — read `extraction.problems` before quoting a number |
| `repaired-verified` | 66 | 46% | 2–4 | was faulted, was repaired, and the repaired text was re-reviewed clean of major/fatal |
| `flagged` | 15 | 10% | 2–4 | a reviewer found something major or fatal in this exact text and it was not fixed |

143/143 (100%) were read by both lenses; 143/143
(100%) carry a verdict against the exact text in the row rather than against
a version that was later repaired. 81 rows were repaired at least once (round 2: 12, round 3: 3, round 4: 9, round 5: 54, round 6: 3, counting the last repair each). 12 of them were repaired twice: a single verifier faulted the extraction, and the two-lens pair then faulted the repair. The rounds did not apply the same standard — the later ones put two adversarial lenses on records a single verifier had already passed, and that alone accounts for most of the repairs. Read a rising repair count as the review getting stricter, not as the extraction getting worse.

**`confidence` and `unresolved[]` are not comparable across rows.** The 81 repaired rows report `high` on 4% of themselves and carry a median 5 unresolved notes; the 62 never faulted report `high` on 74% and carry a median 2. That gap is the repair pass, not the runs: repairing a row means demoting whatever the digest does not settle, and a row no reviewer objected to was never asked to do that. Compare within `extraction.repair_round`, not across it, and treat the repaired rows' figures as the honest ones.

Problems recorded across all rows: 1472 minor, 21 major,
0 fatal. Minor notes are kept rather than cleared — most are "this quote is
short" or "this field is defensible but under-evidenced", and a reader checking a specific
number is better served by the note than by its absence.

### The evidence anchors, checked without a model

An agent *saying* it grepped is not a grep. Every `evidence_quote` in the file was re-checked
in code against the digest block it cites — whitespace collapsed and a short closed set of
look-alike characters folded to ASCII, no fuzzy matching, no edit distance:

| verdict | anchors | share | meaning |
|---|---|---|---|
| `ok` | 735 | 100.00% | verbatim inside the block it cites |
| `elided` | 0 | 0.00% | two real spans of the cited block joined by `...`, in order |
| `wrong-block` | 0 | 0.00% | the text is in the digest, but not at the cited event |
| `absent` | 0 | 0.00% | nowhere in the digest — a fabricated quote |
| `too-short` | 0 | 0.00% | under 8 characters, so it anchors nothing either way |
| `no-anchor` | 0 | 0.00% | the entry carries no quote or no event index |

735 anchors total. Reproduce with `awm.analysis.evidence.audit(row, digest_text)`;
`tests/test_evidence.py` pins the checker against paraphrase, out-of-order elision and
cross-block quotes, because a lenient checker would turn this table into a restatement of the
extractor's own confidence.

**Coverage: 735 of 2,191 (34%) list entries in the file carry an anchor at all.** The other 1,456 are `inference_tricks`, `discarded`, `unresolved` entries. `hyperparams` is a third case and the easiest to misread: it carries an `evidence_i` on almost every row but never an `evidence_quote`, so it names a block without quoting it, and there is nothing for the matcher to fail. Those fields are the extractor's prose, checked by a reader and not by a matcher, and a green anchor table says nothing whatever about them.

**This table is partly fitted, and the fit is the point.** A `wrong-block` or `absent` verdict
was a major fault, a major fault bought a repair, and the repair was told to fix the quote. So
a clean column here does not say the extractor got its quotes right first time; it says the
quotes that survived the loop are verbatim. What the loop could not do is invent an anchor
where the digest had none — that path ends in a deleted entry or a `flagged` row, both visible
above.

### By wire format

| trace_format | runs | major/fatal | median digest events | anchors ok |
|---|---|---|---|---|
| claude_code | 68 | 7 (10%) | 56 | 429/429 |
| codex | 43 | 5 (12%) | 99 | 195/195 |
| cursor | 6 | 1 (17%) | 59 | 46/46 |
| opencode | 26 | 2 (8%) | 34 | 65/65 |

## What is in the corpus

143 runs, 343 training stages,
392 dataset entries,
201 inference-time tricks,
701 abandoned attempts.
143 carry an accuracy, 0.042–0.912.

**Pipelines** (normalised stage order, by run)

| pipeline | runs |
|---|---|
| `sft` | 70 |
| `sft→sft` | 14 |
| `sft→sft→sft` | 10 |
| `sft→grpo` | 5 |
| `rft→sft` | 4 |
| `sft→grpo→grpo` | 3 |
| `sft→grpo→grpo→grpo` | 2 |
| `sft→grpo→grpo→grpo→grpo` | 2 |
| `sft→rft` | 2 |
| `sft→rft→grpo→grpo` | 2 |
| *+25 more, 29 runs* |  |

The one blank pipeline is not the finding it looks like: 1 where the digest holds no training launch at all (`opencode_opencode_glm-4.7-free_10h`). It still carries a score (0.105): the benchmark grades the submitted model, not the training. Read `confidence` and `unresolved[]` on those rows before counting them as extraction failures.

**Algorithm families** (a run counts once per family, not once per stage)

| family | runs |
|---|---|
| `sft` | 136 |
| `rft` | 33 |
| `grpo` | 21 |
| `merge` | 20 |
| `package` | 14 |
| `other` | 4 |
| `dpo` | 3 |
| `decode-config` | 2 |
| `distill` | 2 |

**Data sources** (by run)

| dataset_id | runs |
|---|---|
| `openai/gsm8k` | 125 |
| `meta-math/metamathqa` | 72 |
| `synthetic:self` | 35 |
| `nvidia/openmathinstruct-2` | 19 |
| `microsoft/orca-math-word-problems-200k` | 12 |
| `synthetic:teacher` | 6 |
| `unknown` | 6 |
| `ai-mo/numinamath-cot` | 1 |
| `clarkkitchen22/synthgsm8k-50k` | 1 |
| `local:broad_train.jsonl` | 1 |
| `local:omi2_gsm.jsonl` | 1 |
| `local:omi2_math.jsonl` | 1 |
| *+2 more, 2 runs* |  |

## Field coverage

| field | runs | share |
|---|---|---|
| algorithm | 142 | 99% |
| dataset | 139 | 97% |
| learning rate | 129 | 90% |
| epochs | 117 | 82% |
| batch size | 118 | 83% |
| total train examples | 62 | 43% |
| a per-dataset share | 103 | 72% |
| an inference-time trick | 94 | 66% |
| something discarded | 143 | 100% |

**A blank is NA, not zero.** `lr: null` means the trajectory did not state a learning rate,
not that the run trained without one. Every share above is a lower bound on what the run
actually did — it measures what the agent wrote down and a reviewer let stand.

## What this file can and cannot be used for

It can: describe what recipes this population of agents converged on; pair a recipe with the
score it got, for a predictor that reads recipes rather than agent names; find the runs that
tried a given method; supply `discarded[]` as negatives.

It cannot: stand in for the trajectories — every row is a lossy reading of one, and the digest
that produced it dropped most of the stream by design. It also cannot be read as ground truth
about the training that happened: it is what the agent said it did, checked against what the
agent's own log shows, which is not the same as what the GPUs did.

**15 rows are `flagged`** — a reviewer found something major or fatal in the exact
text shipped here and it is still there. 14 were repaired first and faulted again on the repaired text; 1 carries the objection unrepaired, because the repair round asked for a replacement text and never got one back. The second kind says nothing about the trajectory — it is a hole in the apparatus, and it is marked the same way because the row is equally unusable either way. They are in the file rather
than dropped, because dropping them would make the audit look cleaner than the data is. Filter on
`extraction.status != "flagged"` if a clean subset is what you need:

- `claude_claude-opus-4-6_10h_run2/gsm8k_Qwen_Qwen3-4B-Base_16855638` — **major** — algorithms[] records ONLY the stage-1 full-parameter SFT (train_full.py [197], train_data_v2.jsonl -> /home/ben/task/final_model, complete at [224]). But that is not the last training run the digest …
- `claude_non_api_claude-opus-4-8_10h_run2/gsm8k_Qwen_Qwen3-4B-Base_17310168` — **major** — These four values describe the SHIPPED v3, but the digest has no v3 training command at all. In the whole stream train_sft.py is invoked exactly once, the abandoned v1 at line 155 ( …
- `claude_non_api_claude-opus-5_10h_run1/gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17415831` — **major** — MAJOR. The stated derivation is contradicted by the digest. The recipe writes: "175 optimizer steps x 96 prompts/step = 16,800 unique prompts (grpo_v1 took rows 0-12,000 and grpo_v2 rows …
- `claude_non_api_claude-opus-5_10h_run1/gsm8k_Qwen_Qwen3-4B-Base_17415829` — **major** — The contamination result is attributed to the shipped corpus, but the digest attributes it to a DIFFERENT, discarded file. datasets[0].filtering reads 'prep_data.py [224] as invoked at [408] ... …
- `claude_non_api_max_claude-fable-5_1m__10h_run2/gsm8k_Qwen_Qwen3-1.7B-Base_17334179` — **major** — MAJOR — unanchored mechanism, contradicted by the only shipped code in the digest. The recipe states the orca filter as "kept only solutions whose final sentence OF THE LAST NON-EMPTY LINE contains …
- `codex_non_api_high_gpt-5.3-codex_10h_run2/gsm8k_Qwen_Qwen3-1.7B-Base_16917995` — **major** — 0.05 has exactly one source: the argparse default of train_gsm8k_masked.py in [277]. But the digest proves on its own that this file was changed before the shipped training run -- parse_args() in …
- `codex_non_api_high_gpt-5.3-codex_10h_run3/gsm8k_Qwen_Qwen3-4B-Base_16919979` — **major** — Neither value comes from v3's invocation command; both come from what the recipe itself admits is "the argparse default in [67], not overridden by v3". But the digest itself refutes that premise: …
- `codex_non_api_max_gpt-5.6-sol_10h_run1/gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17397511` — **major** — MAJOR — fabricated hyperparameter pinned to a block that does not contain it. The entry reads "probes at lr 5e-6 cosine [394] and lr 1e-6 constant [426]". Block [394] (digest line 186) is …
- `codex_non_api_max_gpt-5.6-sol_10h_run2/gsm8k_Qwen_Qwen3-1.7B-Base_17404247` — **major** — MAJOR. Listed as a SHIPPED stage-2 dataset with n_examples=7371, but no block in the digest shows any training run reading this file. `data/system_train.jsonl` occurs exactly once in the whole digest …
- `codex_non_api_xhigh_gpt-5.5_10h_run2/gsm8k_HuggingFaceTB_SmolLM3-3B-Base_17138216` — **major** — Cites event numbers that do not exist in the digest at all. The text says runs/full_exact_e2_stop_imend "appears only as an input ([110],[214],[222],[248])", but the digest's block indices run …
- `cursor_cli_cursor-grok-4.5-high_10h_run2/gsm8k_Qwen_Qwen3-1.7B-Base_17404232` — **major** — States that "the contamination filters in the digest belong to the other, unused files combined_math_train.jsonl [72], combined_math_clean.jsonl [195] and fewshot_math_train.jsonl [243]". …
- `glmx_glm-5.2-preview_1m__10h_run2/gsm8k_Qwen_Qwen3-1.7B-Base_17341965` — **major** — "The harness records 0.818." — 0.818 appears NOWHERE in the digest (grep for 0.818/818 returns nothing). The digest's only recorded scores are 0.828 and 0.810 in the [923] echo, and 82.8 / 80.97 / …
- `kimi_claude_k3-0715_1m__10h_run2/gsm8k_Qwen_Qwen3-4B-Base_17404545` — **major** — Reports 7473 gold rows, but the digest never prints the gold row count of sft_v1.jsonl. The 7473 at [37260] is only a coverage denominator ("Coverage: 7354/7473 questions"), and the 7,473 at [37433] …
- `opencode_opencode_gemini-3.1-pro_10h_run1/gsm8k_Qwen_Qwen3-1.7B-Base_16868240` — **major** — The main field reports train_v3.py, but no shell event in the digest ever launches it — not one of the 19 shell blocks is `python train_v3.py`. Conversely, train.py (v1) is the only pipeline launched …
- `opencode_zai_glm-5_10h_run2/gsm8k_HuggingFaceTB_SmolLM3-3B-Base_16853280` — **major** — FABRICATED DENOMINATOR. The text reads "The digest ends at [146] (turn 51 of 54)". The string "54" appears NOWHERE in the digest (`grep -n '54'` -> no match). The digest header carries only three …

## Regenerating

The recipe file and this document are both generated; neither is hand-edited. `render()` in
`awm/analysis/report.py` recomputes every figure above from the records themselves, so a
number here cannot drift from the file it describes.
