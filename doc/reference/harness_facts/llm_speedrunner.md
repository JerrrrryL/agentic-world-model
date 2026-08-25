<!-- 调研原文,2026-08-24 由调研 agent 实际 clone 仓库/读取 HF 数据集后产出;VERIFIED = 在文件或页面里看到,UNVERIFIED = 推断。scratchpad 路径为当时的临时 clone 位置。 -->
<!-- 主题:LLM-Speedrunner (Meta, 已被 PI 设定取代) -->

The OpenReview API returned nothing, so the venue stays unverified. Everything else is nailed down from the cloned files and the paper/GitHub pages. Final report follows.

# LLM-Speedrunner (Automated LLM Speedrunning Benchmark) — structural report

Local clone (LFS skipped): `/tmp/claude-30001/-home-gangda-workspace-DeepCommit-ai-hierarchy-verifier/97160149-2e2d-43ef-a087-7b3437df11a9/scratchpad/speedrunner/llm-speedrunner` — referred to as `$REPO` below.

Tag legend: **[V]** = VERIFIED (seen in a file or page), **[U]** = UNVERIFIED / inferred.

---

## 1. Repo facts

| Item | Value |
|---|---|
| URL | https://github.com/facebookresearch/llm-speedrunner [V] |
| License | CC BY-NC 4.0 (`$REPO/LICENSE` first line "Attribution-NonCommercial 4.0 International"; GitHub API reports `license: Other/NOASSERTION`) [V] — **non-commercial**, relevant if our unified repo redistributes the hints/scripts |
| Stars / forks | 146 / 15 (GitHub API, 2026-08-24) [V] |
| Created | 2025-06-05 [V] |
| Commits | **5 total**, all by 2 authors: `4fe4d52f` 2025-06-30 "Initial commit" (minqi), `5582db38` 2025-06-30 typo, `bfde801f` 2025-07-30 "Update transformers version to 4.51.0", `c9dc3ccb` 2025-10-09 "add data file", `ec3e4afd` 2025-10-09 "adds back the conda env tar" (Bingchen Zhao) [V] |
| Last commit | `ec3e4afd` 2025-10-09 [V] |
| Commit frequency, last 3 months | **0** commits since 2026-05-24 (API `commits?since=`) [V]. `pushed_at: 2026-05-06` is Dependabot PR branches only. Effectively unmaintained. |
| Open issues/PRs | 1 open issue (#1 "Release hint data on Hugging Face", Jul 2025, unanswered) + 9 open Dependabot PRs; closed: #2 "Confusion over AIDE or Aider", #3 "cannot find speedrunner-19-21.tar.gz" (fixed by the Oct 2025 LFS commit) [V] |
| Clone size | 11 MB with `GIT_LFS_SKIP_SMUDGE=1` (298 tree entries, 7.6 MB of blobs). **One LFS object**: `conda_envs/speedrunner-19-21.tar.gz` = **5,026,575,451 bytes (5.03 GB)** (pointer file: `oid sha256:de92761c…`, `size 5026575451`). A plain `git clone` pulls it — my first clone hit 1.3 GB after 3 min and timed out [V] |
| Language | GitHub label "Jupyter Notebook" (by bytes — `analysis/main_plots.ipynb` is 617 KB). Actual code: 62 `.py` files, 17,671 LOC (includes 21 record training scripts), 40 YAML, 86 `.txt` hints, no Dockerfile [V] |
| Paper | arXiv **2506.22419** "The Automated LLM Speedrunning Benchmark: Reproducing NanoGPT Improvements", v1 2025-06-27, v2 2025-06-30; 23 authors, Meta + U. Edinburgh (Bingchen Zhao, Despoina Magka, Minqi Jiang, … Jakob Foerster, Yoram Bachrach) [V]. OpenReview id `w98hMEjzu8` exists; venue/decision [U] (page bot-blocked) |

Recommended clone: `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 https://github.com/facebookresearch/llm-speedrunner.git` (0.9 s), then `git lfs pull` only if you need the records 19–21 env.

---

## 2. Task definition format

### 2.1 Directory layout [V]

```
workspace_templates/nanogpt_speedrun/record_{1..21}/train_gpt2.py   # human record R's script (18–30 KB each)
workspace_templates/nanogpt_speedrun/record_{1..21}/results.json    # re-timed baseline metrics for R (see 4.3)
workspace_templates/nanogpt_speedrun/data.jsonl                     # 57-row flattened dataset (see 2.4)
config/task/nanogpt_speedrun/default_config.yaml                    # shared task config (prompt, metrics, slurm)
config/task/nanogpt_speedrun/speedrun_record_{1..20}.yaml           # 3-line per-record configs (NO record_21 task)
data/nanogpt_speedrun_knowledge_in_levels/record_{1..20}/level_{0,1,2,3}_*.txt   # hints for R -> R+1
data/nanogpt_speedrun_knowledge_in_levels/{record_2,record_3}/level_9_muon.txt     # extra: Muon blog post (25.8 KB)
data/nanogpt_speedrun_knowledge_in_levels/{record_11,record_12}/level_9_flexattn.txt # extra: FlexAttention docs (34.5 KB)
data/nanogpt_speedrun_knowledge_in_levels/{muon.md,flex_attn.md,llm_logs.jsonl}
```

A task = hydra config `task=nanogpt_speedrun/speedrun_record_R`, which is just:
```yaml
# @package _global_
defaults: [nanogpt_speedrun/default_config]
template_dirname: nanogpt_speedrun/record_R
slurm_config_args: {job_name: nanogpt_speedrun_record_R}
```
Semantics: agent starts from record R's `train_gpt2.py`; hint dir `record_R` describes the R→R+1 change; target is record R+1's time. Confirmed by `data.jsonl` (`id: 1` has `target_time: 2209926` = record_2's `train_time`) [V].

### 2.2 Task prompt / constraints (`config/task/nanogpt_speedrun/default_config.yaml`) [V]

- `task_description`: "Improve train_gpt2.py so that it achieves or goes below the target val_loss value of 3.28 in the shortest train_time possible."
- `code_instructions` (constraints): must stay runnable via `torchrun --nproc_per_node=8 train_gpt2.py`; do NOT change `train_files`, `val_files`, `val_tokens` / keep `os.environ` usage; keep `save_checkpoint=False`; keep all `print0` statements and their args (log format); prefer editing only `train_gpt2.py`; "Your job will be run on a single 8xH100 node with access to all 8 GPUs."
- `entry_fname: train_gpt2.py`, `fnames: ['train_gpt2.py']`
- `metric_types: {n_steps: int, val_loss: float, train_time: int}`, `metrics_at_most: {val_loss: 3.28}`, `selection_metric: train_time`, `lower_is_better: true`
- `n_iterations: 3` here (overrides the `5` in `config/default.yaml`; README example passes `n_iterations=5` explicitly) [V]
- `slurm_config_args`: `nodes: 1, tasks_per_node: 8, gpus_per_node: 8, cpus_per_task: 12, job_ttl: 60` (minutes), `use_torchrun: true`, `account: maui`, `qos: maui_high`, `env_vars: {NANOGPT_TRAIN_FILES: "/home/zhaobc/fineweb_data/fineweb10B/fineweb_train_*.bin", NANOGPT_VAL_FILES: ".../fineweb_val_*.bin", NANOGPT_VAL_TOKENS: "10485760"}` — Meta-internal paths, must be overridden.

### 2.3 Hint levels [V]

| File | Level | Content | Size (record_1) |
|---|---|---|---|
| `level_0_diff.txt` | 0 (ground truth, not a hint level in the paper) | raw `git diff` R→R+1 | 270 lines |
| `level_1_pseudo.txt` | 1 "pseudocode" | "# Pseudo Code Changes …" | 81 lines |
| `level_2_description.txt` | 2 "text" | natural-language breakdown | 63 lines |
| `level_3_paper.txt` | 3 "mini-paper" | markdown paper w/ abstract, method | 188 lines |
| `level_9_*.txt` | extra knowledge | Muon post / FlexAttention notebook | — |

Provenance: `data/nanogpt_speedrun_knowledge_in_levels/llm_logs.jsonl` = 60 rows `{prompt, response, prompt_tokens, completion_tokens}` = 20 records × 3 prompts ("Given the git diff… generate a high-level pseudo code description", "…detailed natural language description…", "…concise Twitter thread-style summary…"), responses contain R1-style reasoning. Paper: "All hints were first drafted by R1, manually verified, and, where necessary, edited" [V].

**Naming gotcha [V]:** the paper-era launchers use `--knowledge_levels z 1 2 5 [12] [125]` and `data.jsonl` `hint_path`s end in `level_5_*.txt`, but the released files are `level_3_paper.txt`. Globbing `level_5_*.txt` matches nothing → silently zero-hint run. Use `3`/`[123]`. `z` = a non-matching glob = no hints.

**Record 6 quirk [V]:** `record_6/level_0_diff.txt` is 0 bytes (record 6→7 is a pure PyTorch upgrade; `record_6/train_gpt2.py` and `record_7/train_gpt2.py` are byte-identical); `record_6/level_1_pseudo.txt` is a placeholder about merge sort. Paper uses **19 tasks**: start records {1,2,3,4,5,7,8,…,20} (launcher docstring: "we omit record 6 due to PyTorch upgrade").

### 2.4 Released dataset file [V]
`workspace_templates/nanogpt_speedrun/data.jsonl`: 57 rows = 19 records × 3 levels; keys `id, current_results, current_train_script, current_results_path, current_train_script_path, target_time, hint, hint_path`. Self-contained (script + hint + target time inline) — the easiest thing to ingest into a unified repo. Paths inside are Meta-internal (`/home/zhaobc/scientist/...`).

### 2.5 Records (21 scripts; 20 hint dirs; 19 tasks) [V from paper Table E.1 + modded-nanogpt README; human times from `results.json` in ms]

| R | Innovation (R's script) | Leaderboard time | `results.json train_time` (ms) |
|---|---|---|---|
| 1 | llm.c baseline | 45 min | 2,968,348 |
| 2 | Tuned LR & rotary embeddings | 31.4 | 2,209,926 |
| 3 | Muon optimizer | 24.9 | 1,386,147 |
| 4 | Muon improvements | 22.3 | 1,301,740 |
| 5 | Pad embeddings, ReLU², zero-init proj, QK-norm | 15.2 | 949,528 |
| 6 | Distributed Muon overhead | 13.1 | 766,259 |
| 7 | PyTorch 2.5.0 upgrade (no code diff) | 12.0 | 773,072 |
| 8 | Untied embedding and head | 10.8 | 662,205 |
| 9 | Value/embedding skip conns, momentum warmup, logit softcap | 8.2 | 505,531 |
| 10 | bf16 activations | 7.8 | 477,150 |
| 11 | U-net skip connections & double LR | 7.2 | 442,985 |
| 12 | FlexAttention, 1024→64K ctx | 5.03 | 317,839 |
| 13 | Attention window warmup | 4.66 | 289,805 |
| 14 | Value embeddings | 4.41 | 273,107 |
| 15 | U-net value embeddings, code opts | 3.95 | 241,463 |
| 16 | Split value embeddings, block sliding window | 3.80 | 232,971 |
| 17 | Sparsify value embeds, improve rotary, drop attn layer | 3.57 | 220,374 |
| 18 | Logit softcap 30→15 | 3.4 | 211,840 |
| 19 | FP8 head, offset logits, LR decay | 3.142 | 199,442 |
| 20 | Merged QKV, long-short attention | 2.992 | 188,680 |
| 21 | Reduced batch size | 2.933 | 184,262 |

Note the re-timed baselines (records 1–18 run under torch 2.7.1 on Meta's cluster) differ from Keller's leaderboard (R1: 49.5 min vs 45). FSR uses the `results.json` numbers.

### 2.6 Data [V]
- Source: HF dataset `kjj0/fineweb10B-gpt2` via modded-nanogpt's `data/cached_fineweb10B.py [N]`: `fineweb_val_000000.bin` + `fineweb_train_000001..N.bin`, 100M GPT-2 tokens per shard, **200,001,024 bytes each; full set 103 shards = 20.7 GB**. No download script in llm-speedrunner itself.
- Scripts read `NANOGPT_TRAIN_FILES` (glob), `NANOGPT_VAL_FILES`, `NANOGPT_VAL_TOKENS=10485760`.
- Tokens consumed per record (num_iterations × batch tokens, from the scripts' hyperparameters; arithmetic is mine): R1 24576×262,144 = 6.44B (65 shards); R2 5.0B (50); R3 3.67B (37); R4 3.25B; R5–7 2.67B (27); R8 2.4B; R9–10 ~1.7B; R11 1.57B (16); R12 1875×524,288 = 0.98B (10); R13–20 0.73–0.92B; R21 1770×(8×48K) = 0.70B. Loader behaviour with too few shards: R12-style `advance()` wraps (`% len(self.files)`, silently repeats data); R19–21 use `iter(files)` with no cycle → crashes. **Download all 103 shards** for R1–R2, ≥ 37 for R3+, ≥ 10 for R12+.

---

## 3. Harness / agent interface

### 3.1 Launch [V]
```bash
python launch_scientist.py task=nanogpt_speedrun/speedrun_record_1 model=o3_mini \
  science_runner=aide ideator=dummy n_iterations=5 \
  knowledge_src_paths=["data/nanogpt_speedrun_knowledge_in_levels/record_1/level_1_*.txt"] \
  workspace_args.root_path=/your/workspace/dir \
  slurm_config_args.env_vars.NANOGPT_TRAIN_FILES=/data/fineweb10B/fineweb_train_*.bin ...
```
Hydra (`version_base=1.1`, `config_path=config`, `config_name=default.yaml`). `config/default.yaml` defaults: `task: collatz, model: r1_32b, science_runner: bon, ideator: base, coder: aider`, `log_llm_metrics: True`, `workspace_args.root_path: /checkpoint/maui/${USER}/scientist/workspace/${template_dirname}_${now}` (Meta path). `launch_scientist.py` writes `config.yaml` into the workspace root and **re-enters** an existing workspace if one exists there (resumable). `secrets` from `config/secrets/default.yaml` (`AZURE_API_KEY, AZURE_OPENAI_API_KEY, GEMINI_API_KEY`). Paper sweeps: `launchers/launch_llm_speedrun_agent_baselines.py` (submitit array job per (record, level, search, model); each agent run is itself a 6-day CPU slurm job that submits GPU jobs), `launch_llm_speedrun_agent_additional_knowledge.py`, `launch_cumulative_speedup.py`, `launch_slurm.py`.

### 3.2 Scaffold: `BoNScienceRunner` (`$REPO/core/runners/bon_science_runner.py`) [V]
It is a custom tree search, AIDE-*style* (the paper's "AIDE"/"Multi-AIDE" are parameterizations of this same class; coding is done by **Aider**, not the AIDE codebase — issue #2 is exactly this confusion). Parameters (`config/science_runner/{bon,aide}.yaml` + launcher presets):

| preset | runner | n_initial_hypotheses | n_hypotheses | n_iterations | debug_prob | max_bug_depth |
|---|---|---|---|---|---|---|
| flat | bon | 20 | – | 1 | 0 | – |
| tree | bon | 1 | 3 | 20 | 0 | – |
| forest | bon | 3 | 3 | 20 | 0 | – |
| aide | aide | 3 | 1 | 20 | 0.5 | 5 |
| multi-aide (paper main) | aide | 3 | 3 | 20 | 0.5 | 5 |
`config/science_runner/aide.yaml` itself says `n_initial_hypotheses: 5, n_hypotheses: 1, debug_prob: 0.5, max_bug_depth: 3`; `max_n_nodes: 20` everywhere (node budget M=20).

Per-iteration loop (`run()`, lines 201–336):
1. `select_next_open_version`: with prob `debug_prob` pick a random buggy **leaf** with `bug_depth ≤ max_bug_depth`; else the best `is_valid` node by `train_time` among `{'0'} ∪ current_versions`.
2. For each of `n_hypotheses` (1 if debugging; `n_initial_hypotheses` at i=0): `ideator.ideate(...)` — with `ideator=dummy` (`core/ideators/dummy_ideator.py`) this is a pass-through returning the `<knowledge>` hint string split into lines (no LLM call); `ideator=base` asks the LLM for a JSON `{summary, hypothesis}`.
3. `workspace.create_version(from_version=parent)` — copies the parent dir to `v_N`, writes `meta.json {parent, children, created_at}`.
4. `coder.code(...)` — Aider edits `v_N/train_gpt2.py` in place (diff edit format, `max_reflections: 5`, `use_git=False`), prompt built by `core/prompts/coder_prompts.py::basic_code_prompt` (task description + constraints + optional knowledge + package list + last-3 buggy-descendant history + STRICT_DIFF_PROMPT). Hints reach the coder as `ideas=` (the dummy ideator's line list; by code reading it is rendered via `str.format` of a Python list) unless `science_runner_args.knowledge_pass_to_coder=True`, which also injects the raw `<knowledge>` block.
5. `slurm_utils.submit_job(command='train_gpt2.py', working_dir=v_N, nodes=1, tasks_per_node=8, gpus_per_node=8, job_ttl=60, use_torchrun=True, env_vars=...)` — submitit job; rank 0 runs `torchrun --nproc_per_node=8 --rdzv_endpoint=$MASTER_ADDR:$PORT train_gpt2.py` with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. All hypotheses in an iteration run **concurrently** (3 × 8×H100 for multi-AIDE).
6. `JobObserver` polls every 10 s; on completion `_job_callback` → `ScienceRunner.set_results_for_version`: LLM (`assistant`) summarizes last 30,000 chars of stdout+stderr (`analysis_prompts.SUMMARIZE_LOGS_PROMPT`), metrics extracted (see §4.2), `is_valid=False` or slurm FAILED/TIMEOUT → `mark_as_buggy_from_version` (`bug_depth = parent.bug_depth + 1`), writes `v_N/results.json`.
7. Stop when `n_iterations` reached or `n_versions − 1 ≥ max_n_nodes`.

There is no separate evaluation job for nanogpt (`eval_fname: None`); the training run *is* the eval.

### 3.3 LLM backends (`$REPO/core/llm_client.py`, `core/coders/aider.py`) [V]
- Generic OpenAI-compatible: `OpenAI(base_url=model_url, api_key)`, 30-min timeout, `<think>` stripping — used for DeepSeek-R1 via vLLM (`serve_vllm.py`, `config/model/deepseek_r1.yaml` → internal `hpcaas` host).
- Azure OpenAI if `'https://azure' in model_url` (o3-mini, o1-preview, gpt-4o configs point at FAIR Azure gateways; `api_version 2024-12-01-preview`).
- Gemini via `litellm.completion(..., reasoning_effort='high')` — **hard-coded to `gemini/gemini-2.5-flash-preview-04-17`** in both `llm_client.py:77` and `aider.py:26` regardless of the `gemini_2_5.yaml` name `gemini-2.5-pro` [V] (discrepancy with the paper's "Gemini-2.5-Pro" label — [U] which was actually used for the paper runs).
- Claude via `setup_claude_openai_compatible_api.py`: Flask proxy to AWS Bedrock (`gunicorn -w 16 -b 0.0.0.0:8000 ... --timeout 1800`), model configs `claude_{3_5,3_7,4}_sonnet.yaml` → `http://localhost:8000/v1`.
- Aider model mapping `NAME_TO_AIDER_MODEL_NAME`; anything unknown becomes `openai/<name>`. Pinned `aider-chat==0.74.1`, `litellm==1.61.15`, `openai==1.60.2`.

### 3.4 Environment vs agent boundary — plugging in our own agent [V]

**Task-environment components (reusable standalone):**
- Start state: `workspace_templates/nanogpt_speedrun/record_R/{train_gpt2.py,results.json}`; hints: `data/.../record_R/level_L_*.txt`; baseline human times: `results.json` / `analysis/parse_levels.py::human_train_time_dict`.
- Runner: `utils/slurm_utils.py::submit_job` (submitit) + `JobObserver`; local mode `simulate_local_submitit_job` (`slurm_config_args.use_local_runs=true`).
- Scorer: `ScienceRunner.extract_metrics`/`set_results_for_version` (in-run validity + selection) and `analysis/parse_levels.py` (FSR, offline).

**Agent components:** `Ideator`, `Coder` (Aider), `assistant` Agent (log summarizer/metric parser), and the search policy inside `BoNScienceRunner`.

Three integration seams:
1. **Coder seam (keep their search, swap the coding agent):** subclass `core.agent.Agent` and implement
   `code(self, task_description:str, instruction:str|None, ideas, fnames:list[str], workspace:Workspace, version:str, bug_history:str|None=None, knowledge:str|None=None, max_retries=1) -> str`
   (signature at `core/coders/aider.py:108-119`; `BoNScienceRunner._run_exp` lines 111–124 passes exactly these kwargs). Contract: edit files in `workspace.resolve_path(fname, version=version)` in place; return value is only printed. Register with `config/coder/your_coder.yaml` (`coder_args._target_: core.coders.your_coder.YourCoder`, receives `secrets, model_url, model_name, system_prompt, log_llm_metrics`). Note: the shipped base `core/coders/base.py::Coder` is dead code (no `knowledge` kwarg → TypeError; typos `workspace.packges`, undefined `fname`) — Aider is the only working coder.
2. **Runner seam (own search loop):** subclass `core.runners.science_runner.ScienceRunner`, implement `async run(n_iterations)`, reuse `submit_job`/`JobObserver`/`set_results_for_version`; register via `config/science_runner/your_runner.yaml` (`science_runner_args._target_`, wired args `config, workspace, assistant, ideator, coder, slurm_config`).
3. **Bypass the harness entirely (simplest for a unified repo):** the environment is fully specified by: template script + hint file + env vars + `torchrun --nproc_per_node=8 train_gpt2.py` (60-min cap) + parse `step:N/M val_loss:X train_time:Yms` + validity `val_loss ≤ 3.28` + FSR against `human_train_time_dict`. Nothing in `core/` is required to reproduce the scoring.

**Docker:** none (no Dockerfile/compose anywhere in the tree) [V]. **Slurm:** yes, via submitit; account/qos `maui`/`maui_high` hardcoded in configs and launchers [V].

**Local (non-slurm) mode is broken for torchrun tasks [V by code reading]:** with `use_torchrun: true`, `_run_exp` sets `command = 'train_gpt2.py'` (bare), and `simulate_local_submitit_job` runs it with `subprocess.Popen(command, shell=True, cwd=v_N)` → shell "not found". Fix: set `use_torchrun=false` and `entry_fname` to a wrapper, or patch `submit_job` to prefix `torchrun --nproc_per_node=8`. Local mode also runs synchronously inside `submit_job` (no parallel hypotheses) and ignores `job_ttl`.

---

## 4. Scoring

### 4.1 FSR definition [V paper + code]
Paper: FSR_i = (t_i − t'_{i+1}) / (t_i − t_{i+1}), t_i = human time of record i, t'_{i+1} = best agent time to reach val_loss target; benchmark score = mean over tasks.
Code: `$REPO/analysis/parse_levels.py`:
- `human_train_time_dict` (lines 244–266): the 21 ms values in §2.5.
- `compute_gap_in_percentage` (306–337) / `_list` (385–424): `gaps[k] = human[k] − human[k+1]`; `recovered = human[k] − agent_best_time`; `FSR = recovered / gaps[k] if gaps[k] > 0 else 0`; NaN → 0. Negative FSR (agent slower than R_i) is **not clipped** in code; gap for 6→7 is negative → 0.
- Agent time = `min(train_time)` over all `v_*/results.json` in a run's workspace (`gather_metrics`, `analysis/plot_utils.py`), after `metrics.loc[val_loss >= 3.29, 'train_time'] = NaN` (`process_metrics`, line 223) — note the offline validity threshold is **3.29**, looser than the in-run 3.28.
- Cumulative variant (`launchers/launch_cumulative_speedup.py::get_training_time_reduction`, lines 205–226): same, plus any `train_time < 0.4 × baseline` is discarded as implausible; chain R1→R2'→… using the best workspace as the next template; stop when reduction < `failure_threshold`.

### 4.2 In-run measurement [V]
- `train_time` is the training script's own `training_time_ms` (`time.perf_counter` around `torch.cuda.synchronize()`; records ≥12 exclude the first 10 warmup steps; record 1 uses `time.time()` accumulated between val evals). The harness trusts the script's printed number — no external timer (`JobResult` has a `@todo` for runtime).
- Log line format (records 2–21): `step:{step}/{N} val_loss:{v:.4f} train_time:{ms:.0f}ms step_avg:...`. Record 1 prints `val loss {x}` plus the same `step:… val_loss:… train_time:…ms` line.
- Extraction (`core/runners/science_runner.py:105-185`): `metrics_utils.extract_best_line_metrics` (regex `(\w+)\s*:\s*([^,\s]+)`, keys `n_steps,val_loss,train_time`) — **I ran it on a real line and it returns `{}`** (`train_time:289805ms` fails `int()`, and `n_steps` never appears) → fallback to the **LLM** (`assistant.act(PARSE_METRICS_FROM_LOGS)`, JSON validated against the types). So in practice metric parsing is LLM-based, then `val_loss > 3.28` → `is_valid=False`. Buggy/invalid nodes are excluded from selection (`Workspace.get_top_k_versions` skips `is_valid=False`).
- Per-run cap: slurm `timeout_min = job_ttl = 60`; TIMEOUT → `JobStatus.FAILED` → buggy.
- Success: a node with `is_valid=True` (val_loss ≤ 3.28) and `train_time` < the R_i baseline; full credit when `train_time ≤ human[R+1]`.

### 4.3 Baseline `results.json` schema [V]
`{"job_status": "COMPLETED", "metrics": {"n_steps", "val_loss", "train_time"}, "hypothesis": "...", "outcome_summary": "..."}` — the runtime-written version uses key `status` and adds `metrics.is_valid`, `metrics.private`, `hypothesis` (from job metadata).

---

## 5. Logs / trajectories

### 5.1 Per-run artifacts [V]
Workspace root = `workspace_args.root_path` (must override; launchers append `_<slurm_id>`):
```
<root>/config.yaml                       # full resolved hydra config (analysis reads model, levels, runner from it)
<root>/v_0/                              # copy of template (train_gpt2.py, results.json)
<root>/v_N/train_gpt2.py                 # agent's edited script (full file per node)
<root>/v_N/meta.json                     # {parent, children, created_at, bug_depth, stable_ancestor_version}
<root>/v_N/results.json                  # {status, metrics{n_steps,val_loss,train_time,is_valid,private}, hypothesis, outcome_summary}
<root>/v_N/aider.txt                     # Aider chat history (when log_llm_metrics=True)
<root>/v_N/assistant_history.jsonl       # {prompt,response,prompt_tokens,completion_tokens} of summarizer/metric-parser calls
<root>/v_N/ideator_history.jsonl         # ideator LLM calls (empty for dummy)
<root>/v_N/coder_history.jsonl           # NB: bug at bon_science_runner.py:86-89 flushes ideator logs here; AiderCoder.flush_logs is a no-op
<root>/v_N/logs/<uuid>.txt               # training script's own log (cwd = v_N); record 1 also logs/<uuid>.pt
./submitit_logs/<jobid>_0_log.{out,err}  # training stdout/stderr (relative to launch CWD, NOT in the workspace)
./submitit_logs/local/job_<hash>/{stdout,stderr}   # local mode
```
Versions are copies, so a 20-node run stores 20 full scripts; `ignore_list` prevents history files from propagating to children.

### 5.2 Official run logs [V-by-absence]
No agent trajectories/solutions were released. The analysis notebooks (`analysis/*.ipynb`, `parse_levels.py`) read from internal paths (`/checkpoint/maui/zhaobc/scientist/workspace/record_*_2025040*`, `may11_o3_code.csv`, `code_analysis_with_all_versions_knowledge_o3_mini.json`, `/home/user123456/...`); nothing under `facebook/*` on HF matches; issue #1 (hint data on HF) is open and unanswered. The only released data are `data.jsonl` (57 tasks) and `llm_logs.jsonl` (hint-generation LLM logs). Paper compute: 6,840 agent runs × up to 20 h of 8×H100 [V paper].

---

## 6. Complications for running it ourselves

1. **8×H100 is baked in** [V]: prompt constraint (`torchrun --nproc_per_node=8`), `tasks_per_node/gpus_per_node: 8`, batch = 8 × 64K tokens in records 12–21 (`device_batch_size` per GPU), `assert val_tokens % (B*T*world_size) == 0`. Technically `--nproc_per_node=k` with edited batch sizes runs, but `train_time` becomes incomparable to `human_train_time_dict`, so FSR is meaningless without re-timing all 21 baselines on the target hardware (Meta did exactly this — `results.json` are their re-runs). Records 19–21 use FP8 (`torch._scaled_mm`) → Hopper-class GPUs required [V].
2. **Three environments** [V]: `conda_envs/speedrunner-1-11` (python 3.12, `torch==2.7.1`, `triton==3.1.0`, plus heavy extras: tensorflow 2.18, vllm 0.9.0, deepspeed, ray), `speedrunner-12-18` (`torch==2.7.1`, `triton==3.3.0`), and `speedrunner-19-21` = **conda-pack tarball only (5.03 GB, LFS), torch nightly, no yml** — the exact nightly is [U] without downloading it. The paper's "record 7 = PyTorch 2.5.0 upgrade" is moot in the release: everything ≤18 is re-timed on 2.7.1.
3. **FlexAttention** from record 12 (`torch.nn.attention.flex_attention`, `torch.compile(..., mode="max-autotune-no-cudagraphs")`); record 8/17 set `coordinate_descent_tuning = True` (now banned in the upstream speedrun but fine here). Compile/autotune time is excluded from `train_time` by the scripts' warmup logic but counts against the 60-min TTL.
4. **Data**: 20.7 GB from HF `kjj0/fineweb10B-gpt2`; env-var paths hardcoded to `/home/zhaobc/...`.
5. **Meta-internal hardcoding**: workspace root `/checkpoint/maui/...`, slurm `account: maui`/`qos: maui_high`, model URLs (Azure gateways, `*.hpcaas` vLLM hosts), `os.getlogin()` in launchers.
6. **Code rot**: `tests/test_metrics_utils.py` imports `scientist.utils` (stale package name → tests fail); base `Coder` broken; local-run + torchrun broken (§3.4); Gemini hard-coded to flash-preview; regex parser never matches nanogpt logs so scoring depends on an LLM call (§4.2); `level_5` vs `level_3` naming; no maintenance since 2025-10-09.
7. **License**: CC BY-NC 4.0 on the whole repo (hints, scripts, harness) — check compatibility before vendoring into our unified repo; the underlying record scripts come from `KellerJordan/modded-nanogpt` (MIT [U — not checked]).

Sources: [GitHub repo](https://github.com/facebookresearch/llm-speedrunner), [arXiv 2506.22419](https://arxiv.org/abs/2506.22419), [arXiv HTML v2](https://arxiv.org/html/2506.22419v2), [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt), [alphaXiv overview](https://www.alphaxiv.org/overview/2506.22419v2), [HF paper page](https://huggingface.co/papers/2506.22419), [OpenReview PDF](https://openreview.net/pdf?id=w98hMEjzu8).