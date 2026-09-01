# rollout/ — running the studies on PostTrainBench

`hv_recipe` / `hv_noop` and `hv_pack.sbatch` are the crossed rollout (a
deterministic executor replays corpus recipes). This README is about the
scientist × information study.

## The study

Three Claude Opus scientists (`claude-opus-4-6`, `claude-opus-4-8`,
`claude-opus-5`) post-train one base model for gsm8k under three conditions:

| | prior information | world-model agent | prompt | agent |
|---|---|---|---|---|
| **C1** | raw prior runs, read-only at `/home/ben/prior_runs` | none | `prompt_fulltraj` | `claude_fulltraj_noawm` |
| **C2** | the same raw runs | a second Claude Code session (`traj` arm) that reads them and answers `consult` | `prompt_wm_fulltraj` | `claude_wm:<model>:traj` |
| **C3** | none directly | a WMA session (`retrieval` arm) over the extracted experiment cards | `prompt_wm` | `claude_wm:<model>:retrieval` |

Every prompt is PTB's own `prompt.txt` plus sections before `## Rules`:
"Prior runs" for C1/C2, "The world-model agent" (`input/wma_section.md`) for
C2/C3 — so C2 minus that one section is byte-identical to C1 (tested).

Two prior-information versions: the split's 143 train-side runs, or all 193
including the 50 gemma runs; scores and agent identity visible in both. A C3
cell must see the same set through memory (`--memory-sides train` vs
`train,test`) as its C2 counterpart sees on disk.

## How a `claude_wm` cell runs

`agents/claude_wm/solve.sh` starts two Claude Code sessions in the sandbox:

1. **the world-model agent** — cwd `/home/ben/wma` (a copy of this repo's `wma/`:
   `CLAUDE.md`, the `consult` skill), model `WMA_MODEL` (default `claude-opus-4-8`,
   baked in by `setup.sh`), read-only tools plus `awm wm …`. It serves `consult`
   for the whole run and writes everything under `/home/ben/task/wm/`
   (`config.json`, `consults.jsonl`, `cards/`, its transcript).
2. **the scientist** — PTB's own Claude scaffold with the study prompt. It finds
   the WMA by name (`ListAgents`) and consults it with `SendMessage`; the
   answer arrives as a message. It owns the GPU and every evaluation.

`AGENT_CONFIG` = `<scientist model>[:<arm>[:<memory sides>]]`; arms `null`,
`retrieval`, `traj`, `llm` decide what the WMA may read. The WMA's model never
varies with the scientist's.

## One-time setup (host)

```bash
bash rollout/setup.sh                       # private PTB checkout + agents + prompts + the extra-binds patch
python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_143 --sides train
python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_193 --sides train,test
# memory for C3: seed once from the extracted cards
AWM_SESSION_DIR=/tmp/seed awm wm init --arm retrieval --memory-root /data/wm-memory
AWM_SESSION_DIR=/tmp/seed awm wm memory seed results/exp-cards/gsm8k-gemma-holdout-v1 --side train
AWM_SESSION_DIR=/tmp/seed awm wm memory seed results/exp-cards/gsm8k-gemma-holdout-v1 --side test   # "gemma in" version only
```

## Launching

```bash
export PTB_MODEL=google/gemma-3-4b-pt PTB_NUM_HOURS=10
export PRIOR_RUNS=/data/prior_runs_143 WM_MEMORY=/data/wm-memory
# C1 + C2
sbatch rollout/wm_pack.sbatch \
  claude_fulltraj_noawm:claude-opus-4-6 claude_wm:claude-opus-4-8:traj claude_wm:claude-opus-5:traj \
  claude_fulltraj_noawm:claude-opus-4-8 claude_wm:claude-opus-5:traj   claude_wm:claude-opus-4-6:traj \
  claude_fulltraj_noawm:claude-opus-5   claude_wm:claude-opus-4-6:traj
# C3
PRIOR_RUNS= sbatch rollout/wm_pack.sbatch \
  claude_wm:claude-opus-4-6:retrieval claude_wm:claude-opus-4-8:retrieval claude_wm:claude-opus-5:retrieval
```

## What comes back

PTB's result directory per cell (`solve_out.txt` — the scientist's stream-json
trajectory — `metrics.json`, judgements) plus, for `claude_wm` cells,
`task/wm/`: every consult with its request, response, verdict, prediction and
timestamp (`consults.jsonl`, `cards/exp-NN/consult-NN.json`), the recorded
outcome, and the WMA's own transcript (`wma_session.jsonl`). Scoring the world
model = joining `consults.jsonl` to `metrics.json`.

## Pieces

| file | role |
|---|---|
| `patches/apply_extra_binds.py` | adds `POST_TRAIN_BENCH_EXTRA_BINDS` to `run_task.sh` (idempotent) — the read-only mounts for prior runs and memory |
| `build_prompts.py` | PTB `prompt.txt` + sections → `prompt_fulltraj`, `prompt_wm`, `prompt_wm_fulltraj`; review copies under `prompts/`; `input/instruction.md` = `prompt_wm` |
| `agents/claude_fulltraj_noawm/` | C1: PTB's Claude scaffold, unchanged |
| `agents/claude_wm/` | C2/C3: the two-session cell |
| `wm_pack.sbatch` | 8 cells/node, `<agent>:<config>` arms, per-cell prompt and binds |
| `../wma/` | the WMA session's project dir |
| `../tools/build_prior_runs.py` | the prior-runs directory + `INDEX.md` |
