# rollout/ — running the studies on PostTrainBench

Two studies live here. `hv_recipe` / `hv_noop` and `hv_pack.sbatch` are the
crossed rollout (a deterministic executor replays corpus recipes; see the
comments in those files). This README is about the second one.

## The scientist × information study

Three Claude Opus scientists (`claude-opus-4-6`, `claude-opus-4-8`,
`claude-opus-5`) post-train one base model for gsm8k under three information
conditions:

| | prior information | WMA | prompt | agent |
|---|---|---|---|---|
| **C1** | raw files of the prior runs, read-only at `/home/ben/prior_runs` | none | `prompt_fulltraj` | `claude_fulltraj_noawm` |
| **C2** | the same raw files | runtime + arm | `prompt_wm_fulltraj` | `claude_wm` |
| **C3** | none directly; WMA memory seeded from the reconstructed cards | runtime + `retrieval` | `prompt_wm` | `claude_wm` |

Two versions of "prior": the split's 143 train-side runs, or all 193 including
the 50 gemma runs. Scores and agent identity are visible in both (decision
2026-08-31). Whatever set a cell sees, the C3 cell it is compared with must
see the same set through memory (`--memory-sides train` vs `train,test`).

### One-time setup (host)

```bash
# 1. private PTB checkout + agents + prompts + the extra-binds patch
bash rollout/setup.sh                       # HV_PTB_DIR / HV_PTB_SHA / AWM_REPO_REF override the defaults

# 2. the prior-runs directories (copies; ~0.9 GB each)
python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_143 --sides train
python tools/build_prior_runs.py posttrainbench/gsm8k-gemma-holdout-v1 --out /data/prior_runs_193 --sides train,test

# 3. a WMA memory seeded from the reconstructed cards (train side, optionally test)
awm wm --dir /tmp/seed init --memory-root /data/wm-memory
awm wm --dir /tmp/seed memory seed results/exp-cards/gsm8k-gemma-holdout-v1 --side train
awm wm --dir /tmp/seed memory seed results/exp-cards/gsm8k-gemma-holdout-v1 --side test   # only for the "gemma in" version
```

`setup.sh` also copies `agents/claude_non_api/oauth_token` from the shared
checkout into both new agent dirs — `run_task.sh` binds it into the sandbox
only when it sits in the agent's own directory.

### Launching

```bash
export PTB_MODEL=google/gemma-3-4b-pt PTB_NUM_HOURS=10
export PRIOR_RUNS=/data/prior_runs_143 WM_MEMORY=/data/wm-memory

sbatch rollout/wm_pack.sbatch \
  claude_fulltraj_noawm:claude-opus-4-6  claude_wm:claude-opus-4-8:retrieval  claude_wm:claude-opus-5:retrieval \
  claude_fulltraj_noawm:claude-opus-4-8  claude_wm:claude-opus-5:retrieval    claude_wm:claude-opus-4-6:retrieval \
  claude_fulltraj_noawm:claude-opus-5    claude_wm:claude-opus-4-6:retrieval
```

`PRIOR_RUNS_FOR=claude_fulltraj_noawm` restricts the prior-runs bind to C1, so
`claude_wm` cells in the same pack run as C3. Held-out cells append `:ro` to
the config (`claude_wm:claude-opus-4-8:retrieval:train:ro`) and get memory
read-only.

`AGENT_CONFIG` for `claude_wm` is `<model>[:<arm>[:<memory sides>[:ro]]]`.

### What comes back

The usual PTB result directory per cell (`solve_out.txt` — the stream-json
trajectory — `metrics.json`, judgements) plus, for `claude_wm` cells,
`task/wm/`: `events.jsonl`, every card with its contract, observations, pings,
replies, and seal, and `awm_sha.txt` naming the runtime that ran. `task/` is
copied whole, so nothing there is size-capped.

### Pieces

| file | role |
|---|---|
| `patches/apply_extra_binds.py` | adds `POST_TRAIN_BENCH_EXTRA_BINDS` to `run_task.sh` (idempotent) |
| `build_prompts.py` | writes `prompt_fulltraj.txt`, `prompt_wm.txt`, `prompt_wm_fulltraj.txt` into the checkout; review copies under `prompts/` |
| `agents/claude_fulltraj_noawm/` | C1: PTB's Claude scaffold, unchanged |
| `agents/claude_wm/` | C2/C3: clones awm at `AWM_REPO_REF`, `awm wm init`, skill + Stop hook, then the same Claude invocation |
| `wm_pack.sbatch` | 8 cells/node, `<agent>:<config>` arms, per-cell prompt and binds |
| `../tools/build_prior_runs.py` | the prior-runs directory + `INDEX.md` |
