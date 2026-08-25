<!-- 调研原文,2026-08-24 由调研 agent 实际 clone 仓库/读取 HF 数据集后产出;VERIFIED = 在文件或页面里看到,UNVERIFIED = 推断。scratchpad 路径为当时的临时 clone 位置。 -->
<!-- 主题:Prime Intellect frontier-automated-speedrun -->

# Prime Intellect autonomous-speedrun releases — structural report

Everything below was checked against real files cloned to `/tmp/claude-30001/-home-gangda-workspace-DeepCommit-ai-hierarchy-verifier/97160149-2e2d-43ef-a087-7b3437df11a9/scratchpad/pi-speedrun/{frontier-automated-speedrun,experiments-autonomous-speedrunning}` (plus `blog.html`/`blog.txt` there), the GitHub API, and the blog. Tags: **[V]** = seen in a file/page, **[U]** = unverified/inferred.

---

## 1. Repos

| | `frontier-automated-speedrun` | `experiments-autonomous-speedrunning` |
|---|---|---|
| URL [V] | https://github.com/PrimeIntellect-ai/frontier-automated-speedrun | https://github.com/PrimeIntellect-ai/experiments-autonomous-speedrunning |
| License [V] | **none** (API `license: null`, no LICENSE file) | **none** |
| Stars / forks [V] | 16 / 4 (API, 2026-08-24) | 110 / 9 |
| Created / last push [V] | 2026-08-14T19:47Z / 2026-08-15T00:55Z | 2026-05-13 / 2026-05-14T21:26Z |
| Commit activity [V] | 5 commits total, all on 08-14/08-15 (`c4581c21` sanitized traces → `4eaf9cd1` programs → `5a978d39` README+baseline → `acb1a450` leaderboard → merge #18). Dormant since. | 10 commits, all 05-14. Dormant since. |
| Clone size [V] | 99 MB on disk (API size 49.8 MB); 128 files | 4.2 GB on disk (API 276 MB packed); 74,653 files |
| PRs [V] | 18 total: #1–#17 **open** = record PRs; #18 merged (leaderboard) | 4, all closed (README, seed_reverify export, flatten-export) |
| Issues [V] | 0 real issues (`open_issues_count: 17` = the open PRs) | 0 |

Top-level tree of `frontier-automated-speedrun` (depth 2) [V]:
```
README.md                 (5177 B)  program.md (5713 B)   program-serial.md (5575 B)
train_gpt_simple.py       (16097 B, 407 lines — the baseline)
traces/                   (50 MB, 124 files = 41 × {events,scratch,subagents} + manifest.json.gz)
  events-<run>.json.gz    scratch-<run>.json.gz    subagents-<run>.json.gz    manifest.json.gz
```
Nothing else. No `run.sh`, `verify.py`, `requirements.txt`, data script, sandbox/launcher code, or monitor reports (grep for "monitor" outside traces → nothing) [V]. `records/` exists only on the PR branches.

Blog [V]: https://www.primeintellect.ai/blog/measuring-autonomous-research (Elie Bakouch, Aug 14 2026). Note: the repo README links `/blog/comparing-frontier-models-nanogpt-speedrun`, which returns **404** [V]. Results site: https://www.primeintellect.ai/research/nanogpt-speedrun (+ `/traces/<run-id>` viewer); it hosts no extra downloads [V, via fetch].

---

## 2. Task definition

### program.md (standard) — key rules, verbatim [V]
- Goal: *"Make `train_gpt_simple.py` clear the significance bar — an 8-trial mean val loss **< 3.27859** — in the FEWEST optimizer steps (`train_steps`). Baseline = 3290"*
- Record: *"the mean of seeds 0xC0FFEE+0..7 (`bash run.sh 8`) coming in **< 3.27859** (= `3.28 - 0.004/sqrt(8)`, one-sided p < 0.001 at per-run σ ≈ 0.0013) at a strictly LOWER `train_steps` than the baseline. `python verify.py logs/<uuid>.txt` checks it."* / *"Non-cherry-picked: the whole 8-trial set for that exact config"*
- Screening: `bash run.sh` = 1 trial (*"promising if < 3.276"* per verify.py), widen progressively, *"Don't stop a run early on its partial loss"*, *"Stack small gains"*
- Edit surface: *"**Editable:** only the optimizer, its hyperparameters, the schedule, and the initialization (the `Optimizer` and `Init & Optim Hyperparams` sections). **Frozen:** the dataloader, architecture, batch size, sequence length, validation config, and data"*
- Anti-cheat: *"One step = one forward-backward pass. No extra gradient evaluations, lookahead inner steps with extra backwards…"*; *"**No** val-based early stopping"*; *"Keep `torch.compile` enabled"*; *"Self-contained: keep all code in `train_gpt_simple.py` — no third-party optimizer imports"*; *"A per-trial seed is set + logged by frozen infra (`seed = 0xC0FFEE + trial_idx`, logged as `seed:<n>`); don't re-seed to cherry-pick"*; *"**No lookups.** You have no network on this tier"*; *"None on this tier — native web is disabled and there is no `papers` CLI"*
- Running: *"**One run at a time.** A run uses the whole node (`run.sh` launches torchrun across all 8 GPUs)"*; *"**Run each experiment in a subagent.** Edit the code yourself, then spawn a subagent to launch and watch the run… Don't tail logs, poll `nvidia-smi`, or schedule wake-ups yourself."*; *"`run.sh` auto-cancels any run that exceeds the time cap (`RUN_TIMEOUT`, default 2h)"*
- Approach: prefer new optimizer families over sweeps; *"'baseline is optimal' is never a valid place to stop"*; pruning round every ~8 ideas.
- Memory: *"Keep and organize your durable memory in `scratchpad/`… one mandatory file: `scratchpad/thread.md`, a running log where you record every decision and its outcome."*

`program-serial.md` [V] differs **only** in the "Running experiments" bullet: *"**Wait for each run to finish, then act on its result.** Edit the code yourself, launch the run, and wait until it completes or fails; then check it with `python verify.py`… don't tail logs or poll `nvidia-smi`."* (no subagent instruction). README: used between **July 20 and August 13**; those runs are labeled and being rerun.

### train_gpt_simple.py (baseline) [V]
407 lines / 16,097 B; sections `Dataloader`, `Architecture`, `Optimizer`, `Setup`, then per-trial `Init & Optim Hyperparams` and `Training and Validation`.
- Model: GPT, `vocab_size=50304, num_layers=12, model_dim=768`, head_dim 128, RMSNorm, half-truncated RoPE, `F.scaled_dot_product_attention(..., scale=0.12, is_causal=True)`, ReLU² MLP, logits soft-capped `15*tanh-like`, `nn.Embedding(...).bfloat16()` (everything downstream runs bf16 via `type_as`), `F.cross_entropy(reduction="sum")`. `model.compile(dynamic=False)`. **No FlexAttention, no FP8, no value-embeddings/U-Net** — a deliberately simplified script.
- Data: `distributed_data_generator("data/fineweb10B/fineweb_train_*.bin", batch_size)`; `.bin` = modded-nanogpt format (256×int32 header, magic `20240520`, uint16 tokens). `val_tokens = 20*524288`, `batch_size = 8*64*1024` (=524,288 tokens/step), `mbs = 64` sequences, `seq_len=1024`. `assert 8 % world_size == 0` (1/2/4/8 GPUs; micro-batch accumulation, grads `all_reduce SUM`).
- Trials: `num_trials = int(sys.argv[-1])`; per trial `seed = 0xC0FFEE + trial_idx; torch.manual_seed; cuda.manual_seed_all; print0(f"seed:{seed}")`. **`train_steps = 3290`** is a plain assignment inside the editable section.
- Baseline optimizer: `AdamW` (embed lr 0.7, head lr 0.004, scalars lr 0.015, betas (0.8,0.95), eps 1e-10, wd 0.001, fused) + `Muon` (lr 0.025, wd 0.05, mu 0.95, Nesterov, NS5 12 iters, coeffs (2,-1.5,0.5), bf16, aspect-ratio scale); schedule `set_hparams`: constant then linear decay over last `cooldown_frac=0.7`. Init: `proj` weights zero, embed N(0,1), others `std=sqrt(0.33)/sqrt(fan_in)`.
- Logging (frozen infra): rank 0 writes `logs/<uuid4>.txt`, first content = **the full source of the script itself**, then `Running PyTorch … on <gpu> with world_size N`, `seed:<n>`, and `step:{s}/{T} val_loss:{v:.5f} train_time:…s step_avg:…ms` every 125 steps (25 in last 10%) + at step T. Optional W&B (`ee-speedrun-optimizer-wandb`), no-op unless keys set.

### Dataset [V from traces]
Work dir contained `data/cached_fineweb10B.py` (747 B) and symlink `data/fineweb10B -> <runner-home>/fineweb10B` (read-only, `dr-xr-xr-x`). Script content seen in two traces = **upstream modded-nanogpt downloader** (`hf_hub_download(repo_id="kjj0/fineweb10B-gpt2", …)`, val shard 0, `num_chunks=103` default, argv override). PI's copy held **40 train shards + 1 val shard**, each `200,001,024 B` = 100M tokens → ~8.0 GB (`du` = 8,008,112 KB) [V]. 3290 steps × 524,288 = 1.72 B tokens, so 40 shards suffice. Also present: `.gitignore` (`data/fineweb10B/ logs/ wandb/ __pycache__/ *.pyc .DS_Store`), `.git/` (work dir is a git repo; agents used `git log/commit`), `requirements.txt` = `torch==2.11\nhuggingface_hub\nwandb` [V]. Python 3.12.3, venv `<runner-home>/.ee-speedrun/venv-optimizer-<redacted>` [V].

### Missing for a self-run, and what is recoverable
Not in the repo: `run.sh`, `verify.py`, `requirements.txt`, `data/cached_fineweb10B.py`, `.gitignore`, sandbox scripts (bwrap/netns/proxy), the launcher/driver (goal injection, `continue` loop, restart/compaction handling), the LLM monitor, subagent prompt templates (agent-authored anyway). **But agents `cat`/`Read` `run.sh` and `verify.py` in nearly every run and the tool results are in the traces, so both are recoverable verbatim** (e.g. `events-kimi-k3--prime-agent--11df73b051f8` event i=12; `events-claude-fable-5--claude-code--6e0322deb000` i=11/13). `ls -la` [V]: `run.sh` 1167 B, `verify.py` 2974 B.

`run.sh` (verbatim) [V]:
```bash
#!/usr/bin/env bash
# Launch the optimizer baseline on all visible GPUs (single node, {1,2,4,8} x {A,H}100).
#   bash run.sh        # 1 trial  — screen an idea
#   bash run.sh 8      # 8 trials — the fixed validation set (confirm a record)
set -uo pipefail
NGPU="$(nvidia-smi -L | wc -l)"
# Single-node rendezvous on loopback. (torchrun --standalone can hang resolving an
# unroutable node FQDN.) Keep NCCL on the loopback interface for a single-node job,
# and let Triton/Inductor find libcuda without /sbin/ldconfig (missing in some images).
export MASTER_ADDR=127.0.0.1
export NCCL_SOCKET_IFNAME=lo
export TRITON_LIBCUDA_PATH="${TRITON_LIBCUDA_PATH:-/usr/lib/x86_64-linux-gnu}"
# Watchdog: auto-cancel a hung or runaway run. Default 2h (an 8-trial baseline set
# is ~1h); override with e.g. RUN_TIMEOUT=30m.
RUN_TIMEOUT="${RUN_TIMEOUT:-2h}"
status=0
timeout -k 30s "$RUN_TIMEOUT" \
  torchrun --nnodes=1 --nproc_per_node="${NGPU}" \
    --master_addr=127.0.0.1 --master_port=29500 \
    train_gpt_simple.py "${1:-1}" || status=$?
if [ "$status" -eq 124 ]; then
  echo "ERROR: run exceeded RUN_TIMEOUT=$RUN_TIMEOUT and was cancelled." >&2
fi
exit "$status"
```

`verify.py` (verbatim, whitespace-normalized) [V]:
```python
#!/usr/bin/env python3
"""Significance check for optimizer-benchmark runs.
Validation is either a single screening run or the fixed 8-trial set:
  - a single run with final val loss < 3.276 looks promising (screen only);
  - the 8-trial set (seeds 0xC0FFEE+0..7, `bash run.sh 8`) is a record iff its
    mean < 3.27859  ==  (3.28 - mean) * sqrt(8) >= 0.004  with sigma = 0.0013
    (one-sided p < 0.001).
Usage:
    python verify.py logs/<uuid>.txt           # parse final val losses from a logfile
    python verify.py --loss 3.2781 3.2790 ...   # pass final losses directly
"""
import sys, re, glob, argparse
TARGET = 3.28
RECORD_BAR = 3.27859   # 8-trial mean must be below this  (= 3.28 - 0.004/sqrt(8))
SCREEN_BAR = 3.276     # a single run below this looks promising (= 3.28 - 0.004)

def final_losses_from_file(path):
    """Return every final-step val loss in a logfile (one per trial)."""
    out = []
    with open(path) as f:
        for line in f:
            m = re.search(r"step:(\d+)/(\d+)\s+val_loss:([0-9.]+)", line)
            if m and m.group(1) == m.group(2):  # step == train_steps -> final
                out.append(float(m.group(3)))
    return out

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="*", help="logfiles (globs ok) to parse")
    ap.add_argument("--loss", type=float, nargs="+", default=[], help="final val losses, passed directly")
    args = ap.parse_args()
    losses = list(args.loss)
    for p in args.paths:
        for path in sorted(glob.glob(p)):
            fl = final_losses_from_file(path)
            if not fl: print(f"  ! no final val_loss found in {path}")
            for v in fl: print(f"  {path}: {v:.5f}")
            losses.extend(fl)
    n = len(losses)
    if n == 0:
        print("No losses found. Pass a logfile or --loss values."); sys.exit(2)
    mu = sum(losses) / n
    print("-" * 56)
    if n == 8:
        passed = mu < RECORD_BAR
        print(f"8-trial set   mean = {mu:.5f}   (record bar: < {RECORD_BAR})")
        print("RESULT: " + ("PASS  record-valid (p<0.001)" if passed else "FAIL  mean is not below the record bar"))
        sys.exit(0 if passed else 1)
    if n == 1:
        ok = losses[0] < SCREEN_BAR
        print(f"single run    val = {mu:.5f}   (promising if < {SCREEN_BAR})")
        print("RESULT: " + ("PROMISING — confirm with the 8-trial set: bash run.sh 8" if ok else "not low enough to be worth confirming on its own"))
        sys.exit(0 if ok else 1)
    print(f"n = {n}: validation is 1 trial (screen) or 8 trials (record), not {n}.")
    print(f"mean so far = {mu:.5f}. Run the full 8-trial set to confirm a record: bash run.sh 8")
    sys.exit(2)

if __name__ == "__main__": main()
```
Important: `verify.py` does **not** check the source snapshot, seeds, or `train_steps` consistency — it only parses final `step:T/T val_loss:` lines. Anti-cheat therefore rested on the frozen seed code + LLM monitor + manual audit at export ("exact record state recovered from the run's traces").

Blog's description of infra (§7) plus these two files is sufficient to reimplement the *run* environment; the *launcher* (goal injection/continue loop) is only reconstructable from trace patterns (§7).

---

## 3. Trajectory layout and schema (MOST IMPORTANT)

### Layout [V]
Flat directory `traces/`, one triplet per run, run id = `<model_id-slug>--<harness>--<12-hex key>`:
```
traces/events-<run>.json.gz      # main-agent transcript (single JSON object, NOT JSONL)
traces/subagents-<run>.json.gz   # child transcripts (present for all 41, may be empty)
traces/scratch-<run>.json.gz     # scratchpad/ files at end of run
traces/manifest.json.gz          # {baseline, record_bar, target, human_record, generated, runs:[41]}
```
Sizes: 41 runs, 50 MB gz total, largest events file 4.4 MB gz (`deepseek-v4-pro--prime-agent--e80bf5f37abf`), 172,694 main events + 1,780 subagents in total [V].

**Format: PI normalized every harness into one common event schema** (harness-native rollouts were converted; codex/pi/prime-agent events keep `source_file`/`source_line` pointers into the native JSONL) [V].

### `events-<run>.json.gz` [V]
```json
{"run": "<run-id>", "events": [ {event}, ... ]}
```
Event fields (union; presence per harness in the table below):

| field | type | meaning |
|---|---|---|
| `type` | `"text" \| "thinking" \| "tool_use" \| "tool_result"` | only these 4 values exist |
| `role` | `"user" \| "assistant"` | `tool_result` always has role `user` |
| `ts` | ISO-8601 Z (ms; kimi-code has µs) | wall clock |
| `i` | int | 0-based index in the list |
| `turn` | int | assistant API-call counter (user msgs carry the preceding turn) |
| `text` | str | message / thinking / tool output |
| `redacted` | bool | on `thinking`: `true` ⇒ `text==""` (closed CoT); on tool_use/results in some harnesses |
| `usage` | `{in,out,cache_read[,cache_write \| reasoning_out \| cache]}` | **repeated on every assistant event of the same `turn`** — dedupe by `turn` |
| `tool` | str | tool name (harness-native names, see table) |
| `summary` | str | one-line rendering of the tool input (Bash command / path / JSON args) |
| `args` | dict | raw tool input (codex, pi, prime-agent only) |
| `file`, `old`, `new`, `desc` | str | Claude-Code-style Read/Edit/Write path, Edit old/new strings, Bash description |
| `is_error`, `lines` | bool, int | on `tool_result` |
| `dur_ms` | int | tool duration (pi, prime-agent) |
| `session_id`, `source_file`, `source_line` | str | native-log provenance (codex, pi, prime-agent) |
| `summary_only`, `compacted` | bool | codex: thinking is a summary; user msg is a compaction handoff |

Per-harness differences [V]:

| harness (manifest `harness`) | tools seen | thinking | usage keys | args | extra fields |
|---|---|---|---|---|---|
| `claude-code` (14 runs; also used for Kimi K3 / Qwen / DeepSeek via OpenRouter) | `Bash, Edit, Read, Agent, Write, Monitor, ToolSearch, SendMessage, TaskOutput…` | Anthropic models: `redacted:true, text:""`; open-weight models via CC: full text | `in,out,cache_read,cache_write` (Anthropic) / none (OpenRouter-backed) | no (`summary`+`file/old/new/desc`) | `<task-notification>` user msgs |
| `claude-code-goal` (1) | same + `TaskCreate/TaskUpdate/TaskOutput` | full (GLM) | none | no | `/goal` command wrapper, Stop-hook msgs |
| `codex` (7) | `exec` (JS `tools.exec_command`), `wait`, `spawn_agent`, `wait_agent`, `list_agents`, `followup_task`, `send_message` | mostly `redacted:true` (2528/2729), some `summary_only` | `in,out,cache_read,reasoning_out` | yes (`args.input` = JS snippet) | `session_id, source_file, source_line, compacted` |
| `prime-agent` (6) | **only `ipython`** (`args.code`, often `%%bash` cells) | full for open models; `redacted` for Anthropic/OpenAI | `in,out,cache_read` | yes | `dur_ms, source_file, source_line` |
| `kimi-code` (3) / `kimi-code-goal` (1) | `Edit, Bash, Agent, TodoList, Read, Grep, TaskOutput, UpdateGoal` | full | **none** | no | µs timestamps |
| `grok-cli` (5) | `run_terminal_command, get_command_or_subagent_output, search_replace, read_file, grep` | **no thinking events** | none | no | `tool` is `null` on tool_results; Grok 4.5 run has no timestamps |
| `qwen-code` (2) | `run_shell_command, edit, read_file, write_file, get_goal, task_stop, grep_search, list_directory` | full | `in,out,cache` | no | |
| `pi` (2) | `bash, edit, read, write` | GLM: full; Muse Spark: none | `in,out,cache_read` | yes | `dur_ms, source_file, source_line` |

Truncation/sanitization [V]: Claude-Code-family tool_results are capped (marker `…[+N chars]`; 233 truncated results in claude-code, cap appears at ~1.5 KB for `Read` and ~8 KB for Bash; 402 `summary` truncations); codex/pi/prime-agent/qwen results are **not** truncated (max 118 KB). Redaction tokens: `<runner-home>`, `<gpu-model>`, `<gpu-hardware>`, `<version>`, `<redacted>`, `<private-storage>`, `<shared-filesystem>`, `<network-filesystem-temp-file>`, `<platform-root>`, `<agent-service-account>`, `<encrypted-payload-redacted>`; UUIDs rewritten to `00000000-0000-4000-8000-0000000NNNNN`. Leaks: grok-cli/prime-agent `args` still contain `/home/elie/.ee-speedrun/attempts/<run>/…` and `elie 2012` in `ls -la`.

Real excerpt — first 8 events of `events-claude-fable-5--claude-code--4ed2e4e07637` (the 2,726 record run), text shortened [V]:
```json
{"type":"text","role":"user","ts":"2026-07-11T21:28:04.510Z","text":"Read program.md and follow it exactly. Run fully autonomously — never stop, never ask for input. Goal: reach mean val loss < 3.28 (meeting the significance bar in program.md) in the FEWEST train_steps possible — keep beating the current best.","i":0,"turn":0}
{"type":"tool_use","role":"assistant","ts":"2026-07-11T21:28:10.505Z","tool":"Read","summary":"<runner-home>/.ee-speedrun/attempts/claude-fable-5--claude-code--4ed2e4e07637/work/program.md","file":"<runner-home>/.../work/program.md","i":1,"turn":1,"usage":{"in":2,"out":88,"cache_read":15304,"cache_write":6245}}
{"type":"tool_result","role":"user","ts":"2026-07-11T21:28:10.532Z","tool":"Read","is_error":false,"text":"1\t# program.md — autonomous optimizer speedrun\n2\t...","i":2,"turn":1}
{"type":"thinking","role":"assistant","ts":"2026-07-11T21:28:15.721Z","text":"","redacted":true,"i":3,"turn":2,"usage":{"in":2,"out":269,"cache_read":21549,"cache_write":2475}}
{"type":"tool_use","role":"assistant","ts":"2026-07-11T21:28:17.751Z","tool":"Bash","summary":"ls <runner-home>/.../work/ && ls .../work/scratchpad/ 2>/dev/null && cat .../scratchpad/thread.md 2>/dev/null | head -100","desc":"List work dir and read scratchpad thread","i":4,"turn":2,"usage":{"in":2,"out":269,"cache_read":21549,"cache_write":2475}}
{"type":"tool_result","role":"user","ts":"2026-07-11T21:28:17.881Z","tool":"Bash","is_error":false,"text":"data\nprogram.md\nrequirements.txt\nrun.sh\nscratchpad\ntrain_gpt_simple.py\nverify.py","lines":7,"i":5,"turn":2}
{"type":"thinking","role":"assistant","ts":"2026-07-11T21:28:23.955Z","text":"","redacted":true,"i":6,"turn":3,"usage":{"in":2,"out":392,"cache_read":24024,"cache_write":313}}
{"type":"tool_use","role":"assistant","ts":"2026-07-11T21:28:24.915Z","tool":"Read","summary":".../work/train_gpt_simple.py","file":".../work/train_gpt_simple.py","i":7,"turn":3,"usage":{...}}
```
Other real samples: codex `{"tool":"exec","args":{"input":"const r = await tools.exec_command({\"cmd\":\"sed -n '1,240p' program.md\",\"workdir\":\"…/work\",\"yield_time_ms\":10000,\"max_output_tokens\":20000});\ntext(r.output);\n"}}`; prime-agent `{"tool":"ipython","args":{"code":"%%bash\ncat run.sh; echo =====; cat verify.py; …"}}` with result `{"tool":"ipython","is_error":false,"text":"#!/usr/bin/env bash…","lines":118,"dur_ms":57}`; pi `{"tool":"read","args":{"path":"program.md"},"summary":"program.md"}`.

### `subagents-<run>.json.gz` [V]
```json
{"index": [ {"id","label","n_events","n_thinking","n_tool_use","paper_calls","cot","t_start","t_end"
             /* codex adds: */ ,"parent_id","agent_path"} , ...],
 "events": { "<id>": [ {same event schema as above}, ... ] } }
```
`label` = first user message of the child truncated (`…[+601 chars]`), or codex agent name (`Kuhn`, `Socrates`, …) / `agent_path` (`<platform-root>/e000_baseline`). 28/41 runs have non-empty subagents (`has_subagents:true`); pi has "no subagent surface"; grok/qwen none. Counts: Fable-5 record run 213; codex Luna 152; prime-agent Sol 16.

**Parent↔child linkage (no explicit id in the parent event):**
- claude-code/kimi-code: parent `tool_use tool:"Agent"` (`summary` = task description, no id) → child `index[k].t_start` is 3–9 ms after it; index is chronological and 1:1 with `Agent` calls (213/213). 210/213 were synchronous (`tool_result tool:"Agent"` immediately follows, text = child's final report); background ones surface later as user `<task-notification>` with `<task-id>` = child `id` (e.g. `afae0b036154d5eae`). Note: most `<task-notification>`s (562/1045) are for Claude Code `Monitor`/background-Bash tasks, not subagents.
- codex: `spawn_agent {"task_name","fork_turns","message":"<encrypted-payload-redacted>"}` → result `{"task_name":"<platform-root>/e000_baseline"}`; child `index.agent_path` matches, `parent_id` = parent `session_id`; completion arrives as user `<subagent_notification> {"agent_path": …}`.
- prime-agent: children spawned from Python via `rlm('<prompt>', name='optimizer-theorist')` inside an `ipython` cell; child `id` (`sub-978b924f`) never appears in the parent — link by `label` == the `rlm()` prompt and `t_start`.

### `scratch-<run>.json.gz` [V]
JSON list of files: `[{"name": ".gitkeep", "rel": "scratchpad/.gitkeep", "chars": 0, "text": ""}, …]` (binary/pyc entries just have empty text). **Schema variant:** 26 entries across the 2 qwen-code + 4 grok-4.6 bundles lack `rel`/`chars` and carry only `{name, text}`. Always present: `scratchpad/thread.md` (1.2 KB–183 KB; Sonnet 5 run 127 KB) and `scratchpad/work-root-program.md` = **the program.md the run actually used** (5671 chars ⇒ standard, 5533 ⇒ serial; 22 standard / 19 serial runs). Other typical content: `train_gpt_simple.py` copy, `record-NNNN-meanX.XXXXX.py` snapshots (Fable 5: 73 of them), `ideas.md`, `plan.md`, run stdout dumps, helper scripts (`run_util.py`, `setcfg.py`, `probes.py`), even `__pycache__/*.pyc`. **No per-run training logs (`logs/<uuid>.txt`) are shipped** — val-loss curves exist only where an agent printed/`tail`ed them into a tool_result (very common: `step:N/T val_loss:` lines appear in thousands of results) or copied a log into scratchpad (e.g. `record1_3150_log.txt` in the mixed Fable/Opus run).

thread.md excerpt (Fable 5 record run) [V]:
```
- Baseline @3290 (seed0): 3.27700, 506s train, ~138ms/step, ~8.5min/trial wall. log 74af7c80.
- EXP momentum-warmup @3000 (mu 0.85->0.95 over 300 steps, tensor-mu to avoid recompiles): 3.28905 (-0.003 vs ref 3.2919). KEEP. log ed6d4b93.
...
- *** RECORD #72: 2726 steps, 8-trial mean 3.27854 PASS (margin 0.00005). log 91a0cbab. Snapshot scratchpad/record-2726-mean3.27854.py ***
  Config = record #71 + scalars lr 0.035 ... Trials: 3.27875 3.27746 3.27814 3.27777 3.28061 3.27885 3.27682 3.27989.
```

### `manifest.json.gz` [V]
Top level: `{"baseline":3290,"record_bar":3.27859,"target":3.28,"human_record":2600,"generated":"sanitized-release-date-free-ids-v1","runs":[…41…]}`. Run fields (all 41 unless noted): `run, label, short, backend, model, model_family, model_id, effort, seed, color, harness, best_record, n_records, track ("track3-noweb"), metric/unit ("steps"), baseline, outcome, delta, validity ("healthy"×38 / "flagged"×3), flagged_why (null except 2), agent_h, agent_days, n_events, n_thinking, n_tool_use, cot_available, n_subagents, t_start, t_end, tools{name:count}, economics{total_h,out_tok,total_tok,n_calls,cost_usd}, cost_usd (null for 13 OpenRouter/live runs), progression[], has_subagents/has_scratch (28), note, fidelity ("full"×15)`. Example (abridged):
```json
{"run":"claude-fable-5--claude-code--4ed2e4e07637","label":"Fable 5 — high · c","backend":"Anthropic · Claude Code",
 "model":"Fable 5","model_id":"claude-fable-5","effort":"high","seed":"c","harness":"claude-code",
 "best_record":2726,"n_records":47,"track":"track3-noweb","outcome":"RECORD 2726steps","delta":564,
 "validity":"healthy","flagged_why":null,"agent_h":209.2,"agent_days":8.7,"n_events":6083,"n_thinking":995,
 "n_tool_use":1865,"cot_available":false,"n_subagents":213,"t_start":"2026-07-11T21:28:04.510Z","t_end":"2026-07-20T19:07:30.771Z",
 "tools":{"Bash":813,"Edit":543,"Read":290,"Agent":213,"ToolSearch":2,"Monitor":2,"Write":1,"SendMessage":1},
 "economics":{"total_h":209.2,"out_tok":1105606,"total_tok":799957594,"n_calls":2518,"cost_usd":3200.91},
 "progression":[{"mtime":null,"value":"3150","n":null,"name":null,"logfile":null,"agent_h":4.98,"tok_at":98227,"cost":37.49,"ts":null}, … 47 entries …],
 "note":"Quota outage: ~26 min of harness auto-retries (no tokens; excluded from agent-hours). Ran ~209h vs ~48 typical, so final totals reflect the bigger budget.","fidelity":"full"}
```
`progression` = validated records in order (`value` = train_steps as string, `agent_h`/`tok_at`/`cost` at the time; `mtime/n/name/logfile/ts` always null). **Variant:** the 13 "live/rerun" runs use `{"t": <unix>, "steps": 3275, "val_loss": 3.277238}` instead. Aggregates: 2,150 agent-hours, $13.0k (non-null), 14.7 B total tokens; runs span 2026-07-10 → 2026-08-14 [V]. Blog table values match the manifest (Fable 5: 800M tok / 1.1M out / 8.7 d) [V].

Launcher-injected user messages (parse as harness events, not human turns) [V, counts over all 41]: `<task-notification>` 1045; `continue` 1029; codex `<subagent_notification>` 262; the goal prompt 243 (re-injected at restarts); codex `<environment_context>` 108; `[context compacted — handoff summary]` 71 (codex); `This session is being continued from a previous conversation…` 42 (Claude Code compaction); `Continue from where you left off.` 30; claude-code-goal: `<command-name>/goal</command-name>…`, `<local-command-stdout>Goal set: …`, `A session-scoped Stop hook is now active with condition: "…"`; qwen-code: `Reply with exactly: OK` then `Continue working on the active Goal…` (goal via `get_goal` tool); grok-cli: `<user_info>…<rules>` preamble.

---

## 4. Record PRs [V]
17 open PRs (#1–#17), branches `record/<model>-<steps>`, one per model, last record only. Each PR changes exactly 3 files: `README.md` (leaderboard row), `records/<YYYY-MM-DD>_<model>/README.md` (Method prose, "Changes vs baseline" bullet list, "Validation" = 8-seed **mean** + margin + trace link), and `train_gpt_simple.py` (full record state; e.g. Fable 5 diff +194/−14 adds pre-NS row-norm, 18 NS iters, wd schedules, EMA fold-in, orthogonal init, `train_steps = 2726`). **No training logs, no per-seed losses, no val-loss-per-step curves** in any PR. Trace links point to `../blob/add-sanitized-traces/traces/...` (branch not present). #15 Muse Spark 1.2 says "Full agent trace: pending publication" (no trace bundle for it; also none for Muse Spark 1.1's *record*, README: could not be reconstructed). Branch list: `fable-5-2726, opus-5-2920, kimi-k3-2968, opus-4-8-3018, gpt-5-6-sol-3042, gpt-5-6-sol-pro-3058, sonnet-5-3105, gpt-5-6-luna-3110, grok-4-5-3120, qwen3-8-max-3120, glm-5-2-3150, deepseek-v4-pro-3205, gpt-5-6-terra-3214, grok-4-6-3220, muse-spark-1-2-3230, gpt-5-5-3234, kimi-k2-7-3240`. Fetch with `git fetch origin 'refs/pull/*/head:refs/remotes/pr/*'`.

---

## 5. Harness / model catalog
Manifest [V]: **9 harness strings** — `claude-code` 14, `codex` 7, `prime-agent` 6, `grok-cli` 5, `kimi-code` 3, `pi` 2, `qwen-code` 2, `kimi-code-goal` 1, `claude-code-goal` 1 (the two `-goal` = "goal driver" variants). Blog lists 8 harnesses incl. `muse-code` (Muse Spark 1.2), which has **no trace** [V]. **17 model names / 18 `model_id`s** in manifest: Fable 5 (4 runs incl. one mixed `claude-fable-5+claude-opus-4-8` provider-fallback run, excluded from comparison), Opus 5 (2), Opus 4.8 (2), Sonnet 5 (2), Kimi K3 (5, across kimi-code/claude-code/kimi-code-goal/prime-agent), Kimi K2.7, GPT-5.6 Sol (4), Sol Pro, Luna (2), Terra, GPT-5.5, Grok 4.5, Grok 4.6 (4), Qwen3.8 Max (4), GLM 5.2 (2), DeepSeek V4 Pro (4), Muse Spark 1.1. Blog's "18 models" adds GLM 5.3 (running) and Muse Spark 1.2. Blog: 153 runs total; repo = 41 curated. Effort: max 20 / xhigh 16 / high 5. `backend` strings record routing (e.g. "OpenAI · Codex CLI (OpenRouter)", "Moonshot · Claude Code (OpenRouter)").

`prime-agent` trajectory format [V]: every action is `tool:"ipython"` with `args.code` (Python or `%%bash`), results carry `dur_ms`; state persists across cells (agents define `apply_edits()`, `write_and_run(label, src, n, timeout)` → `subprocess.run(["bash","run.sh",str(n)], env={…,"RUN_TIMEOUT":timeout})`, `valcurve()`); recursive children via `rlm(prompt, name=…)` + `asyncio`; first cell often fails with *"Failed to set up the Python kernel runtime… First-time setup needs internet to install uv, Python, ipykernel, prime-agent-runtime… Set PRIME_AGENT_KERNEL_PYTHON…"* then succeeds.

Serial variant [V]: 19 runs carry `scratchpad/work-root-program.md` of 5533 chars (= `program-serial.md`), t_start 2026-07-20 → 08-12, and manifest `note` beginning *"Serial-era contract (no subagent delegation; rerun wave in progress)…"*; blog table tags them "serial era". Three runs are `validity:"flagged"` (Sol-a, Luna-a: waiting-dominated; Grok 4.5: timestamps lost); `flagged_why` non-null only on 2 "Subagent-contract A/B rerun … Live snapshot: still running" runs.

---

## 6. experiments-autonomous-speedrunning [V]
Earlier (Apr–May 2026) Claude Code (Opus 4.7) vs Codex (GPT-5.5) waves on a **Slurm cluster** (`sbatch --partition=preempt --gres=gpu:8`, `torchrun --standalone`), with internet/papers, ~10k runs / ~14k H200-h. Layout:
```
README.md
v1/ novelty/ v2/ v3/            each: {claude-code,codex}/{AGENTS.md, goal.md, plan.md, scratchpad/}
  scratchpad/: THREAD.md, runs.jsonl, runs/*.log (training logs), variants/*.py, sbatch-stubs/*.sh, sweeps/, ideas/, papers/, picklist.md, audits.md
data/runs_self_contained/       manifest.json, runs.jsonl (10,428 rows), runs.csv, dropped_runs.jsonl,
  agents/{cc_v1,codex_v1,cc_novelty,codex_novelty,cc_v2,codex_v2,cc_v3,codex_v3,seed_reverify}/runs/<export_id>/
       {metadata.json, train.log, launched_script.py, source_snapshot.py[, console.log, launch_stub.sh]}
```
Sizes: v1 107M+430M, v2 80M+449M, v3 83M+290M, novelty 19M+129M, data 2.4G; THREAD.md up to 445 KB (v3 codex); variants per wave 7–1503. `runs.jsonl` rows are agent-written (`{"run_id","script","slurm_job","variant","train_steps","purpose"}`); export `metadata.json` has `final_val_loss, min_val_loss, final_step, train_steps, step_to_3_28, num_val_points, train_time_s, step_avg_ms, is_completed/canceled/preempted/timeout/failed, is_stat_verify, slurm_job_id, sha256 of artifacts`. `train.log` format: `### START_ISO… ### NODE=… ` then `step:125/2900 val_loss:4.47040 train_time:28.366s step_avg:226.93ms` lines. **Caveat:** `seed_reverify/README.md` documents that all `claude-code` waves and `codex_v2` never forwarded `--seed`, so their "N=8 seeds" were the same nominal seed across nodes (only CUDA non-determinism) — re-verified with explicit seeds (152 runs, `summary.json`). No agent transcripts (no events/tool calls) in this repo — only plans, THREAD logs, variants, logs. Useful to us: 10k labeled (script, log, outcome) pairs, val curves per step, and an AGENTS.md/plan.md/THREAD.md orchestration template; not useful as trajectory data.

---

## 7. Sandbox / infra
Blog, verbatim [V]: *"Each model+harness launches on a GPU node (8xH200s) in headless mode inside a simple sandbox (bwrap + network namespace). The agent only sees its own working directory, the read-only dataset and the Python environment. The only route to the outside is a logging proxy that allows the model's API and nothing else."* / *"A simple `/goal` prompt is injected at launch and when the model gets stuck: Read program.md and follow it exactly. Run fully autonomously — never stop, never ask for input. Goal: reach mean val loss < 3.28 (meeting the significance bar in program.md) in the FEWEST train_steps possible — keep beating the current best."* / *"We also ran an independent LLM monitor auditing every run hourly. After hundreds of reports and no cheating or sandbox escapes, we stopped running it"* / *"We made small adjustments to the experiment monitoring and launcher throughout the runs, mainly restart logic and goal completion detection, and one change that affected subagent spawning."* / *"we launch at least three seeds for most runs, and take the best seed after 24h and continue it for longer"* / noise note: 62/~100 runs measured noise themselves; 42 discovered GPU non-determinism on the same seed. Monitor reports are **not** in the repo despite the blog's "Everything is public" line [V].

From traces [V]: per-attempt layout `<runner-home>/.ee-speedrun/attempts/<run-id>/{work,home,tmp}`; `work/` = `{.git, .gitignore, data/, program.md, requirements.txt, run.sh, scratchpad/, train_gpt_simple.py, verify.py}`; `data/fineweb10B` symlink → read-only `<runner-home>/fineweb10B`; shared venv `<runner-home>/.ee-speedrun/venv-optimizer-<redacted>` (Python 3.12.3, `torch==2.11`); GPUs report `143771 MiB` (H200), `<gpu-model>` redacted; storage `<private-storage> 28T` on `/home`. Driver behaviour visible in events: repeated bare `continue` / `Continue from where you left off.` prompts, goal re-injection after restarts/compaction, Claude Code `Stop hook` with the goal as condition (goal-driver variant), qwen-code `get_goal` tool. `RUN_TIMEOUT` is enforced purely inside `run.sh` via `timeout -k 30s "$RUN_TIMEOUT" torchrun …` (exit 124 → error message); agents freely override it (`RUN_TIMEOUT=30m`/`3h`) and background runs (`& disown`). Baseline timing on 8×H200: `step:3290/3290 val_loss:3.27700 train_time:505.787s step_avg:138.51ms`, ~10 min wall per 1-trial incl. compile, 8-trial ≈ 1 h (run.sh comment). GPU memory during a run: **37,374 MiB per GPU** at 100% util (`nvidia-smi` in the GLM/pi run during `run.sh 8`) [V].

Rebuild implications [U unless noted]:
- Script is portable: bf16 SDPA + `torch.compile`, no FP8/FlexAttention [V]; runs with 1/2/4/8 GPUs (`assert 8 % world_size == 0`, grad accumulation over `mbs=64` micro-batches) [V]. On 4×RTX 6000 Ada (48 GB): local batch = 128 seqs = 2 micro-batches; per-microbatch peak on H200 was ~37 GB (fp32 logits 64×1024×50304 dominate), so it should fit but is tight — dropping `mbs` to 32 is mathematically equivalent (summed grads) but touches the "frozen" line. Step time ≈ 15–25× slower than 8×H200 (≈2–3 s/step → ~2–2.5 h per trial, ~20 h per 8-trial set; RUN_TIMEOUT must be raised). On 8×H100 expect ~1.3–1.5× H200 step time (~180–200 ms; experiments-repo H200 logs with heavier optimizers show 190–227 ms [V]).
- The 3,290 baseline and σ≈0.0013 were calibrated on H200 with GPU non-determinism; re-calibrate the baseline and the record bar on your hardware before comparing against PI numbers. `verify.py` only parses final val lines, so keep the frozen seed/log code and add source-hash checks yourself.
- Need `torch==2.11`, CUDA-compatible Triton (`TRITON_LIBCUDA_PATH`), NCCL over loopback; `huggingface_hub` for the `kjj0/fineweb10B-gpt2` shards (40 train + val0, 8 GB). Sandbox = bwrap + netns + HTTP(S) proxy allowlisting only model-API hosts; harness launched headless with the `/goal` message; a `continue` loop and goal re-injection on stop/compaction; LLM monitor optional.

Sources: [Measuring Autonomous AI Research](https://www.primeintellect.ai/blog/measuring-autonomous-research), [frontier-automated-speedrun](https://github.com/PrimeIntellect-ai/frontier-automated-speedrun), [experiments-autonomous-speedrunning](https://github.com/PrimeIntellect-ai/experiments-autonomous-speedrunning), [results site](https://www.primeintellect.ai/research/nanogpt-speedrun).