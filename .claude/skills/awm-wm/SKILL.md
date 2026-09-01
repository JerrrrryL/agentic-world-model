---
name: awm-wm
description: Use whenever proposing, launching, monitoring, or closing a training experiment in a session that has a world-model agent (a `wm/` directory). Covers the plan, the agent's questions, the brief, the checkpoint hook, pings and replies, and finalize.
---

# Working with the world-model agent

A world-model agent runs beside you. It has read previous attempts at this
task and it runs your evaluations. It never trains and never decides for you.
It talks through `wm/inbox.md`; lines marked `REPLY NEEDED` hold your turn
until answered. You never write experiment cards — the agent does, from your
plan.

## Propose

Write a short plan in your own words, then hand it over:

```bash
awm wm propose plan_exp01.md        # foreground; allow ~15 min; never run two at once
```

A plan the agent can turn into a card without asking says: the problem you
are addressing and what you expect to change; the exact launch command in a
```bash block (it must accept `--resume_from_checkpoint <path>`); the output
directory for checkpoints; the planned number of optimizer steps; the data
files it reads; and `--limit N` for the intermediate evaluations. Anything
missing comes back as a `question`:

```bash
awm wm reply exp-01/p-1 --answer "setup.progress.total: 1200
setup.resume_argv: yes
evaluation.protocol.n: 150"
```

Then the **brief**: the card it wrote (`wm/cards/exp-01/card.yaml` — read
it), precedents, a prediction if it has one, objections, and the proposed
contract. Launch only after `accept`/`override`:

```bash
awm wm reply exp-01/p-2 --choose accept
awm wm reply exp-01/p-2 --choose amend --answer "setup.output_dir: /home/ben/task/runs/exp01"
awm wm reply exp-01/p-2 --choose override --why "..."
awm wm reply exp-01/p-2 --choose withdraw --why "..."
```

## Train

Call the hook after every save (`wm/hook_example.py` has a `TrainerCallback`
and a plain function). Exit `0` continue · `3` stop after this save (the agent
evaluates, then relaunches you with the resume command) · `4` abort. Land a
save on the last step with `final=True`.

## Reply

| ping | options | on silence |
|---|---|---|
| `question` | `--answer "field: value"` lines, or `--answer-file answers.yaml` | run stays unlaunched |
| `yield_request` | `accept` (runs at your next save), `reject` | rejected |
| `decision` | `continue`, `more_eval`, `select:<obs-id>` (seals it), `abort` | the default on the ping |

## Close

```bash
awm wm finalize exp-01 --decision adopt --summary "dev150 +0.12 vs parent at step 750; watch set 17/41 fixed, 0 regressions"
```

`--decision` is `adopt | reject | iterate | abandon_line`; `adopt` copies the
sealed checkpoint into `final_model/`. The agent writes the measurements from
its own observations. `awm wm status [exp-01]` shows state, budgets, and
pending replies.
