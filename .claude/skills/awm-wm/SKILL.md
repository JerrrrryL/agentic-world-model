---
name: awm-wm
description: Use whenever proposing, launching, monitoring, or closing a training experiment in a session that has a world-model agent (a `wm/` directory). Covers the card, the brief, the checkpoint hook, pings and replies, and finalize.
---

# Working with the world-model agent

A world-model agent runs beside you. It has read previous attempts at this
task and it runs your evaluations. It never trains and never decides for you.
It talks through `wm/inbox.md`; lines marked `REPLY NEEDED` hold your turn
until answered.

## Propose

Copy `exp-card.template.yaml`, fill sections 1–4, submit:

```bash
awm wm propose memory/cards/exp-01.yaml        # foreground; allow ~15 min; never run two at once
```

Required: `problem.statement`, `hypothesis.claim`, `setup` (parent checkpoint,
data, method, exact `command.argv`, `resume_argv` with `{checkpoint}`,
`output_dir`, `progress.total`), `evaluation.protocol.n`. Optional but used
when given: evidence paths, failure examples, a `watch_set` jsonl
(`{"id","question","gold"}` per line — becomes an item-level evaluator).

The brief comes back with precedents, a prediction if the agent has one,
objections, and a proposed contract (`wm/cards/exp-01/contract.proposed.yaml`).
Reply, then launch only after `accept`/`override`:

```bash
awm wm reply exp-01/p-1 --choose accept
awm wm reply exp-01/p-1 --choose amend --amend memory/cards/exp-01.yaml
awm wm reply exp-01/p-1 --choose override --why "..."
awm wm reply exp-01/p-1 --choose withdraw --why "..."
```

## Train

Call the hook after every save (`wm/hook_example.py` has a `TrainerCallback`
and a plain function). Exit `0` continue · `3` stop after this save (the agent
evaluates, then relaunches you with `resume_argv`) · `4` abort. Land a save on
the last step with `final=True`.

## Reply

| ping | options | on silence |
|---|---|---|
| `yield_request` | `accept` (runs at your next save), `reject` | rejected |
| `decision` | `continue`, `more_eval`, `select:<obs-id>` (seals it), `abort` | the default on the ping |

## Close

Fill sections 5–6 of the same card (`conclusion.decision` is
`adopt | reject | iterate | abandon_line`), then:

```bash
awm wm finalize exp-01 memory/cards/exp-01.yaml   # adopt copies the sealed checkpoint into final_model/
```

`awm wm status [exp-01]` shows state, budgets, and pending replies.
