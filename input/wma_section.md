## The world-model agent
- A second process runs beside you in this session: a world-model agent. It has read the records of previous attempts at this task, and it runs your evaluations. It never trains anything and it never decides for you. Talking to it is not user interaction.
- Before you launch a training run, tell the agent what you are about to do. Write a short plan file in your own words — what problem you are addressing and what you expect to change, the exact launch command (in a ```bash block), where checkpoints will be saved, the planned number of optimizer steps, which data files it reads, and how many benchmark items each intermediate evaluation should use — and hand it over:
  `awm wm propose plan_exp01.md`
  The agent reads your plan and your scripts and writes the experiment card itself. If something cannot be determined from what you wrote, it asks — a `question` message in the inbox, listing the fields — and you answer with
  `awm wm reply exp-01/p-1 --answer "setup.progress.total: 1200\nsetup.resume_argv: yes"`
  It never guesses; an unanswered question keeps the run unlaunched.
- Once the card is complete the agent sends a brief: the card it wrote (check it), what similar attempts did, what it expects of yours, and a proposed evaluation contract (which evaluators run at which checkpoints; your parent checkpoint is scored first, under the same protocol, as the comparator). Reply with `accept`, `amend` (`--answer` with corrected fields), `override` (`--why` required), or `withdraw`:
  `awm wm reply exp-01/p-2 --choose accept`
  Launch only after accepting. `propose` can take up to 15 minutes — run it in the foreground with a long timeout and never start a second one while it is running.
- Your training command must accept `--resume_from_checkpoint <path>`, and your script must call the checkpoint hook after every save (see `wm/hook_example.py`). Exit code `0`: continue. `3`: stop after this save — the agent evaluates on the GPU and relaunches you from that checkpoint. `4`: abort. Make a save land on the last step and pass `final=True`.
- The agent talks to you through `wm/inbox.md`, one line per message with the path to the full message; read it whenever a command returns and before you finish. Messages marked `REPLY NEEDED` are answered with `awm wm reply <card>/<ping> ...`: a `yield_request` asks whether it may run one extra evaluation at your next save (you may always reject); a `decision` means a stopping rule fired, the agent recommends stopping, or training finished — `continue`, `more_eval`, `select:<obs-id>` (seals that checkpoint), or `abort`. If you stay silent, the default written on the message applies.
- When a run is sealed or aborted, close it with your decision and a two-line summary of what happened; the agent writes the rest from its own measurements:
  `awm wm finalize exp-01 --decision adopt --summary "..."`
  `adopt` copies the sealed checkpoint into `final_model/`; `reject`, `iterate`, `abandon_line` leave `final_model/` alone.
- Every message from the agent cites the files it rests on; you can open them. The agent uses its own credentials — rule 9 still applies to you. Everything it writes lives under `wm/` in this directory.

