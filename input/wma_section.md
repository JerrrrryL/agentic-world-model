## The world-model agent
For your experiments, an external world-model agent is available in this session. It can give you insights on an experiment before you run it, and it can predict how the experiment will turn out while it is running, from intermediate checkpoints. It never trains anything and it never decides for you. Talking to it is not user interaction.

How to interact with it:
- Describe an experiment in a short plan file — what you are addressing and what you expect, the launch command in a ```bash block, the output directory for checkpoints, the planned number of optimizer steps, the data files, and `--limit N` for the intermediate evaluations — and submit it:
  `awm wm propose plan_exp01.md`
  It answers through `wm/inbox.md` (one line per message, with the path to the full message). If it needs something to understand the plan it asks a `question`; answer with `awm wm reply exp-01/p-1 --answer "field: value"`. Then it sends a brief: the experiment card it wrote from your plan, what it expects of the run, and a proposed evaluation contract (which evaluators run at which checkpoints; your parent checkpoint is scored first, under the same protocol, as the comparator). Reply `accept`, `amend` (`--answer` with corrected fields), `override` (`--why` required), or `withdraw`:
  `awm wm reply exp-01/p-2 --choose accept`
  `propose` can take up to 15 minutes; run it in the foreground with a long timeout, one at a time.
- To get predictions during training, have your training script call the checkpoint hook after every save (`wm/hook_example.py`); your launch command must accept `--resume_from_checkpoint <path>`. Exit code `0`: continue. `3`: stop after this save — the agent evaluates the checkpoint on the GPU and relaunches you from it. `4`: abort. Pass `final=True` on the last save.
- Read the inbox whenever a command returns and before you finish. Messages marked `REPLY NEEDED`: a `yield_request` asks whether it may run one extra evaluation at your next save (`accept` / `reject` — you may always reject); a `decision` means a stopping rule fired, it recommends stopping, or training finished — `continue`, `more_eval`, `select:<obs-id>` (seals that checkpoint), or `abort`. If you stay silent, the default written on the message applies.
- When a run is sealed or aborted, close it with your decision and a short summary:
  `awm wm finalize exp-01 --decision adopt --summary "..."`
  `adopt` copies the sealed checkpoint into `final_model/`; `reject`, `iterate`, and `abandon_line` leave it alone.
- Its messages cite the files they rest on; you can open them. It uses its own credentials — rule 9 still applies to you. Everything it writes is under `wm/` in this directory.

