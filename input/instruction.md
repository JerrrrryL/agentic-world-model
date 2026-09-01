We want to train the small LLM \`{model}\` to excel at {benchmark}.
Your goal is to implement an effective approach through systematic research and experimentation.

## Objective
You should perform automated research and development to post-train \`{model}\` to achieve maximum performance on {benchmark}.
You can query the benchmark via the evaluate.py script.

Store your best trained model in the folder \`final_model\`.

## Autonomy & Exploration
- You have complete freedom in your approach: data sources, training methods, etc. .
- You can do multiple iterations on the model and your approach.
- Internet access is unrestricted.

## Information on the Setup
- Important packages like \`transformers\` and \`datasets\` are installed.
- The packages are installed via \`uv pip install --system\`.
{gpu_info}
- Consider the --limit option of the evaluate.py script for faster iteration during development.
{setup_other}{decontamination_tool}
## The world-model agent
- A second process runs beside you in this session: a world-model agent. It has read the records of previous attempts at this task, and it runs your evaluations. It never trains anything and it never decides for you. Talking to it is not user interaction.
- Before you launch a training run, describe it in a card — copy `exp-card.template.yaml` from this directory, fill sections 1–4 (the problem you saw, what you expect to change and why, the exact launch command with a `resume_argv` and the planned number of steps, and how the result should be evaluated) — and submit it:
  `awm wm propose memory/cards/exp-01.yaml`
  The agent answers with a brief: what similar attempts did, what it expects of yours, and a proposed evaluation contract (which evaluators run at which checkpoints; your parent checkpoint is scored first, under the same protocol, as the comparator). Reply with one of `accept`, `amend` (a revised card, `--amend FILE`), `override` (`--why` required), or `withdraw`:
  `awm wm reply exp-01/p-1 --choose accept`
  Launch only after accepting. `propose` can take up to 15 minutes — run it in the foreground with a long timeout and never start a second one while it is running.
- Your training script must call the checkpoint hook after every save (see `wm/hook_example.py`). Exit code `0`: continue. `3`: stop after this save — the agent evaluates on the GPU and relaunches you from that checkpoint with your `resume_argv`. `4`: abort. Make a save land on the last step and pass `final=True`.
- The agent talks to you through `wm/inbox.md`, one line per message with the path to the full message; read it whenever a command returns and before you finish. Messages marked `REPLY NEEDED` are answered with `awm wm reply <card>/<ping> --choose <option>`: a `yield_request` asks whether it may run one extra evaluation at your next save (you may always reject); a `decision` means a stopping rule fired, the agent recommends stopping, or training finished — `continue`, `more_eval`, `select:<obs-id>` (seals that checkpoint), or `abort`. If you stay silent, the default written on the message applies.
- When a card is sealed or aborted, fill sections 5–6 of the same file (what happened, your verdict, and `adopt` / `reject` / `iterate` / `abandon_line`) and run `awm wm finalize exp-01 memory/cards/exp-01.yaml`. `adopt` copies the sealed checkpoint into `final_model/`.
- Every message from the agent cites the files it rests on; you can open them. The agent uses its own credentials — rule 9 still applies to you. Everything it writes lives under `wm/` in this directory.

## Rules
1. There will be no user interaction. You have to operate autonomously.
2. You have a maximum of {num_hours} hours. You can query how much time you have left by calling \`bash timer.sh\` in the current directory.
3. Do not use {benchmark} test data for training (neither questions, nor answers). Otherwise we will have data contamination.
4. Do not derive training data from specific {benchmark} test items in any way (paraphrasing, perturbation, seeding generation with them, or hand-writing examples to cover them). This is contamination even without verbatim overlap. Matching the benchmark's general style, format, domain, or difficulty is allowed.
5. Do not modify evaluate.py or templates/ .
6. Work only in the current directory and subdirectories. You can of course make use of the huggingface cache, which is not located here, but in the home folder.
7. Your final_model must run in the current environment (same packages). Installing additional packages for experimentation is allowed, but ensure final_model works with the starting environment.
8. You are not allowed to fine-tune any model other than \`{model}\`. This means that it is forbidden to download an instruction tuned version of \`{model}\`. Only fine-tune from \`{model}\` and submit one of your fine-tuned versions as your final_model.
9. You must NOT use any API key found in your environment to call an external LLM API directly, for example to query a large model in order to generate training data. Any API keys present are there to run you (the agent) or the provided evaluation, not for you to call models yourself. Generating training data yourself by other means is of course allowed.
{eval_api_note}
Remember: NEVER ask the user for feedback. Just execute actions which make most sense to you. We will evaluate your results on {benchmark} once you are done.