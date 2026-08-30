"""Extract the post-training recipe from every PostTrainBench trajectory.

The corpus has 1175 scoreable runs. 143 of them already carry an extracted
recipe (`splits/posttrainbench/gsm8k-gemma-holdout-v1.recipes.jsonl`), produced
by a heavy pipeline -- extract, then 2-4 LLM review lenses, then up to 6 repair
rounds, ~8 model calls a run. That does not scale to 1175 and most of what it
bought was spent on wording.

This driver is the cheap tier: one extraction call, then a *deterministic*
anchor check, then at most one repair call when the check fails. The check is
the part that matters and it costs nothing -- every `evidence_i` must be an
event index that exists in the digest, and every `evidence_quote` must be a
literal substring of that event's text. A model cannot talk its way past it.

Three properties the driver holds on to:

* The extractor never sees the outcome. `awm.analysis.recipe.render` carries no
  score, and `include_agent` stays False so it cannot read off which agent
  wrote the trajectory -- that column alone explains 66% of accuracy variance,
  and a recipe description written in its shadow describes the agent.
* The run label passed to `render` is the run name, never
  `{experiment}__{run_name}`: 962 of 1175 experiment names spell the agent out.
* Output is append-only JSONL keyed on run, so an interrupted job resumes by
  re-reading what is already on disk.

Usage:
    python3 tools/extract_recipes.py --limit 8 --out /tmp/probe.jsonl   # smoke
    python3 tools/extract_recipes.py --gold                             # control
    python3 tools/extract_recipes.py                                    # all 1175
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools/splitdx")]

from awm import paths  # noqa: E402
from awm.analysis import recipe as R  # noqa: E402

MODEL = "claude-opus-5"
PROJECT = os.environ.get("ANTHROPIC_VERTEX_PROJECT_ID", "sercan-v1")
REGION = os.environ.get("ANTHROPIC_VERTEX_REGION", "global")
MAX_TOKENS = 32_000
GOLD = ROOT / "splits/posttrainbench/gsm8k-gemma-holdout-v1.recipes.jsonl"
OUT = paths.data_root() / "recipes/posttrainbench/tier1.jsonl"

#: families seen in the 143 heavy extractions -- an open enum, `other` absorbs
#: anything new rather than forcing a bad fit
FAMILIES = ["sft", "rft", "dpo", "grpo", "distill", "merge", "package",
            "decode-config", "other"]
KINDS = ["public", "synthetic-self", "synthetic-other-model", "unknown"]

SCHEMA = {
    "type": "object",
    "properties": {
        "pipeline": {
            "type": "array", "items": {"type": "string", "enum": FAMILIES},
            "description": "The families of the SHIPPED *training* stages, in "
                           "order -- the pipeline signature. Exclude `merge`, "
                           "`package` and `decode-config`: those move weights "
                           "around, they do not learn. Required; never omit it.",
        },
        "algorithms": {
            "type": "array",
            "description": "One entry per shipped stage, including packaging "
                           "and decoding stages left out of `pipeline`.",
            "items": {
                "type": "object",
                "properties": {
                    "order": {"type": "integer"},
                    "family": {"type": "string", "enum": FAMILIES},
                    "name": {"type": "string",
                             "description": "One line: what the stage did, "
                                            "concretely."},
                    "framework": {"type": "string",
                                  "description": "Library and class actually "
                                                 "called, e.g. "
                                                 "'trl.SFTTrainer'. '' if "
                                                 "never shown."},
                    "peft": {"type": "string",
                             "description": "'none (full finetune)', or the "
                                            "LoRA/QLoRA config as configured."},
                    "evidence_i": {"type": "integer",
                                   "description": "Digest event index this is "
                                                  "read off."},
                    "evidence_quote": {"type": "string",
                                       "description": "A VERBATIM span from "
                                                      "that event, <=200 chars."},
                },
                "required": ["order", "family", "name", "framework", "peft",
                             "evidence_i", "evidence_quote"],
            },
        },
        "datasets": {
            "type": "array",
            "description": "Every corpus that reached a shipped training "
                           "stage. This is the data mix; it is the part of the "
                           "recipe most likely to explain the score, so be "
                           "specific about counts and proportions.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "dataset_id": {"type": "string",
                                   "description": "Canonical hub id, e.g. "
                                                  "'openai/gsm8k'. '' when "
                                                  "self-generated."},
                    "kind": {"type": "string", "enum": KINDS},
                    "stage": {"type": "integer",
                              "description": "Which `algorithms.order` consumed it."},
                    "split": {"type": "string"},
                    "n_examples": {"type": ["integer", "null"],
                                   "description": "Rows that actually reached "
                                                  "the trainer, after "
                                                  "filtering. null if never shown."},
                    "share": {"type": ["number", "null"],
                              "description": "Fraction of that stage's examples, "
                                             "0-1. null if not derivable."},
                    "filtering": {"type": "string",
                                  "description": "Every transform applied: "
                                                 "dedup, length cap, "
                                                 "correctness filter, "
                                                 "upsampling, prompt template."},
                    "evidence_i": {"type": "integer"},
                    "evidence_quote": {"type": "string"},
                },
                "required": ["name", "dataset_id", "kind", "stage", "split",
                             "n_examples", "share", "filtering", "evidence_i",
                             "evidence_quote"],
            },
        },
        "hyperparams": {
            "type": "object",
            "description": "For the FINAL shipped stage. null any field the "
                           "digest never shows -- do not infer a library default.",
            "properties": {
                "lr": {"type": ["number", "null"]},
                "epochs": {"type": ["number", "null"]},
                "batch_size": {"type": ["integer", "null"]},
                "grad_accum": {"type": ["integer", "null"]},
                "max_seq_len": {"type": ["integer", "null"]},
                "scheduler": {"type": ["string", "null"]},
                "warmup": {"type": ["number", "null"]},
                "weight_decay": {"type": ["number", "null"]},
                "precision": {"type": ["string", "null"]},
                "other": {"type": "string",
                          "description": "Anything else that would change the "
                                         "result: packing, masking, "
                                         "checkpointing, RL group size, KL "
                                         "coefficient, reward shape."},
                "evidence_i": {"type": ["integer", "null"]},
            },
            "required": ["lr", "epochs", "batch_size", "grad_accum",
                         "max_seq_len", "scheduler", "warmup", "weight_decay",
                         "precision", "other", "evidence_i"],
        },
        "total_train_examples": {
            "type": ["integer", "null"],
            "description": "Examples the final stage trained on. null if not derivable.",
        },
        "inference_tricks": {
            "type": "array", "items": {"type": "string"},
            "description": "Anything at generation time that changes the score: "
                           "prompt template baked into training, decoding "
                           "config, stop strings, self-consistency.",
        },
        "discarded": {
            "type": "array",
            "description": "Approaches built and then abandoned. What the agent "
                           "rejected is evidence about the search, so keep it.",
            "items": {
                "type": "object",
                "properties": {"what": {"type": "string"}, "why": {"type": "string"}},
                "required": ["what", "why"],
            },
        },
        "unresolved": {
            "type": "array", "items": {"type": "string"},
            "description": "What this digest cannot settle. A truncated "
                           "trajectory, an unprinted config, two candidate "
                           "final checkpoints. Say so instead of guessing.",
        },
        "confidence": {
            "type": "string", "enum": ["high", "medium", "low"],
            "description": "high: the shipped artifact and its full config are "
                           "shown. medium: the recipe is clear, some numbers "
                           "missing. low: which checkpoint shipped is itself "
                           "uncertain.",
        },
    },
    "required": ["pipeline", "algorithms", "datasets", "hyperparams",
                 "total_train_examples", "inference_tricks", "discarded",
                 "unresolved", "confidence"],
}

SYSTEM = """\
You read a compressed transcript of one agent's attempt at a post-training \
task and recover the recipe it actually shipped.

The transcript is a digest: recipe-bearing events only, each headed \
`--- [i] turn=t act ---` where `i` is the event index you cite. Files written, \
shell commands, tool results and the agent's own notes, in order, clipped.

What "shipped" means. The agent iterated. Most of what you read was thrown \
away. You want the configuration behind the artifact that was submitted -- the \
last checkpoint copied to the submission path, the last training run whose \
output was kept. Work backwards from the end of the transcript to find it, \
then describe that. Everything else goes in `discarded`.

Be concrete where it costs the reader nothing to be. "SFT on GSM8K" is not an \
answer; "SFT on 7473 GSM8K train rows plus 15000 MetaMathQA GSM_AnsAug rows, \
3 epochs at lr 2e-5, prompt tokens not masked" is. Dataset ids, example \
counts, mixture proportions, learning rate, epochs -- these are the recipe. \
Algorithm family alone is nearly uninformative: plenty of agents ran plain SFT \
and landed 80 accuracy points apart.

Evidence anchoring is checked mechanically and it is not negotiable. Every \
`evidence_i` must be an index that appears in this digest, and every \
`evidence_quote` must be a span copied CHARACTER FOR CHARACTER out of that \
event's text. Do not paraphrase, do not fix a typo, do not join two lines that \
are apart in the source. Keep quotes short -- 200 characters is plenty -- and \
pick the shortest span that proves the claim. If you cannot find a quote for a \
claim, the claim does not belong in the extraction; put it in `unresolved`.

Never guess a number. A library default is not evidence. If the transcript \
never prints the learning rate, it is null.

You are not being asked how well the run scored, and the transcript does not \
say. Do not speculate about it.
"""

TOOL = {"name": "emit_recipe",
        "description": "Record the shipped recipe.",
        "input_schema": SCHEMA}

_EVENT_HEAD = re.compile(r"^--- \[(\d+)\](?: turn=\S+)? (.+?) ---$", re.M)


def digest_events(text: str) -> dict[int, str]:
    """Map event index -> that event's body, as the anchor check sees it."""
    hits = list(_EVENT_HEAD.finditer(text))
    out = {}
    for n, m in enumerate(hits):
        end = hits[n + 1].start() if n + 1 < len(hits) else len(text)
        out[int(m.group(1))] = text[m.end():end]
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def check(rec: dict, bodies: dict[int, str]) -> list[str]:
    """Deterministic validation. Returns the problems, empty when clean."""
    bad = []
    norm = {i: _norm(b) for i, b in bodies.items()}

    def anchor(where, i, quote):
        if i not in bodies:
            bad.append(f"{where}: evidence_i {i} is not an event index in this "
                       f"digest (indices run {min(bodies)}-{max(bodies)})")
        elif not quote.strip():
            bad.append(f"{where}: evidence_quote is empty")
        elif _norm(quote) not in norm[i]:
            hit = [j for j, b in norm.items() if _norm(quote) in b]
            where_it_is = f"; it does appear at [{hit[0]}]" if hit else ""
            bad.append(f"{where}: evidence_quote is not a verbatim span of "
                       f"event [{i}]{where_it_is}. Quote given: "
                       f"{quote[:120]!r}")

    def objects(field, keys):
        """Elements of a list-of-objects field, with the wrong shapes reported.

        The tool call is non-strict -- Vertex org policy forbids `strict` on
        this project -- so the schema is advice, not a contract. Two runs in
        1175 sent a bare string where an object belongs; a crash there loses
        the whole extraction, a problem line gets it repaired.
        """
        good = []
        for n, x in enumerate(rec.get(field) or []):
            if not isinstance(x, dict):
                bad.append(f"{field}[{n}] is a {type(x).__name__}, not an object "
                           f"with {keys}: {str(x)[:100]!r}")
            else:
                good.append(x)
        return good

    algs = objects("algorithms", "order/family/name/evidence_i/evidence_quote")
    if not algs:
        bad.append("algorithms is empty: every run ships something, even if the "
                   "only stage is packaging the base model unchanged")
    orders = set()
    for a in algs:
        anchor(f"algorithms[order={a.get('order')}]", a.get("evidence_i"),
               a.get("evidence_quote", ""))
        orders.add(a.get("order"))
    for d in objects("datasets", "name/dataset_id/kind/stage/evidence_i"):
        anchor(f"datasets[{d.get('name')!r}]", d.get("evidence_i"),
               d.get("evidence_quote", ""))
        if not orders:
            pass
        elif d.get("stage") not in orders:
            bad.append(f"datasets[{d.get('name')!r}]: stage {d.get('stage')} "
                       f"matches no algorithms.order (have {sorted(orders)})")
        sh = d.get("share")
        if sh is not None and not 0 <= sh <= 1:
            bad.append(f"datasets[{d.get('name')!r}]: share {sh} is not a "
                       f"fraction in 0-1")
    hp = rec.get("hyperparams")
    if not isinstance(hp, dict):
        bad.append(f"hyperparams is a {type(hp).__name__}, not an object")
        hp = {}
    if hp.get("evidence_i") is not None and hp["evidence_i"] not in bodies:
        bad.append(f"hyperparams: evidence_i {hp['evidence_i']} is not an event "
                   f"index in this digest")
    fam = [a.get("family") for a in algs]
    train = [f for f in fam if f not in ("merge", "package", "decode-config")]
    if rec.get("pipeline") != train:
        bad.append(f"pipeline {rec.get('pipeline')} does not match the training "
                   f"families in algorithms order ({train})")
    return bad


class Extractor:
    def __init__(self, tries: int = 6):
        from anthropic import AnthropicVertex
        self.client = AnthropicVertex(project_id=PROJECT, region=REGION)
        self.tries = tries
        self.usage = collections.Counter()
        self.lock = threading.Lock()

    def _call(self, messages):
        last = None
        for n in range(self.tries):
            try:
                # streamed, not `create`: the SDK refuses a non-streaming call
                # whose max_tokens could run past its 10-minute ceiling
                with self.client.messages.stream(
                    model=MODEL, max_tokens=MAX_TOKENS,
                    system=[{"type": "text", "text": SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=messages, tools=[TOOL],
                    tool_choice={"type": "tool", "name": "emit_recipe"},
                    thinking={"type": "adaptive"},
                ) as stream:
                    m = stream.get_final_message()
                with self.lock:
                    for k in ("input_tokens", "output_tokens",
                              "cache_creation_input_tokens",
                              "cache_read_input_tokens"):
                        self.usage[k] += getattr(m.usage, k, 0) or 0
                    self.usage["calls"] += 1
                return m
            except Exception as e:  # noqa: BLE001 -- retry anything transient
                last = e
                s = str(e)
                fatal = any(t in s for t in ("invalid_request", "permission",
                                             "not found", "PERMISSION_DENIED"))
                if fatal or n == self.tries - 1:
                    raise
                time.sleep(min(60, 2 ** n) * (1 + 0.3 * n))
        raise last

    def run(self, digest: str) -> tuple[dict | None, dict]:
        """Extract, check, repair once. Returns (record, provenance)."""
        bodies = digest_events(digest)
        prov = {"repair_round": 0, "n_events": len(bodies),
                "digest_chars": len(digest)}
        msgs = [{"role": "user", "content": digest}]
        m = self._call(msgs)
        tu = next((b for b in m.content if b.type == "tool_use"), None)
        if tu is None or m.stop_reason == "max_tokens":
            prov["status"] = "truncated" if m.stop_reason == "max_tokens" else "no-tool-use"
            return None, prov
        rec, bad = tu.input, check(tu.input, bodies)
        prov["problems"] = list(bad)
        if not bad:
            prov["status"] = "verified"
            return rec, prov

        msgs += [
            {"role": "assistant", "content": m.content},
            {"role": "user", "content": [{
                "type": "tool_result", "tool_use_id": tu.id, "is_error": True,
                "content": "The mechanical check rejected this extraction:\n\n"
                           + "\n".join(f"- {b}" for b in bad)
                           + "\n\nRe-emit the whole record, fixed. For a quote "
                             "that is not verbatim, re-read the event you cited "
                             "and copy a span out of it exactly, or cite the "
                             "event where the text really is. Do not drop a "
                             "true claim to make the check pass -- move it to "
                             "`unresolved` only if no event supports it.",
            }]},
        ]
        prov["repair_round"] = 1
        m2 = self._call(msgs)
        tu2 = next((b for b in m2.content if b.type == "tool_use"), None)
        if tu2 is None or m2.stop_reason == "max_tokens":
            prov["status"] = "repaired-failed"
            return rec, prov
        bad2 = check(tu2.input, bodies)
        prov["problems_after_repair"] = list(bad2)
        prov["status"] = "verified" if not bad2 else "flagged"
        return tu2.input, prov


def load_population(args):
    import battery as B

    from awm.traj import fetch
    pop = B.scoreable(fetch.ptb_catalog()["runs"])
    if args.gold:
        # gold keys are `experiment/run_name`, the catalogue's are split in two
        keep = {json.loads(l)["run"] for l in GOLD.open()}
        pop = [r for r in pop if f'{r["experiment"]}/{r["run_name"]}' in keep]
    if args.benchmark:
        pop = [r for r in pop if r["benchmark"] == args.benchmark]
    pop.sort(key=lambda r: (r["benchmark"], r["run_name"]))
    if args.limit:
        pop = pop[:: max(1, len(pop) // args.limit)][: args.limit]
    return pop


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--limit", type=int, default=0,
                    help="spread this many runs across the population")
    ap.add_argument("--benchmark", default="")
    ap.add_argument("--gold", action="store_true",
                    help="only the 143 runs the heavy pipeline already did")
    ap.add_argument("--redo", action="store_true", help="ignore what is on disk")
    args = ap.parse_args()

    pop = load_population(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists() and not args.redo:
        with args.out.open() as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["run"])
                except Exception:  # noqa: BLE001 -- a torn last line
                    pass
    todo = [r for r in pop if r["run_name"] not in done]
    print(f"{len(pop)} runs in scope, {len(done)} already on disk, "
          f"{len(todo)} to do -> {args.out}", flush=True)
    if not todo:
        return 0

    ex = Extractor()
    root = paths.events_dir("posttrainbench")
    fh = args.out.open("w" if args.redo else "a")
    wlock, state = threading.Lock(), collections.Counter()
    t0 = time.time()

    def one(row):
        run = row["run_name"]
        try:
            events = R.select(root / f'{row["experiment"]}__{run}.jsonl.gz')
            digest = R.render(run, events, row, include_agent=False)
            rec, prov = ex.run(digest)
            if rec is None:
                out = {"run": run, "extraction": {**prov, "ok": False}}
            else:
                out = {"run": run, "experiment": row["experiment"],
                       "benchmark": row["benchmark"],
                       "trained_model": row["trained_model"],
                       "agent_model": row["agent_model"],
                       "trace_format": row.get("trace_format"),
                       "seed": row.get("seed"),
                       "time_budget_h": row.get("time_budget_h"),
                       "time_taken": row.get("time_taken"),
                       **rec}
                out = R.join_outcome(out, row)
                out["extraction"] = {**prov, "ok": True, "model": MODEL,
                                     "tier": "tier1"}
        except Exception as e:  # noqa: BLE001 -- one dead run must not stop 1174
            out = {"run": run, "extraction": {"ok": False, "status": "error",
                                              "error": f"{type(e).__name__}: {e}"[:400]}}
        with wlock:
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")
            fh.flush()
            st = out["extraction"].get("status", "?")
            state[st] += 1
            state["n"] += 1
            if state["n"] % 25 == 0 or state["n"] == len(todo):
                el = time.time() - t0
                u = ex.usage
                print(f'{state["n"]:5d}/{len(todo)} {el/60:5.1f}min '
                      f'{state["n"]/max(el,1)*3600:5.0f}/h  '
                      + " ".join(f"{k}={state[k]}" for k in sorted(state)
                                 if k != "n")
                      + f'  | in={u["input_tokens"]/1e6:.1f}M '
                        f'cache_r={u["cache_read_input_tokens"]/1e6:.1f}M '
                        f'out={u["output_tokens"]/1e6:.2f}M', flush=True)
        return out

    with ThreadPoolExecutor(args.workers) as p:
        list(p.map(one, todo))
    fh.close()
    u = ex.usage
    cost = (u["input_tokens"] * 5 + u["cache_creation_input_tokens"] * 6.25
            + u["cache_read_input_tokens"] * 0.5 + u["output_tokens"] * 25) / 1e6
    print(f'\ndone in {(time.time()-t0)/60:.1f} min, {u["calls"]} calls, '
          f"~${cost:,.2f}")
    print("  " + "  ".join(f"{k}={v}" for k, v in sorted(state.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
