"""Rank a whole choice set, and report the metric the thesis is actually about.

`traj_read.py` showed a model comparing two trajectories picks the better one
86.7 % of the time. That is a discriminator shown both sides. The thesis needs a
*predictor*: given a cell's worth of candidate runs, pick the three most likely
to be best, and pay `regret@3 = best_in_cell - best_of_the_three_you_picked`.

Two things break when you go from a pair to a choice set, and both are the point:

- **Scoring in isolation is a different task.** In a pair, "which of these two"
  can be answered by any relative difference. Alone, the model has to say how
  good a run is against a standard it has to supply itself. Stage A measures
  exactly that, one trajectory per call.
- **A choice set contains the agent confound at full strength.** A cell holds
  ~40 runs from ~20 agent families, and memorising "which agent scores well" is
  worth 66 % of the variance -- `a-lookup-table-saturates-the-gsm8k-gemma-split`
  is that baseline hitting regret 0.0000. So every number here is reported twice:
  over the whole cell, and over within-family sub-sets where that table is
  useless.

The pipeline is the cheap thing that a deployment would actually do:

    stage A   score all n runs independently                      n calls
    stage B   round-robin the top K by that score, both orders    K(K-1) calls
              Copeland over B, ties broken by A

Stage B exists because A and B fail differently: A is calibration-limited, B is
the arm we measured at 86.7 % but costs O(n^2). Shortlisting with A and deciding
with B costs O(n + K^2) and is the only part of this that would survive contact
with a real choice set.

    python3 tools/choice_rank.py --stage a          # 1175 calls, ~$150
    python3 tools/choice_rank.py --stage b --topk 6 # ~840 calls, ~$210
    python3 tools/choice_rank.py --report
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tools/splitdx")]

import recipe_signal2 as RS  # noqa: E402
import traj_read as T  # noqa: E402

OUT = T.OUTDIR
SCORE_TOOL = {
    "name": "rate",
    "description": "Rate one post-training attempt.",
    "input_schema": {
        "type": "object",
        "properties": {
            "quality": {
                "type": "integer", "minimum": 0, "maximum": 100,
                "description": "Where this attempt lands among all serious "
                               "attempts at this exact task: 0 = worse than "
                               "doing nothing to the base model, 50 = a "
                               "median competent attempt, 100 = the best "
                               "attempt you can imagine anyone making.",
            },
            "predicted_accuracy": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "Your point estimate of the held-out benchmark "
                               "score this run's shipped model will get.",
            },
            "beats_base_model": {"type": "boolean"},
            "why": {"type": "string", "description": "One sentence."},
        },
        "required": ["quality", "predicted_accuracy", "beats_base_model", "why"],
    },
}

SCORE_SYSTEM = """\
You are shown one attempt at a post-training task: an agent was given a base \
model, a benchmark, and a time budget, and told to improve the model's score on \
that benchmark. You see a digest of what it did. Rate how well it did.

You are NOT shown the benchmark result, and score-shaped numbers have been \
blanked out of the text. Any number that survives is the agent's own local \
measurement on its own sample -- not the graded result. Judge from what was \
actually done: whether the training data matches what the benchmark asks for, \
whether the output format matches what the grader accepts, how much data and how \
many steps, whether the optimiser settings are sane for that scale, whether the \
agent verified anything end to end, and whether what it finally shipped is what \
it validated.

The most common ways these attempts fail are not exotic: shipping a checkpoint \
that was never evaluated, training on a format the grader rejects, running out \
of budget mid-way, breaking the chat template or EOS handling, and training so \
little that nothing changed. Runs that do nothing at all are common; so are runs \
that make the model worse than it started.

`quality` is the ranking signal and must spread out -- you will be compared \
against other attempts at the same task, so avoid clustering everything at 50.\
"""


def cells(rows):
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["benchmark"], r["trained_model"])].append(r)
    return {k: sorted(v, key=lambda r: r["run"]) for k, v in sorted(g.items())}


# --- stage A: score every run on its own ---------------------------------------

def stage_a(rows, workers=48, budget=T.DIGEST_BUDGET):
    path = OUT / "score_a.jsonl"
    done = {}
    if path.exists():
        for line in path.open():
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            done[d["run"]] = d
    todo = [r for r in rows if r["run"] not in done]
    print(f"  stage A: {len(done)} cached, {len(todo)} to score")
    if not todo:
        return done

    jobs = [("redact", r["experiment"], r["run"],
             {k: r.get(k) for k in T.RC.HEADER_KEYS}) for r in todo]
    with ProcessPoolExecutor(16) as ex:
        text = dict(ex.map(T._text_worker, jobs, chunksize=4))

    j = T.Judge()
    fh = path.open("a")
    lock, st = threading.Lock(), collections.Counter()
    t0 = time.time()

    def one(r):
        msg = (f"{text[r['run']]}\n\nRate this attempt.")
        out = None
        for n in range(6):
            try:
                with j.c.messages.stream(
                    model=T.MODEL, max_tokens=16_000,
                    system=[{"type": "text", "text": SCORE_SYSTEM,
                             "cache_control": {"type": "ephemeral"}}],
                    messages=[{"role": "user", "content": msg}],
                    tools=[SCORE_TOOL],
                    tool_choice={"type": "tool", "name": "rate"},
                    thinking={"type": "adaptive"},
                ) as s:
                    m = s.get_final_message()
                with j.lock:
                    for k in ("input_tokens", "output_tokens",
                              "cache_creation_input_tokens",
                              "cache_read_input_tokens"):
                        j.usage[k] += getattr(m.usage, k, 0) or 0
                    j.usage["calls"] += 1
                tu = next((b for b in m.content if b.type == "tool_use"), None)
                out = tu.input if tu else None
                break
            except Exception as e:  # noqa: BLE001
                if "invalid_request" in str(e) or n == 5:
                    out = {"error": f"{type(e).__name__}: {e}"[:200]}
                    break
                time.sleep(min(60, 2 ** n))
        rec = {"run": r["run"], **(out or {})}
        with lock:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            st["n"] += 1
            if st["n"] % 100 == 0:
                print(f'      {st["n"]}/{len(todo)} '
                      f'{(time.time()-t0)/60:.1f}min', flush=True)
        return rec

    with ThreadPoolExecutor(workers) as tp:
        for rec in tp.map(one, todo):
            done[rec["run"]] = rec
    fh.close()
    u = j.usage
    cost = (u["input_tokens"] * 5 + u["cache_creation_input_tokens"] * 6.25
            + u["cache_read_input_tokens"] * 0.5
            + u["output_tokens"] * 25) / 1e6
    print(f'  stage A: {u["calls"]} calls, ~${cost:,.2f}')
    return done


# --- stage B: round-robin the shortlist ----------------------------------------

def stage_b(rows, a_scores, topk=6, workers=48):
    """Only the top `topk` of each cell, both orders, Copeland.

    A full round-robin over a 40-run cell is 1560 ordered comparisons; over 28
    cells that is 43,680 calls. Shortlisting first is not a shortcut bolted on
    for cost, it is the algorithm -- an O(n) scorer that narrows to K and an
    O(K^2) comparator that decides is what anyone would actually deploy.
    """
    path = OUT / "score_b.jsonl"
    done = {}
    if path.exists():
        for line in path.open():
            try:
                d = json.loads(line)
            except Exception:  # noqa: BLE001
                continue
            done[(d["cell"], d["x"], d["y"])] = d.get("winner")

    short = {}
    for cell, rs in cells(rows).items():
        rs = [r for r in rs if isinstance(a_scores.get(r["run"], {})
                                          .get("quality"), (int, float))]
        rs.sort(key=lambda r: (-a_scores[r["run"]]["quality"], r["run"]))
        short[cell] = rs[:topk]

    jobs = []
    for cell, rs in short.items():
        for x, y in itertools.permutations(rs, 2):
            key = (f"{cell[0]}|{cell[1]}", x["run"], y["run"])
            if key not in done:
                jobs.append((cell, x, y))
    print(f"  stage B: {len(done)} cached, {len(jobs)} comparisons to make "
          f"({sum(len(v) for v in short.values())} runs shortlisted)")

    if jobs:
        need = {r["run"]: r for cell, x, y in jobs for r in (x, y)}
        tj = [("redact", r["experiment"], r["run"],
               {k: r.get(k) for k in T.RC.HEADER_KEYS}) for r in need.values()]
        with ProcessPoolExecutor(16) as ex:
            text = dict(ex.map(T._text_worker, tj, chunksize=4))
        j = T.Judge()
        fh = path.open("a")
        lock, st = threading.Lock(), collections.Counter()
        t0 = time.time()

        def one(job):
            cell, x, y = job
            w, raw = j.ask(text[x["run"]], text[y["run"]])
            rec = {"cell": f"{cell[0]}|{cell[1]}", "x": x["run"], "y": y["run"],
                   "winner": w, "confidence": raw.get("confidence"),
                   "why": raw.get("why")}
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                st["n"] += 1
                if st["n"] % 100 == 0:
                    print(f'      {st["n"]}/{len(jobs)} '
                          f'{(time.time()-t0)/60:.1f}min', flush=True)
            return rec

        with ThreadPoolExecutor(workers) as tp:
            for rec in tp.map(one, jobs):
                done[(rec["cell"], rec["x"], rec["y"])] = rec["winner"]
        fh.close()
        u = j.usage
        cost = (u["input_tokens"] * 5 + u["cache_creation_input_tokens"] * 6.25
                + u["cache_read_input_tokens"] * 0.5
                + u["output_tokens"] * 25) / 1e6
        print(f'  stage B: {u["calls"]} calls, ~${cost:,.2f}')
    return done, short


def copeland(cell, shortlist, b_pairs, a_scores):
    """Wins over ordered pairs; a run compared both ways counts both. Ties fall
    back to the stage-A score, which is the only information Copeland lacks."""
    key = f"{cell[0]}|{cell[1]}"
    wins = collections.Counter()
    for x, y in itertools.permutations(shortlist, 2):
        w = b_pairs.get((key, x["run"], y["run"]))
        if w == "A":
            wins[x["run"]] += 1
        elif w == "B":
            wins[y["run"]] += 1
    return sorted(shortlist,
                  key=lambda r: (-wins[r["run"]],
                                 -a_scores.get(r["run"], {}).get("quality", 0),
                                 r["run"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="", help="a, b, ab, or empty")
    ap.add_argument("--topk", type=int, default=6)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    rows = T.load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    cs = cells(rows)
    print(f"{len(rows)} runs, {len(cs)} cells, "
          f"sizes {min(len(v) for v in cs.values())}-"
          f"{max(len(v) for v in cs.values())}")

    a = {}
    if "a" in args.stage:
        a = stage_a(rows, args.workers)
    else:
        p = OUT / "score_a.jsonl"
        if p.exists():
            for line in p.open():
                try:
                    d = json.loads(line)
                except Exception:  # noqa: BLE001
                    continue
                a[d["run"]] = d
    if "b" in args.stage:
        stage_b(rows, a, args.topk, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
