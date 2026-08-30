"""Does the recipe carry signal the catalogue row does not?

A better split is necessary and not sufficient. The battery shows metadata is not
*sufficient* to predict the outcome; the thesis needs the stronger claim that
trajectory content is *available*. This probes that on the only rows where
recipes exist today -- the 143 gsm8k train rows of the shipped split.

Three passes, because the first two answers were both wrong:

1. Raw one-way R2 on the within-cell delta. Ranks the recipe top.
2. Same, minus a within-cell permutation null. A one-way R2 on a near-unique
   categorical is high whatever the labels are -- `dataset_names` has 107 levels
   over 143 rows, so it is nearly a row id. Ranking features of different
   cardinality by raw R2 measures cardinality. This pass reverses pass 1.
3. The one that matters. The recommended split *removes agent identity from
   train*, so the question is not "does the agent predict the score" but "does
   the recipe predict it once the agent is known". Residualise on agent_family,
   re-test against a within-*family* permutation null.

Run: OMP_NUM_THREADS=4 python3 tools/splitdx/recipe_signal.py   (from the repo root)
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import battery as B  # noqa: E402

RECIPES = HERE.parents[1] / "splits/posttrainbench/gsm8k-gemma-holdout-v1.recipes.jsonl"
DRAWS = 500


def cell(r):
    return (r["benchmark"], r["trained_model"])


def feat(r):
    """Seven coarse summaries of the extracted recipe.

    Deliberately coarse: these are counts and family signatures, not the recipe
    text. A model reading the trajectory has more to work with -- but if even
    these show nothing, "more to work with" is the claim that needs evidence.
    """
    algs = r.get("algorithms") or []
    ds = r.get("datasets") or []
    fams = tuple(a.get("family") for a in algs if isinstance(a, dict))
    names = tuple(sorted({d.get("name") for d in ds if isinstance(d, dict)}))
    kinds = tuple(sorted({d.get("kind") for d in ds if isinstance(d, dict)}))
    return {
        "pipeline_signature": ">".join(fams) or "(none)",
        "n_stages": str(len(algs)),
        "uses_rl": str(bool({"grpo", "dpo", "ppo", "rl", "rlhf"} & set(fams))),
        "dataset_kinds": "|".join(kinds) or "(none)",
        "dataset_names": "|".join(names)[:120] or "(none)",
        "n_datasets": str(len(ds)),
        "n_inference_tricks": str(len(r.get("inference_tricks") or [])),
        "n_discarded": str(len(r.get("discarded") or [])),
        "confidence": str(r.get("confidence")),
    }


#: pass 3 drops the two that pass 2 showed are cardinality artefacts or constant
CONDITIONED = ("pipeline_signature", "n_stages", "uses_rl", "dataset_kinds",
               "n_datasets", "n_inference_tricks", "n_discarded")


def load():
    rows = [json.loads(l) for l in RECIPES.open()]
    rows = [r for r in rows if isinstance(r.get("accuracy"), (int, float))]
    for r in rows:
        r.update(feat(r))
        r["agent_family"] = B.agent_family(r["agent_model"])
    return rows


def _null(rows, y, f, blocks, rng):
    """R2 the feature's own level structure earns from noise, permuting inside
    the blocks that the design (or the residualisation) holds fixed."""
    out = []
    for _ in range(DRAWS):
        yp = y.copy()
        for ix in blocks:
            yp[ix] = y[rng.permutation(ix)]
        out.append(B.one_way_r2(rows, yp, f))
    return np.array(out)


def main() -> int:
    rows = load()
    cells = sorted({cell(r) for r in rows})
    recipe = list(feat(rows[0]))
    meta = ["agent_model", "agent_family", "trace_format", "trained_model"]
    counts = collections.Counter(cell(r) for r in rows)
    print(f"{len(rows)} rows with an accuracy, {len(cells)} cells: "
          + ", ".join(f"{k[1]}={v}" for k, v in counts.most_common()))

    med = {k: float(np.median([r["accuracy"] for r in rows if cell(r) == k])) for k in cells}
    y = np.array([r["accuracy"] - med[cell(r)] for r in rows], dtype=float)
    by_cell = [np.array([i for i, r in enumerate(rows) if cell(r) == k]) for k in cells]

    print("\n=== 1. does the recipe vary INSIDE a cell? (distinct values / rows) ===")
    print(f'{"feature":>22}  ' + "  ".join(f"{k[1][:14]:>14}" for k in cells))
    for f in recipe + meta:
        line = f"{f:>22}  "
        for k in cells:
            sub = [r for r in rows if cell(r) == k]
            line += f"{len(set(r[f] for r in sub)):>7}/{len(sub):<6} "
        print(line)

    print(f"\n=== 2. cardinality-controlled: R2 minus its within-cell permutation null "
          f"({DRAWS} draws) ===")
    rng = np.random.default_rng(0)
    print(f'{"feature":>22} {"levels":>7} {"R2":>8} {"null":>8} {"excess":>8} {"z":>7}')
    out = []
    for f in recipe + meta:
        obs = B.one_way_r2(rows, y, f)
        n = _null(rows, y, f, by_cell, rng)
        z = (obs - n.mean()) / n.std() if n.std() > 0 else float("nan")
        out.append((obs - n.mean(), z, f, len(set(r[f] for r in rows)), obs, n.mean(),
                    f in recipe))
    for excess, z, f, lv, obs, nm, is_recipe in sorted(out, reverse=True):
        print(f"{f:>22} {lv:7d} {obs:8.4f} {nm:8.4f} {excess:+8.4f} {z:+7.1f}  "
              f'{"RECIPE" if is_recipe else "meta"}')

    # --- pass 3
    fam_idx = collections.defaultdict(list)
    for i, r in enumerate(rows):
        fam_idx[r["agent_family"]].append(i)
    blocks = [np.array(v) for v in fam_idx.values()]
    resid = y.copy()
    for ix in blocks:
        resid[ix] -= y[ix].mean()
    print("\n=== 3. recipe signal AFTER conditioning on agent_family ===")
    print(f"    {len(blocks)} families; residual keeps "
          f"{100*resid.var()/y.var():.0f}% of the within-cell variance")
    rng = np.random.default_rng(0)
    print(f'{"feature":>22} {"levels":>7} {"R2":>8} {"null":>8} {"excess":>8} {"z":>7}')
    surv = []
    res3 = []
    for f in CONDITIONED:
        obs = B.one_way_r2(rows, resid, f)
        n = _null(rows, resid, f, blocks, rng)
        z = (obs - n.mean()) / n.std() if n.std() > 0 else float("nan")
        res3.append((obs - n.mean(), z, f, len(set(r[f] for r in rows)), obs, n.mean()))
    for excess, z, f, lv, obs, nm in sorted(res3, reverse=True):
        flag = "  <-- survives" if z > 3 else ""
        if z > 3:
            surv.append(f)
        print(f"{f:>22} {lv:7d} {obs:8.4f} {nm:8.4f} {excess:+8.4f} {z:+7.1f}{flag}")

    print(f"\n  {len(surv)} of {len(CONDITIONED)} recipe features survive conditioning "
          f"on the agent: " + (", ".join(surv) if surv else "(none)"))
    best_z = max(z for _, z, *_ in res3)
    print(f"  strongest surviving z = {best_z:+.1f}; reaching z=3 at this effect size needs "
          f"about {(3/best_z)**2:.1f}x the rows (~{int(len(rows)*(3/best_z)**2)}). "
          f"The full corpus has 1175, so this is resolvable -- after the extraction "
          f"is run beyond gsm8k.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
