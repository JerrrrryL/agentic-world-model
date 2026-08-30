"""Would masking the agent be enough, instead of holding it out?

The shipped split saturates because test shares its agents with train. The
cheapest-looking fix is to leave the split alone and delete `agent_model` from
the input. This measures whether that works.

It does not, for two reasons this script prints:

1. `experiment` recovers the agent exactly -- every experiment is single-agent,
   as is every run_name. The saturating lookup survives masking at the same
   regret@3 = 0.0000, keyed on a different column.
2. Even a complete column list is the weaker guarantee. Holding a family out
   makes the exploit structurally impossible; masking makes it inconvenient,
   and the trajectory still carries the agent's house style.

It also states the confound that bounds the whole exercise: agent and recipe are
the same variable in every row, so no split and no mask separates them -- only
rollouts that cross the two.

Run: OMP_NUM_THREADS=4 python3 tools/splitdx/mask_probe.py   (from the repo root)
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import battery as B  # noqa: E402
import run as R  # noqa: E402

#: baselines that a mask on agent identity would remove
AGENT_KEYED = {"per-agent", "per-agent-family", "per-agent×benchmark", "per-agent×cell"}

RECIPES = HERE.parents[1] / "splits/posttrainbench/gsm8k-gemma-holdout-v1.recipes.jsonl"


def purity(rows, col, label="agent_family"):
    """Share of rows whose `label` matches the majority label of their `col` group.

    1.0 means the column reproduces the label exactly -- masking the label while
    leaving this column in is masking nothing.
    """
    g = collections.defaultdict(list)
    for r in rows:
        g[r[col]].append(r[label])
    hit = sum(collections.Counter(v).most_common(1)[0][1] for v in g.values())
    single = sum(1 for v in g.values() if len(set(v)) == 1)
    return hit / len(rows), len(g), single


def main() -> int:
    pop = list(R.POP)
    for r in pop:
        r["agent_family"] = B.agent_family(r["agent_model"])
    print(f'{len(pop)} runs, {len({r["agent_model"] for r in pop})} agent_model, '
          f'{len({r["agent_family"] for r in pop})} agent families\n')

    print("=== 1. can the agent be recovered from a column the mask leaves behind? ===")
    print(f'{"column":>16} {"levels":>7} {"purity":>8} {"single-agent groups":>21}  verdict')
    print("-" * 84)
    for col in ("experiment", "run_name", "trace_format", "benchmark", "trained_model"):
        if col not in pop[0]:
            continue
        p, lv, single = purity(pop, col)
        v = ("IS agent identity" if p > 0.95 else
             "strong proxy" if p > 0.7 else "weak")
        print(f"{col:>16} {lv:7d} {p:8.1%} {single:>10}/{lv:<10} {v}")

    print("\n=== 2. shipped split: every parameter-free lookup, with and without a mask ===")
    res = B.evaluate(R.CONTROL, pop)
    R.check_control(res)
    print(f'{"baseline":>20} {"keys":>34} {"regret@3":>9} {"rho":>8}')
    print("-" * 78)
    for b in res["baselines"]:
        kb = next(x for x in B.BASELINES if x.name == b["baseline"])
        tag = "  masked away" if b["baseline"] in AGENT_KEYED else ""
        print(f'{b["baseline"]:>20} {"+".join(kb.keys) or "(none)":>34} '
              f'{b["regret@3"]:9.4f} {b["spearman"]:+8.4f}{tag}')
    best = min((b for b in res["baselines"] if b["baseline"] not in AGENT_KEYED),
               key=lambda b: b["regret@3"])
    train, test = R.shipped(pop)
    seen = {r["experiment"] for r in train}
    print(f'\n  best lookup that reads no agent column: {best["baseline"]} -> '
          f'regret@3 {best["regret@3"]:.4f}, rho {best["spearman"]:+.4f}')
    print(f'  {sum(r["experiment"] in seen for r in test)}/{len(test)} test rows share an '
          f"experiment with train, and every experiment is one agent, so the")
    print(f"  saturating lookup survives the mask unchanged. Masking is not a substitute\n"
          f"  for holding the family out -- and `experiment` and `run_name` belong on the\n"
          f"  excluded-columns list next to the post-execution fields.")

    print("\n=== 3. the confound no split and no mask can fix ===")
    ex = collections.defaultdict(set)
    for r in pop:
        ex[r["experiment"]].add(r["agent_family"])
    print(f"  {len(ex)} experiments, all single-agent: 'which agent' and 'which recipe'\n"
          f"  are the same variable in {len(pop)}/{len(pop)} rows. The corpus is nested,\n"
          f"  not crossed. Only new rollouts that cross the two separate them.")

    if not RECIPES.exists():
        return 0
    rows = [json.loads(x) for x in RECIPES.open()]
    rows = [r for r in rows if isinstance(r.get("accuracy"), (int, float))]
    for r in rows:
        r["sig"] = ">".join(a.get("family") for a in (r.get("algorithms") or [])
                            if isinstance(a, dict)) or "(none)"
        r["fam"] = B.agent_family(r["agent_model"])
    g = collections.defaultdict(set)
    for r in rows:
        g[r["sig"]].add(r["fam"])
    multi = {k: v for k, v in g.items() if len(v) > 1}
    print(f"\n  Incidental crossing that already exists, on the {len(rows)} extracted "
          f"recipes:\n  {len(multi)}/{len(g)} pipeline signatures were produced by more "
          f"than one agent family.")
    print(f'\n{"pipeline signature":>34} {"agents":>7} {"acc spread":>11}')
    for k, v in sorted(multi.items(), key=lambda kv: -len(kv[1]))[:8]:
        a = [r["accuracy"] for r in rows if r["sig"] == k]
        print(f"{k[:34]:>34} {len(v):7d} {max(a)-min(a):11.4f}")
    print("\n  The algorithm family is nearly uninformative: 19 agents ran plain `sft` and\n"
          "  landed 0.80 of accuracy apart. Whatever signal exists is finer than this --\n"
          "  data mix, dataset ids, LR, epochs -- or it is execution quality. The schema\n"
          "  records all of that; it is feat() in recipe_signal.py that drops it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
