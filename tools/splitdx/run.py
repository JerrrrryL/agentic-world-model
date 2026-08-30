"""Evaluate candidate split designs through one shared battery.

    python3 run.py designs/my_design.py [more.py ...]

Each design file must define ``DESIGNS``: a list of ``battery.Design``. Every
number every design reports comes from this one code path, so two designs'
numbers are comparable and neither author can pick a friendlier metric.

The shipped split is always evaluated first, as a positive control. It is known
to saturate — per-agent lookup, top-3 regret 0.0000, Spearman 0.7507 — and if
this run does not reproduce that, the harness is broken and nothing below it
means anything.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1])]

import battery as B  # noqa: E402
from awm import splits  # noqa: E402
from awm.traj import fetch  # noqa: E402

POP = B.scoreable(fetch.ptb_catalog()["runs"])
_S = splits.load("posttrainbench/gsm8k-gemma-holdout-v1")
_TR, _TE = set(_S.train), set(_S.test)


def _path(r):
    return f'{r["experiment"]}/{r["run_name"]}'


def shipped(rows):
    return ([r for r in rows if _path(r) in _TR], [r for r in rows if _path(r) in _TE])


CONTROL = B.Design(
    name="CONTROL — shipped gsm8k-gemma-holdout-v1",
    split=shipped,
    note="known-saturated; must print per-agent regret@3 = 0.0 and spearman 0.7507",
)


def check_control(res: dict) -> None:
    per_agent = next(b for b in res["baselines"] if b["baseline"] == "per-agent")
    bad = []
    if per_agent["regret@3"] != 0.0:
        bad.append(f'regret@3 is {per_agent["regret@3"]}, known value 0.0')
    if abs(per_agent["spearman"] - 0.7507) > 5e-4:
        bad.append(f'spearman is {per_agent["spearman"]}, known value 0.7507')
    if abs(res["r2_train"]["agent_model"] - 0.6632) > 5e-4:
        bad.append(f'agent_model R² is {res["r2_train"]["agent_model"]}, known value 0.6632')
    if bad:
        raise SystemExit("CONTROL FAILED — the harness is not measuring what it did before:\n  "
                         + "\n  ".join(bad))


def load(path: str) -> list:
    spec = importlib.util.spec_from_file_location(Path(path).stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return list(mod.DESIGNS)


def main(argv: list[str]) -> int:
    res = B.evaluate(CONTROL, POP)
    print(B.render(res), "\n")
    check_control(res)
    for path in argv:
        for d in load(path):
            try:
                print(B.render(B.evaluate(d, POP)), "\n")
            except Exception as exc:  # a broken design must not hide the others
                print(f"### {d.name}\n    !! raised {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
