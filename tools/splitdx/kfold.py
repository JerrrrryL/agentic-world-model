"""Remove the roster knob: don't ship one holdout, ship K folds and pool.

A single agent-family holdout is a defensible split whose headroom happens to
depend by a factor of ten on which families you picked (see roster.out). Nobody
can adjudicate that choice from the outside, so the honest form of this design is
a partition: every family is in test exactly once, every run is scored exactly
once out-of-fold, and the reported number is the pooled one. The fold spread then
becomes the error bar instead of a degree of freedom.

Folds are built by greedy balancing on run count -- deterministic, no seed, and
it keeps the folds close enough in size that the pooled mean is not dominated by
one fold.
"""
import sys, collections
import numpy as np
from pathlib import Path as _P
sys.path[:0] = [str(_P(__file__).resolve().parent), str(_P(__file__).resolve().parents[2])]
import battery as B, ceiling as C, run as R

POP = list(R.POP)
fam_of = lambda r: B.agent_family(r['agent_model'])
counts = collections.Counter(fam_of(r) for r in POP)
K = 5

folds = [[] for _ in range(K)]
load = [0]*K
for f, n in counts.most_common():          # largest first into the lightest fold
    i = int(np.argmin(load)); folds[i].append(f); load[i] += n
print(f'{K} folds over {len(counts)} agent families, {len(POP)} runs')
for i, (fs, n) in enumerate(zip(folds, load)):
    print(f'  fold {i}: {n:4d} runs, {len(fs):2d} families  {", ".join(sorted(fs))}')


def design_for(held, tag, target):
    held = frozenset(held)
    def split(rows):
        return ([r for r in rows if fam_of(r) not in held],
                [r for r in rows if fam_of(r) in held])
    return B.Design(name=tag, split=split, target=target,
                    choice=lambda r: (r['benchmark'], r['trained_model']))


for tname, target in (('absolute', B.ABSOLUTE), ('delta-vs-cell', B.DELTA_CELL)):
    print(f'\n=== target: {tname} ===')
    print(f'{"fold":>5} {"n_test":>7} {"sets":>5} {"med":>4} {"dumb@3":>7} {"which":>16} '
          f'{"floor@3":>8} {"gbdt@3":>7} {"rho":>7} {"win":>8} {"win%":>6}')
    rows = []
    for i, fs in enumerate(folds):
        d = design_for(fs, f'fold-{i}', target)
        res = B.evaluate(d, POP)
        best = min((b for b in res['baselines'] if b['regret@3'] == b['regret@3']),
                   key=lambda b: b['regret@3'])
        fl = C.regret_floor(d, POP); mc = C.metadata_ceiling(d, POP)
        sp = res['headroom']['mean_within-choice-set accuracy spread']
        win = best['regret@3'] - fl['floor@3']
        rows.append((res['n_test'], best['regret@3'], fl['floor@3'], win, win/sp,
                     mc['regret@3'], mc['spearman']))
        print(f'{i:5d} {res["n_test"]:7d} {res["choice_sets"]["n"]:5d} '
              f'{res["choice_sets"]["median_size"]:4d} {best["regret@3"]:7.4f} '
              f'{best["baseline"]:>16} {fl["floor@3"]:8.4f} {mc["regret@3"]:7.4f} '
              f'{mc["spearman"]:+7.3f} {win:+8.4f} {100*win/sp:5.1f}%')
    a = np.array(rows); w = a[:, 0]/a[:, 0].sum()
    print(f'{"POOL":>5} {int(a[:, 0].sum()):7d} {"":>5} {"":>4} {a[:, 1]@w:7.4f} {"":>16} '
          f'{a[:, 2]@w:8.4f} {a[:, 5]@w:7.4f} {a[:, 6]@w:+7.3f} {a[:, 3]@w:+8.4f} {100*(a[:, 4]@w):5.1f}%')
    print(f'      fold spread of winnable share: {100*a[:, 4].min():.1f}% - {100*a[:, 4].max():.1f}%'
          f'   (sd {100*a[:, 4].std():.1f} pts)   all positive: {bool((a[:, 3] > 0).all())}')
    print(f'      metadata GBDT beats the dumb lookup in {int((a[:, 5] < a[:, 1]).sum())}/{K} folds')
