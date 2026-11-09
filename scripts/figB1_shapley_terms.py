"""Figure B.1 (Online Appendix B): three terms x three attribution conventions of the 4 h -> 24 h change
in the GBM - M0 standard-AUC difference, with 95% block-bootstrap intervals (Table B.3).

Reads results/shapley_decomposition_v1/2_decomposition_bootstrap.csv (notebook 33; 2,000 joint 7-day-block
draws) and writes scripts/output/figB1_shapley_terms.png. Also prints Table B.3 from the same file.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'scripts' / 'output'
B = pd.read_csv(ROOT / 'results' / 'shapley_decomposition_v1' / '2_decomposition_bootstrap.csv')
B = B[(B.h1 == 240) & (B.h2 == 1440)].set_index(['attribution', 'term'])

TERMS = [('reweighting', r'reweighting ($\pi$)'), ('pre_onset', 'pre-onset discrimination'),
         ('in_progress', 'in-progress discrimination')]
CONV = [('base_h1', 'base: 4-hour definition', 's', 'tab:blue', -0.22),
        ('shapley', 'Shapley–Shorrocks', 'o', 'tab:orange', 0.0),
        ('base_h2', 'base: 24-hour definition', 's', 'tab:green', 0.22)]


def s(x):
    return f'{x:+.3f}'.replace('-', '−')


print('Table B.3 (point [95% interval]; share of draws with the point estimate\'s sign)')
for a, label, *_ in CONV:
    print(f'  {label:<26}' + ' | '.join(f'{s(r.point)} [{s(r.lo)}, {s(r.hi)}]; {r.share_draws_same_sign_as_point:.0%}'
                                         for r in (B.loc[(a, t)] for t, _ in TERMS)))
dD = B.loc[('-', 'dD')]
print(f'  change D2 - D1: {s(dD.point)} [{s(dD.lo)}, {s(dD.hi)}]')

plt.rcParams.update({'font.family': ['Inter', 'DejaVu Sans'], 'font.size': 10, 'mathtext.fontset': 'dejavusans'})
fig, ax = plt.subplots(figsize=(7.2, 3.8))
for a, label, marker, color, dx in CONV:
    r = [B.loc[(a, t)] for t, _ in TERMS]
    y = [q.point for q in r]
    err = [[q.point - q.lo for q in r], [q.hi - q.point for q in r]]
    ax.errorbar([i + dx for i in range(3)], y, yerr=err, fmt=marker, color=color, ms=5.5, capsize=3, lw=1.5, label=label)
ax.axhline(0, color='0.5', lw=1)
ax.set_xticks(range(3), [lab for _, lab in TERMS])
ax.set_ylabel('contribution to the change in $D$')
ax.set_title(f'GBM − M0 standard-AUC difference, 4 h → 24 h: change {s(dD.point)} [{s(dD.lo)}, {s(dD.hi)}]', fontsize=10)
ax.legend(frameon=False, fontsize=8.5, loc='upper right')
ax.spines[['top', 'right']].set_visible(False)
fig.subplots_adjust(left=0.118, right=0.959, bottom=0.109, top=0.904)   # axes box of the published figure
fig.savefig(OUT / 'figB1_shapley_terms.png', dpi=200)
print('Wrote', OUT / 'figB1_shapley_terms.png')
