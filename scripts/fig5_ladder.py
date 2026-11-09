"""Figure 5 of the manuscript: the effect-size ladder of Table 10.

M0's pre-onset and in-progress yields (above-null catches per 100 alerts) by sample under the regime-matched
null, with the Bitcoin backcast's pre-specified circular-shift reading marked. Sources, as named in Table 10:
  development      results/methods_definitions_v1/4_methods_definitions_nulls.csv   (notebook 16)
                   results/onset_aware_training_v1/1_alerts_by_label_and_null.csv   (notebook 26: budget-preserving)
  altcoins         results/HOLDOUT17_multicoin_2018_2019_v1/H17_primary_decisions.csv, H17_all_tests.csv (notebook 17)
  Bitcoin backcast results/onset_aware_training_v1/1_alerts_by_label_and_null.csv   (notebook 26: regime-matched and
                   budget-preserving re-readings of the locked alerts)
                   results/HOLDOUT_2018_2020_confirmatory_v1/HOLDOUT_RESULTS.json, H1_circular_shift_null.csv
                   (notebook 11: the pre-specified circular-shift null, 10,000 draws)
Prints Table 10 and its note, and writes scripts/output/fig5_ladder.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
OUT = ROOT / 'scripts' / 'output'

n16 = pd.read_csv(RES / 'methods_definitions_v1' / '4_methods_definitions_nulls.csv')
n16 = n16[(n16.method == 'M0_VF_logit') & (n16.definition == '4pct')].set_index(['null', 'endpoint'])
n26 = pd.read_csv(RES / 'onset_aware_training_v1' / '1_alerts_by_label_and_null.csv')
n26 = n26[(n26.set == 'M0_VF_logit') & n26.label.str.startswith('A_standard')].set_index(['period', 'null'])
n17 = pd.read_csv(RES / 'HOLDOUT17_multicoin_2018_2019_v1' / 'H17_primary_decisions.csv').set_index('method').loc['M0_VF_logit']
t17 = pd.read_csv(RES / 'HOLDOUT17_multicoin_2018_2019_v1' / 'H17_all_tests.csv')
t17 = t17[(t17.method == 'M0_VF_logit') & (t17.definition == 'd4')].set_index(['null', 'endpoint'])
n11 = json.loads((RES / 'HOLDOUT_2018_2020_confirmatory_v1' / 'HOLDOUT_RESULTS.json').read_text())['H1']
d11 = pd.read_csv(RES / 'HOLDOUT_2018_2020_confirmatory_v1' / 'H1_circular_shift_null.csv')

rm = n16.loc[('regime-matched (+-30d)', 'pre')]
S = {  # sample: alerts, events, pre observed, regime-matched pre null, in-progress observed, regime-matched in-progress null
    'dev': dict(alerts=rm.alarms, events=rm.events, pre=rm.observed, pre_rm=rm.null_mean,
                post=n16.loc[('regime-matched (+-30d)', 'post')].observed, post_rm=n16.loc[('regime-matched (+-30d)', 'post')].null_mean,
                pre_bp=n26.loc[('dev', 'budget-preserving regime')].pre_null_mean),
    'alt': dict(alerts=n17.alerts, events=n17.events, pre=n17.pre, pre_rm=n17.pre_null_mean_regime,
                post=n17.post, post_rm=n17.post_null_mean_regime),
    'btc': dict(alerts=n11['alarms'], events=n11['events'], pre=n11['pre_onset_hits'], pre_rm=n26.loc[('holdout', 'regime-matched')].pre_null_mean,
                post=n26.loc[('holdout', 'regime-matched')].post, post_rm=n26.loc[('holdout', 'regime-matched')].post_null_mean,
                pre_bp=n26.loc[('holdout', 'budget-preserving regime')].pre_null_mean, pre_cs=n11['null_mean_pre_onset'],
                post_cs=(d11.any_hits - d11.pre_onset_hits).mean()),
}


def y(s, obs, null):
    return 100 * (S[s][obs] - S[s][null]) / S[s]['alerts']


rows = [('Development 2023-2026', 'dev', f"5.8 -> 2.0 | 6.3 -> 2.0"),
        ('Altcoins 2018-2019', 'alt', '42.1 -> 0.6 | Holm p 0.0006, joint circular Holm p 0.0012'),
        ('Bitcoin 2018-2020', 'btc', '10.2 -> 0.4 | 10.6 -> 0.3 | circular 11.4 -> 0.2 (p = 0.35)')]
print('Table 10 (M0 pre-onset yield, above-null catches per 100 alerts)')
for name, s, stated in rows:
    v = S[s]
    line = f"  {name}: alerts {v['alerts']:,}, observed {v['pre']}, regime-matched {v['pre_rm']:.1f} -> {y(s, 'pre', 'pre_rm'):.1f}"
    if 'pre_bp' in v:
        line += f", budget-preserving {v['pre_bp']:.1f} -> {y(s, 'pre', 'pre_bp'):.1f}"
    if s == 'alt':
        line += f", Holm p {n17.pre_p_holm_regime:.4f}, joint circular Holm p {n17.pre_p_holm_joint:.4f}"
    if s == 'btc':
        line += f", circular {v['pre_cs']:.1f} -> {y(s, 'pre', 'pre_cs'):.1f} (p = {n11['p_one_sided']:.2f})"
    print(line + f'   [Table 10: {stated}]')
print('Table 10 note (development, altcoins, Bitcoin): text 0.10/0.23/0.28; 19.5/2.6/1.4; 15.6/9.7/8.1; 1.6/2.2/2.3 (2.1)')
print('  events per alert   ' + ' / '.join(f"{S[s]['events'] / S[s]['alerts']:.2f}" for s in S))
print('  pre-onset excess per 100 events   ' + ' / '.join(f"{100 * (S[s]['pre'] - S[s]['pre_rm']) / S[s]['events']:.1f}" for s in S))
print('  in-progress excess per 100 events ' + ' / '.join(f"{100 * (S[s]['post'] - S[s]['post_rm']) / S[s]['events']:.1f}" for s in S))
print('  in-progress yield per 100 alerts  ' + ' / '.join(f"{y(s, 'post', 'post_rm'):.1f}" for s in S)
      + f" (Bitcoin circular-shift draws: {y('btc', 'post', 'post_cs'):.1f})")

# ---------------------------------------------------------------- figure
BLUE, ORANGE = 'tab:blue', 'tab:orange'
plt.rcParams.update({'font.family': ['Inter', 'DejaVu Sans'], 'font.size': 10})
fig, ax = plt.subplots(figsize=(7.2, 3.9))
fig.subplots_adjust(left=0.0997, right=0.98, bottom=0.36, top=0.90)       # axes box of the published figure
labels = ['Development\n2023–26', 'Altcoins\n2018–19', 'Bitcoin\n2018–20']
status = ['Exploratory', 'Pre-specified\none-shot evaluation', 'One-shot evaluation: bars post hoc\n(regime null), line pre-specified']
w = 0.36
for i, s in enumerate(S):
    pre, post = y(s, 'pre', 'pre_rm'), y(s, 'post', 'post_rm')
    ax.bar(i - w / 2, pre, w, color=BLUE)
    ax.bar(i + w / 2, post, w, color=ORANGE)
    txt = f'{pre:.1f}' + (f" ({y(s, 'pre', 'pre_cs'):.1f})" if s == 'btc' else '')
    ax.annotate(txt, (i - w / 2, pre), xytext=(0, 2.6), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    ax.annotate(f'{post:.1f}', (i + w / 2, post), xytext=(0, 2.6), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    ax.annotate(status[i], (i, 0), xycoords=('data', 'axes fraction'), xytext=(0, -34.5), textcoords='offset points',
                ha='center', va='top', fontsize=8, color='#555555')
cs = y('btc', 'pre', 'pre_cs')
ax.plot([2 - w / 2 - 0.15, 2 - w / 2 + 0.15], [cs, cs], color='black', lw=2.2, solid_capstyle='butt')
ax.set_xticks(range(3), labels)
ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 2.7)
ax.set_ylabel('above-null catches per 100 alerts')
ax.set_title('M0 yield by sample, regime-matched null (prospective test: planned, Section 9.4)', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
handles = [Patch(color=BLUE, label='pre-onset warning'), Patch(color=ORANGE, label='in-progress detection'),
           Line2D([], [], color='black', lw=2.2, label='Bitcoin pre-onset under its pre-specified circular-shift null (in parentheses)')]
fig.legend(handles=handles, loc='lower left', bbox_to_anchor=(0.049, 0.0), ncol=2, frameon=False, fontsize=8.5)
fig.savefig(OUT / 'fig5_ladder.png', dpi=200)
print('Wrote', OUT / 'fig5_ladder.png')
