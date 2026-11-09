# Programming sample, page 1: "New cell (added for this sample; run here on the result files saved in the repo)".
# Usage, from the repository root:  python programming_sample/nb18_scores_around_onset.py

# --- context from cell [1], so that this file runs on its own -------------------------
import matplotlib
matplotlib.use('Agg')
import pandas as pd
import matplotlib.pyplot as plt
PRIMARY_DEF = '4pct'
METHODS = ['M0_VF_logit', 'M1_vol_threshold', 'M2_VPIN', 'M3_GBM_rich']
STUDY = (-480, 240)
# --- the cell as printed -----------------------------------------------------------
# Added for this sample: cell [6] saved, for each model and each minute around crash onset, the
# median and quartiles of score percentiles over 83 crashes; redrawn with M0 dashed (M1 hid it).
R18 = 'results/standard_protocol_inflation_v1/'
es = pd.read_csv(R18 + '4_event_study.csv')          # long: one row per (method, minute)
comp = pd.read_csv(R18 + '1_label_composition.csv').query(
    "label == 'L1_event' and split == 'event' and definition == @PRIMARY_DEF").iloc[0]
med = es.pivot(index='minutes_from_onset', columns='method', values='median_pct')  # -> wide
m0 = es[es.method == 'M0_VF_logit']
colors = dict(zip(METHODS, ['#2471a3', '#16a085', '#7f8c8d', '#c0392b']))
fig, ax = plt.subplots(figsize=(9, 3.25))
ax.axvspan(STUDY[0], 0, color='#f5b041', alpha=0.08, lw=0)               # before onset
ax.fill_between(m0.minutes_from_onset, m0.q25, m0.q75, color=colors['M0_VF_logit'],
                alpha=0.18, lw=0)
for mname in METHODS:
    is_m0 = mname == 'M0_VF_logit'
    ax.plot(med.index, med[mname], color=colors[mname], label=mname, lw=1.5 if is_m0 else 1,
            ls=(0, (4, 2)) if is_m0 else '-', zorder=3 if is_m0 else 2)
ax.axvline(0, color='k', lw=1)
ax.axvline(comp.median_onset_to_breach_min, color='k', lw=1, ls=':')
ax.text(STUDY[0] + 10, 0.03, 'before onset: prediction', fontsize=8)
ax.text(5, 0.03, 'after onset: detection', fontsize=8)
ax.set(ylim=(0, 1), ylabel='score percentile vs 2022',
       xlabel='minutes from 1% onset (solid line); dotted line = median 4% breach',
       title=f'Scores around crash onset, {comp.dev_events} crashes (2023-26); '
             'shaded band = M0 interquartile range')
ax.legend(fontsize=8, loc='upper left')
plt.tight_layout()
plt.show()
# --- save the figure -------------------------------------------------------------
fig.savefig('programming_sample/output/scores_around_onset.png', dpi=170)
print('Wrote programming_sample/output/scores_around_onset.png')
