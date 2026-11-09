# Programming sample, page 2: "New cell (added for this sample; run here on the result files saved in the repo)".
# Usage, from the repository root:  python programming_sample/nb34_m0_exact_null.py

# --- non-interactive backend, so that the file runs on its own ---------------------
import matplotlib
matplotlib.use('Agg')
# --- the cell as printed -----------------------------------------------------------
# Added for this sample, run on the repo's files: the notebook (section 2) ran prepare(),
# exact_law() and exact_moments() on the real design (800 M0 alerts, 1.41M candidate minutes,
# up to 25,920 states per month) and saved the results and 1,000 old-resampler draws here.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
RESULTS = 'results/exact_conditional_null_v1/'
law = pd.read_csv(RESULTS + '2b_m0_exact_law.csv',
                  dtype={'k': 'int64', 'prob': 'float64',
                         'last_bin_means_at_least': bool})
m0 = pd.read_csv(RESULTS + '2_m0_exact_vs_budget.csv').iloc[0]  # one row: float64 Series
t_obs = int(m0.pre_observed)                                 # float -> int for indexing
top = int(law.k[law.last_bin_means_at_least].iloc[0])       # last bin means "T >= top"
p_tail = law.loc[law.k >= t_obs, 'prob'].sum()              # P(T >= t_obs) from the law
np.testing.assert_allclose(law.prob.sum(), 1.0, rtol=1e-12)      # a proper distribution
np.testing.assert_allclose(p_tail, m0.exact_p, rtol=1e-10, atol=0)   # files agree
print(f'M0: {int(m0.alerts)} alerts, {int(m0.events)} events, '
      f'{t_obs} caught before onset')
print(f'exact null: mean {m0.exact_mean:.2f}, sd {m0.exact_sd:.2f}, '
      f'P(T >= {t_obs}) = {p_tail:.1e}')
# a Monte Carlo p-value from B draws can never be below 1 / (B + 1)
print(f'old resampler, {int(m0.old_draws)} draws: mean {m0.old_null_mean:.2f}, '
      f'p = {m0.old_p_mc:.4f} (floor {1 / (m0.old_draws + 1):.4f})')
fig, ax = plt.subplots(figsize=(6.4, 2.15))
ax.bar(law.k, law.prob, width=0.8, color='#2471a3',
       label='exact law of T (no Monte Carlo)')
ax.axvline(t_obs, color='#c0392b', lw=1.5, label=f'observed T = {t_obs}')
ax.set_yscale('log')
ax.set(ylim=(1e-10, 50), xlabel=f'pre-onset catches T (last bar: T >= {top})',
       ylabel='probability (log scale)',
       title=f'Exact null for the {int(m0.alerts)} M0 alerts, development period 2023-26')
ax.legend(frameon=False, fontsize=8, loc='upper left', ncol=2)
plt.tight_layout()
plt.savefig('exact_null_m0.png', dpi=170)
plt.show()
# --- move the figure next to the other outputs ------------------------------------
from pathlib import Path
Path('exact_null_m0.png').replace('programming_sample/output/exact_null_m0.png')
print('Wrote programming_sample/output/exact_null_m0.png')
# Printed output in the sample:
# M0: 800 alerts, 83 events, 22 caught before onset
# exact null: mean 6.70, sd 2.36, P(T >= 22) = 7.4e-08
# old resampler, 1000 draws: mean 6.17, p = 0.0010 (floor 0.0010)
