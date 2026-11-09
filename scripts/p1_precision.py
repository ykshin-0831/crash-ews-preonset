"""Rough precision of the pre-onset leg of prediction P1 in a 12-month prospective test (Section 9.4).

Section 9.4: "At the development rates (218 events under the 24-hour definition in 40 months,
30 of them with pre-onset minutes), twelve months bring about 65 events, about 9 of them with
pre-onset minutes. A block interval like that of Table 6 would then have a half-width of about
0.14, against a development difference of 0.090 ... (a rough scaling by the square root of the
number of events)."

Inputs (Tables 5 and 6):
  results/robustness_checks_v1/1_pre_onset_possible_by_window.csv  (notebook 32; Table 5: events, impossible events)
  results/shapley_decomposition_v1/2_decomposition_bootstrap.csv    (notebook 33; Table 6: GBM - M0 pre-onset AUC,
                                                                      24 hours, 2,000 joint 7-day-block draws)
I scale the half-width by sqrt(30 / 9) because the interval rests on the events with pre-onset minutes.
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
T5 = pd.read_csv(ROOT / 'results' / 'robustness_checks_v1' / '1_pre_onset_possible_by_window.csv').set_index('H').loc[1440]
T6 = pd.read_csv(ROOT / 'results' / 'shapley_decomposition_v1' / '2_decomposition_bootstrap.csv')
T6 = T6[(T6.h1 == 240) & (T6.h2 == 1440) & (T6.term == 'dpre_h2')].iloc[0]
DEV_MONTHS = (pd.Timestamp('2026-05-01').to_period('M') - pd.Timestamp('2023-01-01').to_period('M')).n   # 40
TEST_MONTHS = 12

events, feasible = int(T5.events), int(T5.events - T5.impossible_events)
ev12, fe12 = events * TEST_MONTHS / DEV_MONTHS, feasible * TEST_MONTHS / DEV_MONTHS
scale = np.sqrt(feasible / fe12)
for src, lo, hi, d in (('stored values', T6.lo, T6.hi, T6.point), ('Table 6 as printed', -0.175, -0.019, -0.090)):
    hw = (hi - lo) / 2
    print(f'{src}: development difference {d:+.3f} [{lo:+.3f}, {hi:+.3f}], half-width {hw:.4f}; '
          f'x sqrt({feasible}/{fe12:g}) = {scale:.3f} -> 12-month half-width {hw * scale:.3f}')
rows = [('development months', DEV_MONTHS, 40), ('24-hour events / with pre-onset minutes', f'{events} / {feasible}', '218 / 30'),
        ('expected in 12 months: events', f'{ev12:.0f}', '65'), ('expected in 12 months: with pre-onset minutes', f'{fe12:.0f}', '9'),
        ('12-month half-width', f'{(T6.hi - T6.lo) / 2 * scale:.2f}', '0.14'), ('development difference (absolute)', f'{abs(T6.point):.3f}', '0.090')]
print()
for label, got, stated in rows:
    print(f'  {label}: {got}  [Section 9.4: {stated}]  {"MATCH" if str(got) == str(stated) else "DIFFERS"}')
