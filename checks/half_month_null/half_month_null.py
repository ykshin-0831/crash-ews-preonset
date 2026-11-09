"""Calendar placement of M0's alerts and the exact conditional null.

Usage (from the repository root):
    python checks/half_month_null/half_month_null.py describe
    python checks/half_month_null/half_month_null.py month_decile
    python checks/half_month_null/half_month_null.py half_month

Input: results/design_constant_v1/39_design.npz, the stored design of notebook 39
(eligible development minutes, their month and volatility decile, the pre-onset
indicator, the positions of M0's 800 development alerts, the cooldown and the
observed count T_obs = 22).

describe      shares of alerts, pre-onset minutes, eligible minutes and onsets on
              UTC days 1-15, and the share of null alerts expected on days 1-15
              when each month x decile cell places its alerts uniformly over its
              eligible minutes (cooldown ignored).
month_decile  exact null Q_nu of Appendix H.5: M0's counts in each month x decile
              cell and the 240-minute cooldown. Reproduces notebook 34
              (mean 6.70, p = 7.4e-8).
half_month    the same with each cell split into days 1-15 and days 16 onward.

The exact recursion (exactnull_nb34.py) is the code of notebook 34. The exact
runs take about 15 and 30 minutes on one CPU core and need about 2 and 4 GB.
"""
import json
import os
import sys
import time
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
DESIGN = os.path.join(HERE, '..', '..', 'results', 'design_constant_v1', '39_design.npz')


def day_of_month(t_min):
    day = (np.asarray(t_min, np.int64) * 60).astype('datetime64[s]').astype('datetime64[D]')
    return (day - day.astype('datetime64[M]')).astype(int) + 1


def load():
    z = np.load(DESIGN, allow_pickle=False)
    return {k: z[k] for k in z.files}


def describe(z):
    t = z['t_min'].astype(np.int64); mon = z['month_index'].astype(np.int64)
    dec = z['decile'].astype(np.int64); h = z['pre_onset'].astype(bool); pos = z['m0_positions']
    first = day_of_month(t) <= 15
    cells = Counter(zip(mon[pos].tolist(), dec[pos].tolist()))
    exp_first = 0.0
    for (m, d), n in cells.items():
        in_cell = (mon == m) & (dec == d)
        exp_first += n * first[in_cell].mean()
    alert_month = mon[pos]; alert_dom = day_of_month(t[pos])
    by_month = {m: alert_dom[alert_month == m] for m in np.unique(alert_month)}
    onset_first = int((day_of_month(z['onset_min']) <= 15).sum())
    return dict(
        alerts=int(len(pos)), alerts_days_1_15=int(first[pos].sum()),
        months=len(by_month), months_all_alerts_by_day_15=int(sum(v.max() <= 15 for v in by_month.values())),
        latest_alert_day=int(max(v.max() for v in by_month.values())),
        pre_onset_minutes=int(h.sum()), pre_onset_minutes_days_1_15=int((h & first).sum()),
        eligible_minutes=int(len(t)), eligible_share_days_1_15=float(first.mean()),
        events=int(len(z['onset_min'])), onsets_days_1_15=onset_first,
        expected_null_share_days_1_15_month_decile_no_cooldown=float(exp_first / len(pos)))


def exact(z, mode):
    import exactnull_nb34 as E
    t = z['t_min'].astype(np.int64); mon = z['month_index'].astype(np.int64)
    dec = z['decile'].astype(np.int64); h = z['pre_onset'].astype(np.int64); pos = z['m0_positions']
    C = int(z['cooldown_min']); t_obs = int(z['T_obs'])
    half = (day_of_month(t) > 15).astype(np.int64)
    cls = dec if mode == 'month_decile' else dec * 2 + half
    nu = Counter(zip(mon[pos].tolist(), cls[pos].tolist()))
    t0 = time.time()
    P = E.prepare(t, mon, cls, h, dict(nu), C)
    mean, var = E.exact_moments(P)
    law, _ = E.exact_law(P, t_obs)          # P(T = k) for k < T_obs, then P(T >= T_obs)
    cdf = np.cumsum(law[:-1])
    q = lambda a: int(np.argmax(cdf >= a)) if cdf[-1] >= a else None
    # expected number of null alerts on days 1-15: the same recursion with the day-1-15 indicator in place of h
    first = (day_of_month(t) <= 15).astype(np.int64)
    m_first, _ = E.exact_moments(E.prepare(t, mon, cls, first, dict(nu), C))
    return dict(mode=mode, classes_per_month=int(cls.max() + 1), T_obs=t_obs, exact_mean=float(mean),
                exact_sd=float(var ** 0.5), p_exact_T_ge_obs=float(law[-1]),
                quantile_025=q(0.025), quantile_975=q(0.975),
                law_P_T_eq_k=[float(x) for x in law[:-1]],
                excess_per_100_alerts=float(100 * (t_obs - mean) / len(pos)),
                expected_null_alerts_days_1_15=float(m_first), seconds=round(time.time() - t0, 1))


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('describe', 'month_decile', 'half_month'):
        sys.exit(__doc__)
    mode = sys.argv[1]
    z = load()
    res = describe(z) if mode == 'describe' else exact(z, mode)
    print(json.dumps(res, indent=1))
    with open(os.path.join(HERE, f'result_{mode}.json'), 'w') as f:
        json.dump(res, f, indent=1)
