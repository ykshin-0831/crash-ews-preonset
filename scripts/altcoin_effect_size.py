"""Post hoc summaries of the altcoin results in Section 8.2, read from the locked outputs.

The script reruns no model, no null draw and no price download; it reads
  results/HOLDOUT17_multicoin_2018_2019_v1/   (notebook 17, the one-shot altcoin backcast)
      H17_primary_decisions.csv, H17_all_tests.csv, H17_per_coin_counts.csv,
      phase2/<coin>_d4_events.csv (event catalogs of the primary d4 definition)
  results/altcoin_event_level_v1/             (notebook 37, event-level reading of the locked files)
      4_episode_level_test.csv, 5_cluster_bootstrap.csv, 7_paired_pre_onset.csv, 37_SUMMARY.json
  results/altcoin_dev_period_v1/              (notebook 43, frozen coin models on 2023-2026)
      2_pooled.csv, 43_SUMMARY.json
  results/methods_definitions_v1/4_methods_definitions_nulls.csv  (notebook 16, development reference)
and prints each value of Section 8.2 and of the note to Table B.9 (Online Appendix B.6) next to the value the
text states. loco44_summary.py prints the leave-one-coin-out values of Section 8.2 (notebook 44).
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
N17 = RES / 'HOLDOUT17_multicoin_2018_2019_v1'
N37 = RES / 'altcoin_event_level_v1'
N43 = RES / 'altcoin_dev_period_v1'
RESULTS = []


def check(label, got, stated):
    ok = got == stated
    RESULTS.append(ok)
    print(f'  {label}: {got}  [text: {stated}]  {"MATCH" if ok else "DIFFERS"}')


def per100(excess, alerts):
    return 100 * excess / alerts


# ---------------------------------------------------------------- notebook 17: primary decisions
D = pd.read_csv(N17 / 'H17_primary_decisions.csv').set_index('method')
T = pd.read_csv(N17 / 'H17_all_tests.csv')
T = T[T.definition == 'd4'].set_index(['method', 'null', 'endpoint'])
m0, m1, gbm = D.loc['M0_VF_logit'], D.loc['M1_vol_threshold'], D.loc['M3_GBM_rich']
print('Notebook 17, pre-specified decisions (pooled over ten coins)')
check('events', f'{m0.events}', '943')
check('M0 pre-onset vs regime-matched expectation, Holm p', f'{m0.pre} vs {m0.pre_null_mean_regime:.1f}, p = {m0.pre_p_holm_regime:.4f}', '67 vs 42.1, p = 0.0006')
check('M1 pre-onset vs expectation', f'{m1.pre} vs {m1.pre_null_mean_regime:.1f}', '71 vs 41.9')
check('GBM pre-onset vs expectation, Holm p', f'{gbm.pre} vs {gbm.pre_null_mean_regime:.1f}, p = {gbm.pre_p_holm_regime:.2f}', '35 vs 32.5, p = 0.35')

print('\nM0 effect size under the regime-matched null (Section 8.2, paragraph 2)')
ex_pre = m0.pre - m0.pre_null_mean_regime
ex_in = m0.post - m0.post_null_mean_regime
N16 = pd.read_csv(RES / 'methods_definitions_v1' / '4_methods_definitions_nulls.csv')
dev = N16[(N16.method == 'M0_VF_logit') & (N16.definition == '4pct') & (N16['null'] == 'regime-matched (+-30d)') & (N16.endpoint == 'pre')].iloc[0]
dev_ex = dev.observed - dev.null_mean
check('pre-onset excess (events) in alerts; per 100 alerts', f'{ex_pre:.0f} in {m0.alerts:,}; {per100(ex_pre, m0.alerts):.1f}', '25 in 4,101; 0.6')
check('alerts per extra pre-onset warning, altcoins vs development', f'{m0.alerts / ex_pre:.0f} vs {dev.alarms / dev_ex:.0f}', '165 vs 50')
# notebook 17's bound: observed minus the 5th percentile of the regime-matched null
check('95% upper bound of the pre-onset excess, per 100', f'{per100(m0.pre_excess_upper95, m0.alerts):.2f}', '0.85')
check('development pre-onset yield (notebook 16), per 100', f'{per100(dev_ex, dev.alarms):.1f}', '2.0')
check('in-progress excess (events); per 100 alerts', f'{ex_in:.0f}; {per100(ex_in, m0.alerts):.1f}', '91; 2.2')

print('\nShort shift (joint 4h-1d null, pre-specified secondary analysis)')
ys = {}
for meth, row in (('M0_VF_logit', m0), ('M1_vol_threshold', m1), ('M3_GBM_rich', gbm)):
    s = T.loc[(meth, 'joint 4h-1d', 'pre')]
    ys[meth] = per100(s.observed - s.null_mean, s.alerts)
    print(f'    {meth}: pre {s.observed} vs {s.null_mean:.1f} (p = {s.p_one_sided:.3f}), yield {ys[meth]:.2f} per 100; '
          f'in-progress p = {T.loc[(meth, "joint 4h-1d", "post")].p_one_sided:.4f}')
check('no system beats chance before onset (all p > 0.05)', str(all(T.loc[(m, 'joint 4h-1d', 'pre')].p_one_sided > 0.05 for m in ys)), 'True')
check('range of short-shift pre-onset yields, per 100', f'{min(ys.values()):.1f}-{max(ys.values()):.1f}', '0.2-0.3')
yi = {m: per100(T.loc[(m, 'joint 4h-1d', 'post')].observed - T.loc[(m, 'joint 4h-1d', 'post')].null_mean, T.loc[(m, 'joint 4h-1d', 'post')].alerts) for m in ys}
check('Table B.9 note: short-shift pre-onset yields M0, M1, GBM', ', '.join(f'{v:.2f}' for v in ys.values()), '0.24, 0.34, 0.17')
check('Table B.9 note: short-shift in-progress yields M0, M1, GBM', ', '.join(f'{v:.1f}' for v in yi.values()), '1.9, 2.1, 1.1')
s0, r0 = T.loc[('M0_VF_logit', 'joint 4h-1d', 'pre')], T.loc[('M0_VF_logit', 'regime-matched', 'pre')]
check('M0 observed / expected: short shift vs regime-matched', f'{s0.observed / s0.null_mean:.1f} vs {r0.observed / r0.null_mean:.1f}', '1.2 vs 1.6')
check('M0 short-shift yield; alerts per extra warning; p', f'{ys["M0_VF_logit"]:.2f}; {s0.alerts / (s0.observed - s0.null_mean):.0f}; {s0.p_one_sided:.2f}', '0.24; 415; 0.15')
si = T.loc[('M0_VF_logit', 'joint 4h-1d', 'post')]
check('M0 short-shift in-progress yield, per 100', f'{per100(si.observed - si.null_mean, si.alerts):.1f}', '1.9')
check('in-progress beats chance for every system (p <= 0.001)',
      str(all(T.loc[(m, 'joint 4h-1d', 'post')].p_one_sided <= 0.001 for m in ys)), 'True')

print('\nM1 against M0')
check('M1 pre-onset, alerts, yield over regime-matched null', f'{m1.pre}, {m1.alerts:,}, {per100(m1.pre - m1.pre_null_mean_regime, m1.alerts):.1f}', '71, 4,065, 0.7')
C = pd.read_csv(N17 / 'H17_per_coin_counts.csv').pivot(index='coin', columns='method', values='pre')
check('coins where M1 warned at least as many events as M0', f'{int((C.M1_vol_threshold >= C.M0_VF_logit).sum())} of {len(C)}', '9 of 10')
check('pre-onset warnings in all coins, M1 against M0', f'{C.M1_vol_threshold.sum()} against {C.M0_VF_logit.sum()}', '71 against 67')
P = pd.read_csv(N37 / '7_paired_pre_onset.csv').set_index(['a', 'b']).loc[('M0', 'M1')]
check('both / M0 alone / M1 alone / same minute among both', f'{P.both:.0f} / {P.only_a:.0f} / {P.only_b:.0f} / {P.same_alert_minute_when_both:.0f}', '61 / 6 / 10 / 36')

print('\nMarket episodes (breaches at most four hours apart, linked across coins)')
br = []
for coin in C.index:
    e = pd.read_csv(N17 / 'phase2' / f'{coin}_d4_events.csv')
    br += [(t, coin) for t in pd.to_datetime(e.breach_utc, utc=True)]
br.sort()
t = np.array([(b - pd.Timestamp(0, tz='UTC')) // pd.Timedelta(minutes=1) for b, _ in br])
S37 = __import__('json').loads((N37 / '37_SUMMARY.json').read_text())
for gap, want_ep, want_shared in ((240, 419, 721), (60, 548, 571)):
    ep = np.concatenate([[0], np.cumsum(np.diff(t) > gap)])           # new episode after a gap of more than `gap` min
    coins_in_ep = pd.DataFrame(dict(ep=ep, coin=[c for _, c in br])).groupby('ep').coin.nunique()
    check(f'linkage {gap} min: episodes (notebook 37 stored: {S37["market_episodes"][str(gap)]}); events sharing an episode with another coin',
          f'{ep.max() + 1}; {int((coins_in_ep.loc[ep] >= 2).sum())} of {len(t)}', f'{want_ep}; {want_shared} of 943')
E = pd.read_csv(N37 / '4_episode_level_test.csv').set_index(['method', 'null'])
er, ec = E.loc[('M0', 'regime-matched')], E.loc[('M0', 'joint circular')]
check('M0 episodes warned vs expected (regime-matched), p; joint circular p',
      f'{er.observed_episodes_with_pre:.0f} vs {er.null_mean:.1f}, p = {er.p_one_sided:.2f}; {ec.p_one_sided:.2f}', '46 vs 38.2, p = 0.09; 0.03')
B = pd.read_csv(N37 / '5_cluster_bootstrap.csv').set_index(['method', 'null', 'endpoint'])


def boot(k):
    r = B.loc[k]
    return f'{r.excess_per_100_alerts:.2f} [{r.lo95:.2f}, {r.hi95:.2f}]'


check('episode bootstrap, pre-onset, regime-matched', boot(('M0', 'regime-matched', 'pre')), '0.61 [0.12, 1.20]')
check('episode bootstrap, pre-onset, short shift', boot(('M0', 'joint 4h-1d', 'pre')), '0.24 [-0.19, 0.77]')
check('episode bootstrap, in-progress, regime-matched', boot(('M0', 'regime-matched', 'post')), '2.23 [1.54, 2.97]')

print('\nThe same coins over the development period (notebook 43)')
Q = pd.read_csv(N43 / '2_pooled.csv').set_index(['H', 'score', 'quantity'])
S43 = __import__('json').loads((N43 / '43_SUMMARY.json').read_text())['readings']['A2'][1]


def pt(k, f='{:+.3f}'):
    r = Q.loc[k]
    return f'{f.format(r.point)} [{f.format(r.lo)}, {f.format(r.hi)}]'


check('M0 in-progress share of above-chance AUC, 4 h', pt((240, 'M0_VF_logit', 'S_in_pooled'), '{:.1%}'), '69.2% [63.4%, 75.2%]')
check('24-hour events with a pre-onset minute', f'{S43["F"]:.1%} of {S43["events"]:,}', '11.0% of 1,623')
check('GBM - M0, standard, 24 h', pt((1440, 'GBM-M0', 'auc_standard')), '+0.038 [+0.016, +0.062]')
PC = pd.read_csv(N43 / '1_per_coin.csv')
pc = PC[(PC.H == 1440) & (PC.score == 'GBM-M0') & (PC.quantity == 'auc_standard')].point
check('GBM - M0, standard, 24 h, range by coin', f'{pc.min():+.3f} to {pc.max():+.3f}', '-0.177 to +0.207')
check('GBM - M0, pre-onset, 24 h', pt((1440, 'GBM-M0', 'auc_pre')), '-0.037 [-0.085, +0.020]')
check('GBM - M0, standard, 4 h', pt((240, 'GBM-M0', 'auc_standard')), '-0.036 [-0.067, -0.005]')
print(f'\n{sum(RESULTS)} of {len(RESULTS)} values match the text.')
