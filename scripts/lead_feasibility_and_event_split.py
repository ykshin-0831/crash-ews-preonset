"""Lead-time bound, budget check (Appendix A) and event split (Table 7), from stored files only.

Reproduces
  * N F(l), the number of development events that a rule can warn at least l minutes
    before onset (Appendix A, "A lead requirement"), l = 1 (the bound F), 30, 60, 120;
  * the budget check of Appendix A: how many pre-onset sets touch one calendar month,
    how many events fall in one calendar year, and the oracle's attainable count under
    monthly caps of 10, 20 and 40 (annual cap twelve times the monthly cap);
  * the "other events" rows of Table 7 as exact complements of the stored all-event and
    common-event rows (with the negatives fixed, an AUC over a union of positive sets is
    the positive-weighted average of the AUCs over its parts).

Reads
  results/peak_window_v1/1_events_H{120,240,480,1440}.csv     (notebook 22, development events)
  results/robustness_checks_v1/1_pre_onset_possible_by_window.csv,
          1b_events_pre_onset_minutes.csv, 3_common_events.csv,
          3b_common_event_pairs.csv                              (notebook 32)
  results/shapley_decomposition_v1/2_decomposition_bootstrap.csv (notebook 33, Table 6 intervals)
  results/complement_fragility_v1/2_subset_intervals.csv         (notebook 41, intervals of the
                                                                   "other" rows; also a direct check)
Writes
  scripts/output/1_lead_feasibility.csv, 2_budget_check.csv, 3_event_split.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
OUT = ROOT / 'scripts' / 'output'
W_MIN, W_MAX, R = 30, 240, 1440
NW = W_MAX - W_MIN + 1                     # 211 window positions
LEADS = (1, 30, 60, 120)
CAPS = (10, 20, 40)                        # monthly caps I use in the paper (20 = frozen scheduler)
COOLDOWNS = (60, 120, 240)                 # cooldowns I use (60-240 minutes)


def minutes(col):
    """UTC timestamps -> integer minutes since 1970 (independent of the pandas time resolution)."""
    t = pd.to_datetime(col, utc=True)
    return ((t - pd.Timestamp(0, tz='UTC')) // pd.Timedelta(minutes=1)).to_numpy(np.int64)


def month_of(m):
    return pd.DatetimeIndex(pd.to_datetime(np.asarray(m) * 60, unit='s', utc=True)).strftime('%Y-%m').to_numpy()


def load(H):
    E = pd.read_csv(RES / 'peak_window_v1' / f'1_events_H{H}.csv')
    E['pk'], E['on'], E['br'] = minutes(E.peak_utc), minutes(E.onset_1pct_utc), minutes(E.breach_utc)
    E['g'] = E.br - E.on
    assert (E.g == E.onset_to_breach_minutes).all() and (E.pk < E.on).all() and (E.on <= E.br).all()
    assert (np.diff(E.br) >= R).all(), 'refractory period violated'
    E['L'] = np.maximum(0, np.minimum(NW, W_MAX - E.g))             # Lemma 2
    return E.sort_values('br').reset_index(drop=True)


def check(label, got, want, tol=0.0):
    ok = abs(got - want) <= tol if isinstance(want, float) else got == want
    print(f'  {label}: {got} (stated/stored {want}) {"MATCH" if ok else "DIFFERS"}')
    return ok


EV = {H: load(H) for H in (120, 240, 480, 1440)}
S32 = pd.read_csv(RES / 'robustness_checks_v1' / '1_pre_onset_possible_by_window.csv').set_index('H')
L32 = pd.read_csv(RES / 'robustness_checks_v1' / '1b_events_pre_onset_minutes.csv')

# ---------------------------------------------------------------- 1. lead-time bound
print('1. Lead-time bound N F(l) = #{e : g_e <= W_max - l}')
rows = []
for H, E in EV.items():
    ref = L32[L32.H == H].set_index('event_id').pre_onset_possible_minutes
    assert np.array_equal(E.set_index('event_id').L.loc[ref.index].to_numpy(), ref.to_numpy()), 'L_e differs from notebook 32'
    pi = 1 - E.L.sum() / (len(E) * NW)                                # Lemma 3
    assert abs(pi - S32.loc[H, 'pi_share_in_minutes']) < 1e-12
    for l in LEADS:
        Ll = np.maximum(0, np.minimum(NW, W_MAX - E.g - l + 1))
        nf = int((E.g <= W_MAX - l).sum())
        assert nf == int((Ll > 0).sum())
        rows.append(dict(H=H, events=len(E), lead_min=l, NF=nf, F=nf / len(E), pre_onset_minutes=int(Ll.sum()),
                         pi=pi, median_gap=float(E.g.median()), max_gap=int(E.g.max())))
LF = pd.DataFrame(rows)
LF.to_csv(OUT / '1_lead_feasibility.csv', index=False)
print(LF.pivot(index='H', columns='lead_min', values='NF').assign(events=LF.groupby('H').events.first()).to_string())
nf = LF.set_index(['H', 'lead_min']).NF
for H, want in ((240, (81, 70, 43)), (1440, (29, 25, 18))):
    check(f'H={H}: NF(30), NF(60), NF(120) of {len(EV[H])}', tuple(int(nf[H, l]) for l in (30, 60, 120)), want)
check('H=1440: F = NF(1)/N', f'{int(nf[1440, 1])}/218 = {nf[1440, 1] / 218:.3f}', '30/218 = 0.138')

# ---------------------------------------------------------------- 2. budget check
print('\n2. Budget check (Appendix A)')
C3 = pd.read_csv(RES / 'robustness_checks_v1' / '3_common_events.csv')
brows = []
for H in (240, 1440):
    E = EV[H]
    has = E.L > 0
    lo = E.br - W_MAX                                                 # far edge of the window
    hi = np.minimum(E.br - W_MIN, E.on - 1)                           # last pre-onset window minute
    touch = pd.Series([m for a, b in zip(month_of(lo[has]), month_of(hi[has])) for m in {a, b}]).value_counts()
    per_year = pd.Series(month_of(E.br)).str[:4].value_counts()       # all events, by breach year
    # oracle: one alert per non-empty S_e at its far edge (pre-onset because every window minute is valid)
    a = lo[has].to_numpy()
    assigned_m = pd.Series(month_of(a)).value_counts()
    assigned_y = pd.Series(month_of(a)).str[:4].value_counts()
    spacing = int(np.diff(a).min())
    positives = int(C3[(C3.H == H) & (C3.subset == 'all events')].positives.iloc[0])
    print(f' H={H}: events {len(E)}, with a pre-onset set {int(has.sum())}; positives {positives} = {len(E)} x {NW}: '
          f'{positives == len(E) * NW}; far-edge alerts at least {spacing} min apart')
    check(f'H={H}: most pre-onset sets touching one calendar month', int(touch.max()), {240: 7, 1440: 4}[H])
    check(f'H={H}: most events in one calendar year', int(per_year.max()), {240: 36, 1440: 79}[H])
    for cap in CAPS:
        fits = assigned_m.max() <= cap and assigned_y.max() <= 12 * cap and all(c <= spacing for c in COOLDOWNS)
        brows.append(dict(H=H, events=len(E), events_with_pre_onset_set=int(has.sum()),
                          max_sets_touching_one_month=int(touch.max()), max_events_in_one_year=int(per_year.max()),
                          monthly_cap=cap, annual_cap=12 * cap, max_oracle_alerts_one_month=int(assigned_m.max()),
                          max_oracle_alerts_one_year=int(assigned_y.max()), min_oracle_alert_spacing_min=spacing,
                          cooldown_condition_c=f'C <= R - (W_max - W_min) = {R - (W_MAX - W_MIN)}',
                          cooldowns_satisfy_c=all(c <= R - (W_MAX - W_MIN) for c in COOLDOWNS),
                          every_window_minute_valid=positives == len(E) * NW,
                          oracle_attainable=int(has.sum()) if fits else np.nan, NF=int(has.sum()),
                          budget_binds=not fits))
BC = pd.DataFrame(brows)
BC.to_csv(OUT / '2_budget_check.csv', index=False)
print(BC[['H', 'monthly_cap', 'annual_cap', 'max_oracle_alerts_one_month', 'max_oracle_alerts_one_year',
          'oracle_attainable', 'NF', 'budget_binds']].to_string(index=False))

# ---------------------------------------------------------------- 3. event split (Table 7)
print('\n3. Event split (Table 7)')
E4, E24 = EV[240], EV[1440]
d = np.abs(E4.br.to_numpy()[:, None] - E24.br.to_numpy()[None, :])
i4, i24 = np.nonzero(d <= 60)                                         # same event: breaches within 60 minutes
assert len(set(i4)) == len(i4) and len(set(i24)) == len(i24)
P = pd.read_csv(RES / 'robustness_checks_v1' / '3b_common_event_pairs.csv')
assert sorted(zip(E4.event_id[i4], E24.event_id[i24])) == sorted(zip(P.event_id_H240, P.event_id_H1440)), 'pairs differ from notebook 32'
check('common events (identical breach minute)', f'{len(i4)} ({int((d[i4, i24] == 0).sum())})', '52 (29)')
check('median onset shift of the common events, 24 h minus 4 h (min)', f'{np.median(E24.on.to_numpy()[i24] - E4.on.to_numpy()[i4]):.0f}', '-119')
check('feasible 24-hour events that are common', f'{int((E24.L.to_numpy()[i24] > 0).sum())} of {int((E24.L > 0).sum())}', '27 of 30')

S33 = pd.read_csv(RES / 'shapley_decomposition_v1' / '2_decomposition_bootstrap.csv')
S33 = S33[(S33.h1 == 240) & (S33.h2 == 1440)].set_index('term')
S41 = pd.read_csv(RES / 'complement_fragility_v1' / '2_subset_intervals.csv')
srows = []
for H, E, idx in ((240, E4, i4), (1440, E24, i24)):
    common = np.zeros(len(E), bool); common[idx] = True
    parts = {}
    for name, m in (('all events', np.ones(len(E), bool)), ('common events only', common), ('other events', ~common)):
        n = int(m.sum()); pos = n * NW; pre = int(E.L[m].sum())
        parts[name] = dict(H=H, subset=name, events=n, events_with_pre_onset_minutes=int((E.L[m] > 0).sum()),
                           positives=pos, pre_onset_positives=pre, pi=1 - pre / pos)
    c = C3[C3.H == H].set_index('subset')
    for name in ('all events', 'common events only'):
        assert parts[name]['positives'] == c.loc[name, 'positives'] and abs(parts[name]['pi'] - c.loc[name, 'share_in']) < 1e-12
        for k in ('m0_standard', 'gbm_standard', 'm0_pre', 'gbm_pre', 'diff_auc_standard', 'diff_auc_pre'):
            parts[name][k] = float(c.loc[name, k])
    a, b, o = parts['all events'], parts['common events only'], parts['other events']
    for k in ('m0_standard', 'gbm_standard', 'diff_auc_standard'):     # weights: positive minutes
        o[k] = (a['positives'] * a[k] - b['positives'] * b[k]) / o['positives']
    for k in ('m0_pre', 'gbm_pre', 'diff_auc_pre'):                    # weights: pre-onset positive minutes
        o[k] = (a['pre_onset_positives'] * a[k] - b['pre_onset_positives'] * b[k]) / o['pre_onset_positives']
    # intervals: all events = Table 6 (notebook 33), common = notebook 32, other = notebook 41
    t6 = {'diff_auc_standard': f'D_h{1 if H == 240 else 2}', 'diff_auc_pre': f'dpre_h{1 if H == 240 else 2}'}
    s41 = S41[(S41.H == H) & (S41.subset == 'other')].set_index('quantity')
    for k, q in (('diff_auc_standard', 'auc_standard'), ('diff_auc_pre', 'auc_pre')):
        a[k + '_lo'], a[k + '_hi'] = S33.loc[t6[k], 'lo'], S33.loc[t6[k], 'hi']
        b[k + '_lo'], b[k + '_hi'] = c.loc['common events only', k + '_lo'], c.loc['common events only', k + '_hi']
        o[k + '_lo'], o[k + '_hi'] = s41.loc[q, 'block_lo'], s41.loc[q, 'block_hi']
        o[k + '_nb41_direct'] = s41.loc[q, 'diff']
        assert abs(o[k] - s41.loc[q, 'diff']) < 1e-12, 'complement differs from notebook 41'
    srows += [a, b, o]
ES = pd.DataFrame(srows)
# Table 7 does not report a pre-onset AUC that rests on a handful of events (24 h, other events: 113 minutes, 3 events)
ES['pre_onset_reported'] = ES.events_with_pre_onset_minutes >= 10
ES.to_csv(OUT / '3_event_split.csv', index=False)


def ci(r, k):
    return f"{r[k]:+.3f} [{r[k + '_lo']:+.3f}, {r[k + '_hi']:+.3f}]"


print(f"  {'H':>5} {'subset':<19} {'positives':>9} {'pi':>6}  standard diff [95%]        pre-onset diff [95%]")
for _, r in ES.iterrows():
    pre = ci(r, 'diff_auc_pre') if r.pre_onset_reported else f'(not reported: {r.pre_onset_positives} minutes, {r.events_with_pre_onset_minutes} events)'
    print(f'  {r.H:>5} {r.subset:<19} {r.positives:>9,} {r.pi:6.3f}  {ci(r, "diff_auc_standard"):<26} {pre}')
o24 = ES[(ES.H == 1440) & (ES.subset == 'other events')].iloc[0]
o4 = ES[(ES.H == 240) & (ES.subset == 'other events')].iloc[0]
check('24-hour other events: GBM vs M0 standard AUC', f'{o24.gbm_standard:.3f} vs {o24.m0_standard:.3f}', '0.769 vs 0.686')
check('24-hour other events: diff (standard) [notebook 41 block interval]', ci(o24, 'diff_auc_standard'), '+0.083 [+0.056, +0.109]')
check('4-hour other events: diff (standard)', ci(o4, 'diff_auc_standard'), '+0.024 [-0.016, +0.067]')
check('4-hour other events: diff (pre-onset)', ci(o4, 'diff_auc_pre'), '+0.033 [-0.021, +0.090]')
check('pi of the other events (4 h, 24 h)', f'{o4.pi:.3f}, {o24.pi:.3f}', '0.508, 0.997')
check('24-hour pre-onset minutes dropped by the restriction', f'{o24.pre_onset_positives} of {int(EV[1440].L.sum())}', '113 of 3971')
print('Wrote', ', '.join(p.name for p in (OUT / '1_lead_feasibility.csv', OUT / '2_budget_check.csv', OUT / '3_event_split.csv')))
