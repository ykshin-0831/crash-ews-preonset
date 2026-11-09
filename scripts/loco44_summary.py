"""Table B.19 (Online Appendix B.12): the pooled altcoin values when one coin is left out.

Reads notebook 44's stored output and notebook 43's per-coin values; recomputes nothing.
  results/altcoin_dev_period_loco_v1/44_LOCO_all_metrics.csv, 44_SUMMARY.json   (notebook 44)
  results/altcoin_dev_period_v1/1_per_coin.csv                                   (notebook 43, "range by coin")
Prints the table and compares each cell with the appendix (Online Appendix B, Table B.19).
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
LOCO = pd.read_csv(ROOT / 'results' / 'altcoin_dev_period_loco_v1' / '44_LOCO_all_metrics.csv')
S44 = json.loads((ROOT / 'results' / 'altcoin_dev_period_loco_v1' / '44_SUMMARY.json').read_text())
PC = pd.read_csv(ROOT / 'results' / 'altcoin_dev_period_v1' / '1_per_coin.csv')

# (row label, H, model_difference, metric, per-coin quantity or None, cell values as printed in Table B.19)
ROWS = [
    ('M0 in-progress share, 4 h', 240, 'M0', 'S_in_pooled', None,
     ('69.2% [63.4%, 75.2%]', '68.5% to 70.3%', '10 (lowest lower bound 62.0%)', '---')),
    ('GBM - M0, standard, 4 h', 240, 'GBM-M0', 'auc_standard', 'auc_standard',
     ('-0.036 [-0.067, -0.005]', '-0.049 to -0.022', '5 (not without ETH, BNB, ADA, ETC or NEO)', '-0.160 to +0.082')),
    ('GBM - M0, pre-onset, 4 h', 240, 'GBM-M0', 'auc_pre', 'auc_pre',
     ('-0.019 [-0.055, +0.020]', '-0.033 to -0.006', '0', '-0.137 to +0.104')),
    ('GBM - M0, in-progress, 4 h', 240, 'GBM-M0', 'auc_in', 'auc_in',
     ('-0.050 [-0.081, -0.019]', '-0.063 to -0.034', '10', '-0.190 to +0.072')),
    ('GBM - M0, standard, 24 h', 1440, 'GBM-M0', 'auc_standard', 'auc_standard',
     ('+0.038 [+0.016, +0.062]', '+0.019 to +0.062', '9 (not without EOS: +0.019 [-0.004, +0.042])', '-0.177 to +0.207')),
    ('GBM - M0, pre-onset, 24 h', 1440, 'GBM-M0', 'auc_pre', 'auc_pre',
     ('-0.037 [-0.085, +0.020]', '-0.047 to -0.021', '0', '-0.183 to +0.057')),
    ('GBM - M0, in-progress, 24 h', 1440, 'GBM-M0', 'auc_in', 'auc_in',
     ('+0.044 [+0.022, +0.069]', '+0.024 to +0.069', '10 (without EOS: +0.024 [+0.0003, +0.047])', '-0.179 to +0.230')),
    ('GBM - M0, pre-onset within regime, 24 h', 1440, 'GBM-M0', 'auc_pre_within_regime', 'auc_pre_within_regime',
     ('-0.051 [-0.112, +0.020]', '-0.061 to -0.039', '0', '-0.160 to +0.043')),
]


def f(x, pct=False):
    if pct:
        return f'{x:.1%}'
    s = f'{x:+.3f}'
    return f'{x:+.4f}' if s in ('+0.000', '-0.000') else s     # the table prints +0.0003


def ci(r, pct=False):
    return f'{f(r.point, pct)} [{f(r.ci_lo, pct)}, {f(r.ci_hi, pct)}]'


n_ok = n_all = 0
print('Table B.19. Ten altcoins, development period: the pooled values when one coin is left out')
for label, H, md, metric, q, stated in ROWS:
    d = LOCO[(LOCO.H == H) & (LOCO.model_difference == md) & (LOCO.metric == metric)]
    allc, lo1 = d[d.left_out == 'NONE'].iloc[0], d[d.left_out != 'NONE']
    assert len(lo1) == 10 and (lo1.n_coins == 9).all()
    pct = metric == 'S_in_pooled'
    excl = lo1[(lo1.ci_lo > 0) | (lo1.ci_hi < 0)]
    keep = lo1[~lo1.index.isin(excl.index)]
    if pct:
        n_excl = f'{len(excl)} (lowest lower bound {f(lo1.ci_lo.min(), True)})'
    elif len(excl) == 10 and H == 1440:                      # name the leave-one-out row with the lowest lower bound
        w = lo1.loc[lo1.ci_lo.idxmin()]
        n_excl = f'10 (without {w.left_out[:-4]}: {ci(w)})'
    elif len(keep) == 1:
        n_excl = f'{len(excl)} (not without {keep.left_out.iloc[0][:-4]}: {ci(keep.iloc[0])})'
    elif 0 < len(keep) < 10:
        names = [c[:-4] for c in keep.left_out]
        n_excl = f'{len(excl)} (not without {", ".join(names[:-1])} or {names[-1]})'
    else:
        n_excl = str(len(excl))
    by_coin = '---'
    if q:
        p = PC[(PC.H == H) & (PC.score == 'GBM-M0') & (PC.quantity == q)].point
        assert len(p) == 10
        by_coin = f'{f(p.min())} to {f(p.max())}'
    got = (ci(allc, pct), f'{f(lo1.point.min(), pct)} to {f(lo1.point.max(), pct)}', n_excl, by_coin)
    print(f'\n  {label}')
    for name, g, s in zip(('all ten coins', 'range of leave-one-out means', 'intervals excluding zero, of 10', 'range by coin'), got, stated):
        n_all += 1; n_ok += g == s
        print(f'    {name:<32} {g:<46} [B.19: {s}] {"MATCH" if g == s else "DIFFERS"}')
    if not pct and len(excl):
        print(f'    (excluding intervals lie {"below" if (excl.ci_hi < 0).all() else "above" if (excl.ci_lo > 0).all() else "on both sides of"} zero)')

pre = LOCO[(LOCO.metric == 'auc_pre') & (LOCO.left_out != 'NONE')]
print(f'\n  pre-onset differences negative in every leave-one-out mean (both definitions): {bool((pre.point < 0).all())}')
print(f'  notebook 44 reproduced notebook 43 in {S44["verified_43_reference_checks"]} checks (appendix: 169); bootstrap draws {S44["n_bootstrap_draws"]}')
print(f'\n{n_ok} of {n_all} cells match Table B.19.')
