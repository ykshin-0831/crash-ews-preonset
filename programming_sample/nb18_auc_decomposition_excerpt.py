# Programming sample, page 1: cells [1], [2] and [4], exactly as printed.
# Excerpt of notebooks/18_BTC_Standard_Protocol_Inflation_SPOT.ipynb (tidied). It needs the
# one-minute Binance cache on Google Drive and helpers that are not shown (load_btc_spot,
# in_period, cell [3] SCORES), so it does not run on its own; the full notebook is in notebooks/.
# I give the printed outputs as comments after each cell.

# In [1]:
# https://github.com/ykshin-0831/crash-ews-preonset
# notebooks/18_BTC_Standard_Protocol_Inflation_SPOT.ipynb (excerpt, tidied; outputs from Colab)
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from sklearn.metrics import roc_auc_score
from google.colab import drive                   # runs on Colab with Google Drive
drive.mount('/content/drive')
SPOT_BASE = Path('/content/drive/MyDrive/btc_lead_cost_results_77months_spot')
NS = 60 * 10**9                     # nanoseconds per minute (timestamps are int64 ns)
LEAD_MIN, LEAD_MAX, RV_WINDOW = 30, 240, 1440   # warning lead 30-240 min; 24h volatility
HORIZON, EVAL_START, EVAL_END = LEAD_MAX, '2023-01-01', '2026-05-01'
DEFS, PRIMARY_DEF = {'4pct': 0.04}, '4pct'  # crash: 4% below 4h high (notebook: 3%, 5% too)
METHODS = ['M0_VF_logit', 'M1_vol_threshold', 'M2_VPIN', 'M3_GBM_rich']
N_BOOT, SEED, BLOCK_DAYS, STUDY = 500, 20261101, 7, (-480, 240)
def ts(s):
    return pd.Timestamp(s, tz='UTC')
EVAL_LAST = ts(EVAL_END).value - (HORIZON + 5) * NS  # leave room for the 240-min look-ahead
def load_catalog(path):
    events = pd.read_csv(path)
    for col in ['peak_utc', 'onset_1pct_utc', 'breach_utc', 'confirmed_utc']:
        events[col] = pd.to_datetime(events[col], utc=True)  # ISO text -> datetime64[ns, UTC]
    events = events.sort_values('breach_utc').reset_index(drop=True)
    assert events.event_id.is_unique
    return events
# Out:
# Mounted at /content/drive

# In [2]:
t_start = time.time()
px, sg = load_btc_spot()        # 1-minute prices and signals (helpers not shown)
TIMES = px.index.asi8           # int64 ns timestamp of every minute
close, volume = px.close.to_numpy(float), px.volume.to_numpy(float)
z_vol, tsr, z_hs = (sg[c].to_numpy(float) for c in ['z_vol', 'tsr', 'z_hs'])
VALID = (np.isfinite(close) & (close > 0) & np.isfinite(volume) & (volume > 0)
         & np.isfinite(z_vol) & np.isfinite(z_hs) & (tsr >= 0) & (tsr <= 1))
CAL = VALID & in_period(TIMES, '2022-01-01', '2023-01-01')     # 2022: percentile reference
EVM = VALID & (TIMES >= ts(EVAL_START).value) & (TIMES < EVAL_LAST)    # evaluation minutes
# int codes per minute: 24h-vol decile (0-9) and 7-day block (bootstrap unit, cell [4])
log_close = np.log(np.where(np.isfinite(close) & (close > 0), close, np.nan))
RV = pd.Series(log_close).diff().rolling(RV_WINDOW, min_periods=RV_WINDOW // 2).std().to_numpy()
DEC = np.full(len(TIMES), -1, np.int64)                  # -1 = not an evaluation minute
has_rv = EVM & np.isfinite(RV)
DEC[has_rv] = np.minimum((pd.Series(RV[has_rv]).rank(pct=True).to_numpy() * 10).astype(int), 9)
WEEK = np.where(EVM, (TIMES - ts(EVAL_START).value) // (BLOCK_DAYS * 1440 * NS),
                -1).astype(np.int64)
N_WEEKS = int(WEEK[EVM].max()) + 1
print(f'evaluation minutes {int(EVM.sum()):,}, {N_WEEKS} blocks of {BLOCK_DAYS} days; '
      f'features ready in {time.time() - t_start:.0f}s')
# Crash catalog, one row per crash: peak, 1% onset, 4% breach. The notebook rebuilds it
# from prices and stops unless it equals this CSV, so the CSV is used directly here.
CATS = {PRIMARY_DEF: load_catalog(SPOT_BASE / 'event_definition_eligibility_audit'
                                  / 'price_only_event_catalog.csv')}
def labels_for(dname):
    """y = 1 if a breach follows 30-240 min after t; in decline = past its 1% onset"""
    events = CATS[dname]
    breach = events.breach_utc.astype('int64').to_numpy()       # datetime64 -> int64 ns
    onset = events.onset_1pct_utc.astype('int64').to_numpy()
    nxt = np.searchsorted(breach, TIMES + LEAD_MIN * NS, 'left')  # next breach >= t + 30 min
    ok = nxt < len(breach)
    y, in_decline = np.zeros(len(TIMES), np.int8), np.zeros(len(TIMES), bool)
    y[ok] = breach[nxt[ok]] <= TIMES[ok] + LEAD_MAX * NS
    in_decline[ok] = TIMES[ok] >= onset[nxt[ok]]
    return dict(L1_event=y), dict(event=in_decline & (y == 1))
LAB = {dname: labels_for(dname) for dname in DEFS}
# Out:
# evaluation minutes 1,750,612, 174 blocks of 7 days; features ready in 13s

# In [4]:
# 7-day block bootstrap of the AUC without re-ranking 1.75M scores 500 times: count the
# correctly ordered (positive, negative) pairs once per pair of blocks (k, l) into C. A
# draw only changes block multiplicities w, so its AUC is w'Cw / ((w'P)(w'N)).
def week_pairs(sp, wp, sn, wn):
    """C[k, l] = over positives in block k, number of negatives in block l scored lower
    (+0.5 per tie); w = 1 (every block once) gives the ordinary AUC."""
    C = np.zeros((N_WEEKS, N_WEEKS))
    if len(sp) == 0 or len(sn) == 0:
        return C
    order = np.lexsort((sn, wn))
    sn, wn = sn[order], wn[order]                     # negatives sorted by (block, score)
    bounds = np.searchsorted(wn, np.arange(N_WEEKS + 1), 'left')   # block l: [a, b)
    for l in range(N_WEEKS):
        a, b = bounds[l], bounds[l + 1]
        if a == b:
            continue
        seg = sn[a:b]          # sorted scores of the negatives in block l
        # for every positive: negatives below it (lo) and tied with it (hi - lo)
        lo, hi = np.searchsorted(seg, sp, 'left'), np.searchsorted(seg, sp, 'right')
        C[:, l] = np.bincount(wp, weights=lo + 0.5 * (hi - lo), minlength=N_WEEKS)
    return C
def _ratio(a, b):
    return a / b if b > 0 else np.nan
STAT_KEYS = ['auc_standard', 'auc_pre', 'auc_in', 'share_in', 'in_share_of_excess',
             'auc_pre_within_regime']
def make_parts(sm, code, wk, dec):     # code: 0 negative, 1 pre-onset, 2 in-decline
    neg, pre, inn = code == 0, code == 1, code == 2
    def per_block(mask):                          # how many minutes in each 7-day block
        return np.bincount(wk[mask], minlength=N_WEEKS).astype(float)
    parts = dict(C_pre=week_pairs(sm[pre], wk[pre], sm[neg], wk[neg]),
                 C_in=week_pairs(sm[inn], wk[inn], sm[neg], wk[neg]),
                 P_pre=per_block(pre), P_in=per_block(inn), N=per_block(neg),
                 Cs_pre=np.zeros((N_WEEKS, N_WEEKS)), Pq_pre=[], Nq=[])
    for q in range(10):        # within regime: only pairs in the same volatility decile
        neg_q, pre_q = neg & (dec == q), pre & (dec == q)
        parts['Cs_pre'] += week_pairs(sm[pre_q], wk[pre_q], sm[neg_q], wk[neg_q])
        parts['Pq_pre'].append(per_block(pre_q))
        parts['Nq'].append(per_block(neg_q))
    parts['Pq_pre'], parts['Nq'] = np.array(parts['Pq_pre']), np.array(parts['Nq'])
    return parts
def stats_from(parts, w):                  # block weights w (w = 1: point estimate)
    def quad(M):                                       # w'Mw
        return w @ M @ w
    w_neg, w_pre, w_in = w @ parts['N'], w @ parts['P_pre'], w @ parts['P_in']
    ord_pre, ord_in = quad(parts['C_pre']), quad(parts['C_in'])   # ordered pairs
    std = _ratio(ord_pre + ord_in, (w_pre + w_in) * w_neg)
    pre, inn = _ratio(ord_pre, w_pre * w_neg), _ratio(ord_in, w_in * w_neg)
    pi = _ratio(w_in, w_pre + w_in)            # share of positives already in decline
    nq, pq = parts['Nq'] @ w, parts['Pq_pre'] @ w
    # AUC_std = (1-pi) AUC_pre + pi AUC_in; in-decline part of its excess over 0.5:
    return dict(auc_standard=std, auc_pre=pre, auc_in=inn, share_in=pi,
                in_share_of_excess=(_ratio(pi * (inn - 0.5), std - 0.5)
                                    if np.isfinite(inn) else 0.0),
                auc_pre_within_regime=_ratio(quad(parts['Cs_pre']), (pq * nq).sum()))
def evaluate(s, y, in_decline, base, seed, guard=False):
    m = base & np.isfinite(s) & (DEC >= 0)
    code = np.where(y[m] == 1, np.where(in_decline[m], 2, 1), 0).astype(np.int64)
    sm, wk, dec = s[m], WEEK[m], DEC[m]
    parts = make_parts(sm, code, wk, dec)
    point = stats_from(parts, np.ones(N_WEEKS))
    yy, keep_pre = (code > 0).astype(int), code != 2      # keep_pre: drop in-decline
    if guard:      # first call: the pair counts must reproduce sklearn and the identity
        a1, a2 = roc_auc_score(yy, sm), roc_auc_score(yy[keep_pre], sm[keep_pre])
        num = den = 0.0
        for q in range(10):        # within-regime AUC = pair-weighted mean over deciles
            in_q = (dec == q) & keep_pre
            npq, nnq = int(yy[in_q].sum()), int((yy[in_q] == 0).sum())
            if npq and nnq:
                num += roc_auc_score(yy[in_q], sm[in_q]) * npq * nnq
                den += npq * nnq
        if not (abs(a1 - point['auc_standard']) < 1e-9
                and abs(a2 - point['auc_pre']) < 1e-9
                and abs(num / den - point['auc_pre_within_regime']) < 1e-9):
            raise RuntimeError(f'GUARD FAIL: AUC differs from sklearn ({a1}, {a2})')
        pi = point['share_in']
        if abs((1 - pi) * point['auc_pre'] + pi * point['auc_in']
               - point['auc_standard']) > 1e-9:
            raise RuntimeError('GUARD FAIL: AUC decomposition identity')
        print('PASS: AUC machinery equals sklearn (standard, pre-onset, within-regime) '
              'and the decomposition identity holds')
    rng, boot = np.random.default_rng(seed), {k: np.empty(N_BOOT) for k in STAT_KEYS}
    for b in range(N_BOOT):        # draw 174 blocks with replacement -> multiplicities
        w = np.bincount(rng.integers(0, N_WEEKS, N_WEEKS), minlength=N_WEEKS)
        r = stats_from(parts, w.astype(float))
        for k in STAT_KEYS:
            boot[k][b] = r[k]
    for k in STAT_KEYS:                                  # 95% percentile intervals
        lo, hi = (np.nanpercentile(boot[k], [2.5, 97.5]) if np.isfinite(boot[k]).any()
                  else (np.nan, np.nan))
        point[f'{k}_lo'], point[f'{k}_hi'] = float(lo), float(hi)
    return point
# SCORES[(label, definition)][method]: per-minute scores of M0-M3, cell [3] (not shown)
rows, first, ci = [], True, 0
for dname in DEFS:
    lab, sc = 'L1_event', SCORES[('L1_event', dname)]
    Y, IN = LAB[dname]
    for split in ['event']:              # (the notebook also splits by price state)
        for mname in METHODS:
            r = evaluate(sc[mname], Y[lab], IN[split], EVM, SEED + ci, guard=first)
            first, ci = False, ci + 1
            rows.append(dict(label=lab, split=split, definition=dname, method=mname, **r))
auc_table = pd.DataFrame(rows)
display(auc_table[auc_table.definition == PRIMARY_DEF]
        .set_index(['label', 'split', 'method'])[STAT_KEYS].round(3))
# Reading: for M0, M1, M3 the standard AUC 0.76 falls to 0.65-0.67 on pre-onset positives
# (0.55-0.59 within a vol decile); the 41% of positives already in decline account for
# 61-65% of the AUC above 0.5.
# Out:
# PASS: AUC machinery equals sklearn (standard, pre-onset, within-regime) and the decomposition identity
# holds
#                                   auc_standard  auc_pre  auc_in  share_in  in_share_of_excess  auc_pre_within_regime
# label    split method
# L1_event event M0_VF_logit               0.763    0.673   0.891     0.414               0.615                  0.592
#                M1_vol_threshold          0.764    0.674   0.891     0.414               0.613                  0.594
#                M2_VPIN                   0.456    0.436   0.484     0.414                 NaN                  0.464
#                M3_GBM_rich               0.755    0.654   0.899     0.414               0.646                  0.547
