"""Notebook 38 core: daily equity-index crashes, a pre-registered historical external validation.

Every function the analysis uses is in this file; the notebook only calls them. Its SHA-256 is part of
the frozen specification (38_SPEC.json), so the code that produced the results is the code that was registered.
No function writes index levels to any output: outputs carry dates, positions, counts and statistics only.
"""
import io
import json
import time
import hashlib
import urllib.request
import numpy as np
import pandas as pd

CORE_VERSION = '38-core-1'


# ----------------------------------------------------------------------------- data
def _yahoo(ticker, start, end):
    import yfinance as yf
    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False, threads=False)
    if df is None or len(df) == 0:
        return None
    col = df['Close']
    if isinstance(col, pd.DataFrame):
        col = col.iloc[:, 0]
    s = pd.Series(col.to_numpy(float), index=pd.to_datetime(df.index).tz_localize(None).normalize())
    return s


def _stooq(symbol, start, end):
    url = f'https://stooq.com/q/d/l/?s={symbol}&i=d'
    with urllib.request.urlopen(url, timeout=60) as r:
        raw = r.read().decode('utf-8', 'replace')
    if not raw.startswith('Date'):
        return None
    df = pd.read_csv(io.StringIO(raw))
    s = pd.Series(df['Close'].to_numpy(float), index=pd.to_datetime(df['Date']))
    return s[(s.index >= pd.Timestamp(start)) & (s.index < pd.Timestamp(end))]


def clean(s):
    """Drop missing or non-positive closes and duplicate dates (keep the last); sort by date."""
    s = pd.Series(s).dropna()
    s = s[s > 0]
    s = s[~s.index.duplicated(keep='last')].sort_index()
    s.index.name = 'date'
    return s.astype(float)


def coverage_ok(s, spec):
    pre = int((s.index < pd.Timestamp(spec['periods']['calibration'][0])).sum())
    ev = int(((s.index >= pd.Timestamp(spec['periods']['evaluation'][0])) & (s.index < pd.Timestamp(spec['periods']['evaluation'][1]))).sum())
    return pre >= spec['data']['min_days_before_calibration'] and ev >= spec['data']['min_days_in_evaluation'], pre, ev


def fetch_index(ticker, spec, log=print):
    """Yahoo first; if it fails or does not cover, Stooq for the whole series; never mixed."""
    start, end = spec['data']['download_start'], spec['data']['download_end']
    tried = []
    for source in ['yahoo', 'stooq']:
        try:
            if source == 'yahoo':
                s = _yahoo(ticker, start, end)
            else:
                sym = spec['data']['stooq_symbols'].get(ticker)
                s = _stooq(sym, start, end) if sym else None
        except Exception as e:  # network or parser failure
            tried.append(f'{source}: {type(e).__name__}: {str(e)[:80]}'); s = None
        if s is None or len(s) == 0:
            tried.append(f'{source}: no data'); continue
        s = clean(s)
        ok, pre, ev = coverage_ok(s, spec)
        if ok:
            return s, source, tried
        tried.append(f'{source}: coverage {pre} days before calibration, {ev} in evaluation')
    return None, None, tried


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


# ----------------------------------------------------------------------------- features
FEATS = ['z_vol', 'ret_1', 'ret_5', 'ret_20', 'ret_60', 'rv_5', 'rv_20', 'rv_60', 'rv_ratio_5_60',
         'dd_20', 'dd_60', 'dd_250', 'dist_ma200']


def features(close):
    """Causal daily features (data up to the close of day t)."""
    c = close.astype(float); lc = np.log(c); r = lc.diff()
    F = pd.DataFrame(index=close.index)
    for k in (5, 20, 60):
        F[f'rv_{k}'] = r.rolling(k, min_periods=k).std()
    lrv = np.log(F['rv_20'].where(F['rv_20'] > 0))
    base = lrv.rolling(250, min_periods=125)
    F['z_vol'] = (lrv - base.mean().shift(1)) / base.std().shift(1).replace(0, np.nan)
    for k in (1, 5, 20, 60):
        F[f'ret_{k}'] = lc - lc.shift(k)
    F['rv_ratio_5_60'] = F['rv_5'] / F['rv_60']
    for k in (20, 60, 250):
        F[f'dd_{k}'] = 1 - c / c.rolling(k, min_periods=k).max()
    F['dist_ma200'] = lc - np.log(c.rolling(200, min_periods=200).mean())
    F = F.replace([np.inf, -np.inf], np.nan)
    F['B0'] = F['rv_20']
    F['B1'] = F['dd_20']
    return F


# ----------------------------------------------------------------------------- events
def catalog(close, H, theta=0.10, refractory=20, onset_frac=0.25):
    """Breach: close <= (1 - theta) x max of the previous H closes (first day of a run; sustain 1).
    Peak: that maximum (first occurrence). Onset: first close after the peak <= (1 - onset_frac x theta) x peak.
    Next event: at least `refractory` trading days after the previous breach and a peak after it."""
    c = np.asarray(close, float); n = len(c)
    prior = pd.Series(c).rolling(H, min_periods=H).max().shift(1).to_numpy()
    raw = np.isfinite(prior) & (c <= (1 - theta) * prior)
    fresh = raw & ~np.r_[False, raw[:-1]]
    rows = []; last = -10**9
    for i in np.flatnonzero(fresh):
        if i - H < 0 or i - last < refractory:
            continue
        peak = int(i - H + np.argmax(c[i - H:i]))
        if peak <= last:
            continue
        cand = np.flatnonzero(c[peak + 1:i + 1] <= (1 - onset_frac * theta) * c[peak])
        onset = int(peak + 1 + cand[0])
        rows.append(dict(peak_pos=peak, onset_pos=onset, breach_pos=int(i)))
        last = i
    E = pd.DataFrame(rows, columns=['peak_pos', 'onset_pos', 'breach_pos'])
    idx = pd.DatetimeIndex(close.index)
    for k in ['peak', 'onset', 'breach']:
        E[f'{k}_date'] = idx[E[f'{k}_pos'].to_numpy()] if len(E) else pd.Series([], dtype='datetime64[ns]')
    E['onset_to_breach_days'] = E.breach_pos - E.onset_pos
    E['peak_to_breach_days'] = E.breach_pos - E.peak_pos
    return E


def window_labels(n, E, wmin=1, wmax=20):
    """Per-day labels: 0 negative, 1 pre-onset positive, 2 in-progress positive; plus the event index of each positive."""
    lab = np.zeros(n, np.int8); ev = np.full(n, -1, np.int64)
    for k, e in enumerate(E.itertuples(index=False)):
        lo, hi = e.breach_pos - wmax, e.breach_pos - wmin
        for t in range(max(lo, 0), min(hi, n - 1) + 1):
            lab[t] = 1 if t < e.onset_pos else 2; ev[t] = k
    return lab, ev


def feasibility(E, wmax, wmin=1):
    """Event e can be warned before onset iff some day a has wmin <= b - a <= wmax and a < onset."""
    L = np.minimum(E.breach_pos - wmin, E.onset_pos - 1) - (E.breach_pos - wmax) + 1
    return np.maximum(L, 0).to_numpy()


# ----------------------------------------------------------------------------- scheduler and calibration
def schedule(pos, dates, eligible, q_cap=1, y_cap=4, cooldown=20):
    """Calendar scheduler on trading-day positions: at most q_cap per calendar quarter, y_cap per calendar year,
    and at least `cooldown` trading days between alerts. Returns alert positions."""
    pos = np.asarray(pos, np.int64); el = np.asarray(eligible, bool)
    d = pd.DatetimeIndex(dates)
    yr = d.year.to_numpy(); qt = yr * 4 + (d.month.to_numpy() - 1) // 3
    out = []; last = -10**9; qc = {}; yc = {}
    for k in np.flatnonzero(el):
        p = int(pos[k])
        if p - last < cooldown or qc.get(qt[k], 0) >= q_cap or yc.get(yr[k], 0) >= y_cap:
            continue
        out.append(p); last = p; qc[qt[k]] = qc.get(qt[k], 0) + 1; yc[yr[k]] = yc.get(yr[k], 0) + 1
    return np.asarray(out, np.int64)


def score_grid(sc):
    Q = np.unique(np.r_[0., np.linspace(.10, .995, 33), .997, .999, .9999, 1.])
    return np.unique(np.r_[np.nextafter(sc.min(), -np.inf), np.quantile(sc, Q), np.nextafter(sc.max(), np.inf)])


def calibrate(pos, dates, scores, sched):
    """Crypto notebook-5 rule: the most alerts the budget allows on the calibration days, then the strictest threshold."""
    rows = []
    for th in score_grid(scores):
        e = scores >= th
        rows.append((float(th), len(schedule(pos, dates, e, **sched)), int(e.sum())))
    df = pd.DataFrame(rows, columns=['threshold', 'cal_alerts', 'raw_candidates'])
    best = df[df.cal_alerts == df.cal_alerts.max()].sort_values('raw_candidates').iloc[0]
    return float(best.threshold), int(best.cal_alerts)


# ----------------------------------------------------------------------------- matching
def match(E, alerts, wmin=1, wmax=20):
    """Per event: first alert with wmin <= b - a <= wmax (windows of different events do not overlap)."""
    a = np.sort(np.asarray(alerts, np.int64)); b = E.breach_pos.to_numpy(np.int64); o = E.onset_pos.to_numpy(np.int64)
    if len(a) == 0:
        z = np.zeros(len(b), bool); return z, z.copy(), np.full(len(b), -1, np.int64)
    lo = np.searchsorted(a, b - wmax, 'left'); hi = np.searchsorted(a, b - wmin, 'right')
    hit = hi > lo
    al = np.where(hit, a[np.minimum(lo, len(a) - 1)], -1)
    pre = hit & (al < o)
    return hit, pre, al


# ----------------------------------------------------------------------------- AUC
def auc_weighted(s, pos, w):
    """Mann-Whitney AUC of positives (pos True) over negatives with day weights w (ties count one half)."""
    s = np.asarray(s, float); pos = np.asarray(pos, bool); w = np.asarray(w, float)
    o = np.argsort(s, kind='mergesort'); s, pos, w = s[o], pos[o], w[o]
    _, g = np.unique(s, return_inverse=True)
    wp = np.bincount(g, w * pos); wn = np.bincount(g, w * ~pos)
    Wp, Wn = wp.sum(), wn.sum()
    if Wp <= 0 or Wn <= 0:
        return np.nan
    below = np.cumsum(wn) - wn
    return float((wp * (below + 0.5 * wn)).sum() / (Wp * Wn))


class AucPrep:
    """Pre-sorted structure for repeated weighted AUCs of one score on one index (positives vs negatives)."""

    def __init__(self, s, pos, keep, group):
        k = np.flatnonzero(keep)
        o = k[np.argsort(s[k], kind='mergesort')]
        self.pos = pos[o].astype(bool); self.group = group[o]
        _, self.g = np.unique(s[o], return_inverse=True); self.G = int(self.g.max()) + 1 if len(o) else 0

    def auc(self, wday):
        w = wday[self.group]
        wp = np.bincount(self.g, w * self.pos, minlength=self.G); wn = np.bincount(self.g, w * ~self.pos, minlength=self.G)
        Wp, Wn = wp.sum(), wn.sum()
        if Wp <= 0 or Wn <= 0:
            return np.nan
        return float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum() / (Wp * Wn))


class StratAucPrep:
    """Stratified (within-decile) AUC: pairs are compared only inside the same stratum."""

    def __init__(self, s, pos, keep, group, strat):
        self.parts = []
        for z in np.unique(strat[keep]):
            kz = keep & (strat == z)
            if (pos & kz).any() and (~pos & kz).any():
                self.parts.append(AucPrep(s, pos, kz, group))

    def auc(self, wday):
        num = den = 0.0
        for p in self.parts:
            w = wday[p.group]
            wp = np.bincount(p.g, w * p.pos, minlength=p.G); wn = np.bincount(p.g, w * ~p.pos, minlength=p.G)
            num += float((wp * (np.cumsum(wn) - wn + 0.5 * wn)).sum()); den += float(wp.sum() * wn.sum())
        return num / den if den > 0 else np.nan


# ----------------------------------------------------------------------------- nulls
def quarter_positions(eval_pos, quarter_of):
    """Map each calendar quarter to the evaluation positions in it (computed once per index)."""
    q = quarter_of[eval_pos]
    return {int(z): eval_pos[q == z] for z in np.unique(q)}


def null_budget_preserving(rng, alerts, qpos, quarter_of, decile_of, cooldown=20, stats=None):
    """Re-place each alert, in random order, within its own calendar quarter, at least `cooldown` days from its
    original position and from every alert already placed; own RV decile first, then other deciles by distance,
    then any eligible day of the quarter; keep the original only if nothing qualifies (notebook-26 logic, daily)."""
    a = np.asarray(alerts, np.int64); out = np.empty_like(a); placed = []
    for k in rng.permutation(len(a)):
        p0 = a[k]; q = int(quarter_of[p0]); d0 = decile_of[p0]
        cand = qpos[q]
        cand = cand[np.abs(cand - p0) >= cooldown]
        if placed:
            P = np.asarray(placed)
            cand = cand[np.all(np.abs(cand[:, None] - P[None, :]) >= cooldown, axis=1)]
        choice = p0; how = 'kept original'
        if len(cand):
            dc = decile_of[cand]
            for dist in range(10):
                pool = cand[np.abs(dc - d0) == dist]
                if len(pool):
                    choice = int(pool[rng.integers(len(pool))]); how = 'own decile' if dist == 0 else 'other decile'; break
        if stats is not None:
            stats[how] = stats.get(how, 0) + 1
        out[k] = choice; placed.append(choice)
    return np.sort(out)


def null_shift(alerts_by_index, dates_by_index, eval_span, delta_days):
    """Shift every index's alert dates by the same number of calendar days, circularly within the evaluation span,
    and snap each to the first trading day of that index on or after the shifted date (wrapping to the start)."""
    t0, t1 = (pd.Timestamp(x) for x in eval_span); span = (t1 - t0).days
    out = {}
    for s, a in alerts_by_index.items():
        idx = dates_by_index[s]; ev = np.flatnonzero((idx >= t0) & (idx < t1))
        d = idx[a]
        off = ((d - t0).days.to_numpy() + int(delta_days)) % span
        nd = t0 + pd.to_timedelta(off, unit='D')
        j = np.searchsorted(idx[ev].to_numpy(), nd.to_numpy(), 'left')
        j = np.where(j >= len(ev), 0, j)
        out[s] = np.sort(ev[j])
    return out


def holm(ps):
    ps = np.asarray(ps, float); order = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj


def episodes(breach_dates, gap_days):
    """Single linkage of breach dates across indices: a new episode starts when the gap exceeds gap_days."""
    d = pd.DatetimeIndex(breach_dates); o = np.argsort(d.values, kind='stable')
    gaps = np.r_[np.inf, np.diff(d.values[o]).astype('timedelta64[D]').astype(float)]
    ep = np.empty(len(d), np.int64); ep[o] = np.cumsum(gaps > gap_days) - 1
    return ep
