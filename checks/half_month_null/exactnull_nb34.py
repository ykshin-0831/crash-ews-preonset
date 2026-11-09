"""Exact conditional null for budget-constrained alerts (notebook 34 core).
Q_nu: uniform over alert sets A ⊂ positions with spacing >= C (time units) and exact counts nu[month, class].
Positions are sorted times; each has a month index, a class, an indicator h (pre-onset minute) and a weight w (1 = uniform).
Statistic T(A) = sum_{t in A} h(t) (additive; checked separately on the real data)."""
import numpy as np
from numba import njit

# ---------------------------------------------------------------- design preparation
def prepare(times, month, cls, h, nu, C, w=None):
    """Drop positions whose (month, class) has zero count (they can never hold an alert), build radix states per month.
    month: int array (any labels, increasing in time); cls: int in [0, K); nu: dict {(month_label, class): count}."""
    times = np.asarray(times, np.int64); month = np.asarray(month, np.int64); cls = np.asarray(cls, np.int64); h = np.asarray(h, np.int64)
    assert np.all(np.diff(times) > 0)
    w = np.ones(len(times)) if w is None else np.asarray(w, float)
    K = int(max(cls.max() + 1, max(b for _, b in nu) + 1)) if nu else int(cls.max() + 1)
    mlabels = sorted({m for (m, b), v in nu.items() if v > 0})
    mid = {m: k for k, m in enumerate(mlabels)}; M = len(mlabels)
    NU = np.zeros((M, K), np.int64)
    for (m, b), v in nu.items():
        if v > 0: NU[mid[m], b] = v
    ml = np.array(mlabels, np.int64); pos = np.searchsorted(ml, month); pos_c = np.clip(pos, 0, max(M - 1, 0))
    keep = (M > 0) & (pos < M) & (ml[pos_c] == month) if M else np.zeros(len(month), bool)
    if M: keep &= NU[pos_c, np.clip(cls, 0, K - 1)] > 0
    t = times[keep]; mo = pos_c[keep].astype(np.int64); c = cls[keep]; hh = h[keep]; ww = w[keep]
    if M and not np.all(np.diff(mo) >= 0): raise ValueError('months must be increasing in time')
    L = len(t)
    mstart = np.searchsorted(mo, np.arange(M), 'left'); mend = np.searchsorted(mo, np.arange(M), 'right')
    if np.any(mend <= mstart): raise ValueError('a month with a positive count has no admissible position')
    RAD = NU + 1; STR = np.zeros((M, K), np.int64); S = np.ones(M, np.int64); FULL = np.zeros(M, np.int64)
    for m in range(M):
        s = 1
        for b in range(K):
            if NU[m, b] > 0: STR[m, b] = s; s *= RAD[m, b]
        S[m] = s; FULL[m] = int((NU[m] * STR[m]).sum())
    st = STR[mo, c]; rad = RAD[mo, c]
    nx = np.searchsorted(t, t + C, 'left').astype(np.int64)
    R = int((nx - np.arange(L)).max()) + 1 if L else 1
    return dict(t=t, mon=mo, cls=c, h=hh, w=ww, st=st, rad=rad, nx=nx, S=S, FULL=FULL, mstart=mstart, mend=mend, M=M, L=L, R=R,
                NU=NU, keep=keep, mlabels=mlabels, n=int(NU.sum()))

# ---------------------------------------------------------------- backward pass
@njit(cache=False)
def _pairs(D, mode, hi):
    """(destination bin, source bin) pairs for multiplying by u^hi (mode 0, saturating) or (1 + x)^hi (mode 1, truncated)."""
    if hi == 0:
        P = np.empty((D, 2), np.int64)
        for k in range(D): P[k, 0] = k; P[k, 1] = k
        return P
    if mode == 0:
        P = np.empty((D, 2), np.int64)
        for k in range(D - 1): P[k, 0] = k + 1; P[k, 1] = k
        P[D - 1, 0] = D - 1; P[D - 1, 1] = D - 1
        return P
    P = np.empty((2 * D - 1, 2), np.int64)
    for k in range(D): P[k, 0] = k; P[k, 1] = k
    for k in range(D - 1): P[D + k, 0] = k + 1; P[D + k, 1] = k
    return P

@njit(cache=False)
def _backward(nx, mon, st, rad, h, w, S, FULL, D, mode, R):
    """Backward recursion F(i, r; u) over kept positions. mode 0: saturating counter of T (bins 0..D-2 exact, bin D-1 = '>= D-1');
    mode 1: truncated polynomial in x = u - 1 (coefficients Z, Z E[T], Z E[T(T-1)]/2, ...). Ring values share one running log scale e;
    Gfull[i] is the value at the full state of month(i) with its scale lcfull[i] (true value = Gfull * exp(lcfull))."""
    L = len(nx); M = len(S); Smax = 1
    for m in range(M):
        if S[m] > Smax: Smax = S[m]
    ring = np.zeros((R, D, Smax)); Gfull = np.zeros((L, D)); lcfull = np.full(L, -np.inf)
    P0 = _pairs(D, mode, 0); P1 = _pairs(D, mode, 1); e = 0.0; BIG = 1e200; LBIG = np.log(BIG)
    for i in range(L - 1, -1, -1):
        m = mon[i]; Sm = S[m]; cur = ring[i % R]
        # skip term
        if i + 1 < L and mon[i + 1] == m:
            src = ring[(i + 1) % R]
            for k in range(D):
                for s in range(Sm): cur[k, s] = src[k, s]
        else:
            for k in range(D):
                for s in range(Sm): cur[k, s] = 0.0
            if i + 1 < L:
                if lcfull[i + 1] > -np.inf:
                    f = np.exp(lcfull[i + 1] - e)
                    for k in range(D): cur[k, 0] = f * Gfull[i + 1, k]
            else:
                cur[0, 0] = np.exp(-e)
        # take term
        j = nx[i]; wi = w[i]; stb = st[i]; rb = rad[i]
        PP = P1 if h[i] == 1 else P0
        if j < L and mon[j] == m:
            src = ring[j % R]; block = stb * rb; nq = Sm // block
            for p in range(PP.shape[0]):
                kd = PP[p, 0]; ks = PP[p, 1]
                for q in range(nq):
                    base = q * block
                    for r in range(1, rb):
                        o = base + r * stb
                        for l in range(stb):
                            cur[kd, o + l] += wi * src[ks, o - stb + l]
        elif stb < Sm:
            ok = False; f = 0.0
            if j < L and mon[j] == m + 1 and lcfull[j] > -np.inf:
                ok = True; f = wi * np.exp(lcfull[j] - e)
            elif j >= L and m == M - 1:
                ok = True; f = wi * np.exp(-e)
            if ok:
                for p in range(PP.shape[0]):
                    kd = PP[p, 0]; ks = PP[p, 1]
                    if j < L: cur[kd, stb] += f * Gfull[j, ks]
                    elif ks == 0: cur[kd, stb] += f
        # occasional rescale of the whole ring (values only grow as i decreases)
        if (i & 15) == 0 or (i + 1 < L and mon[i + 1] != m):
            mx = 0.0
            for k in range(D):
                for s in range(Sm):
                    if cur[k, s] > mx: mx = cur[k, s]
            if mx > BIG:
                ring *= 1.0 / BIG; e += LBIG
        f_ = FULL[m]
        for k in range(D): Gfull[i, k] = cur[k, f_]
        lcfull[i] = e
    return Gfull, lcfull

def exact_law(P, kcap):
    """Exact law of T under Q_nu: P(T = k) for k < kcap and P(T >= kcap) (last entry)."""
    G, lc = _backward(P['nx'], P['mon'], P['st'], P['rad'], P['h'], P['w'], P['S'], P['FULL'], int(kcap) + 1, 0, P['R'])
    v = G[0]
    if not v.sum() > 0: raise ValueError('no admissible configuration: the counts cannot be placed under the cooldown')
    return v / v.sum(), lc[0] + np.log(v.sum())

def exact_moments(P):
    """Exact E[T] and Var[T] under Q_nu from the factorial-moment pass."""
    G, lc = _backward(P['nx'], P['mon'], P['st'], P['rad'], P['h'], P['w'], P['S'], P['FULL'], 3, 1, P['R'])
    z, f1, f2 = G[0]; mean = f1 / z; e2 = 2 * f2 / z           # E[T(T-1)] = 2 * coefficient of x^2
    return mean, e2 + mean - mean ** 2

def log_count(P):
    """log of the number (weighted) of admissible configurations."""
    G, lc = _backward(P['nx'], P['mon'], P['st'], P['rad'], P['h'], P['w'], P['S'], P['FULL'], 1, 0, P['R'])
    return lc[0] + np.log(G[0, 0])

# ---------------------------------------------------------------- exact sampler
@njit(cache=False)
def _rows(m, lo, hi, after, after_e, nx, mon, st, rad, w, S, mend, Gfull, lcfull):
    """Backward rows (D = 1, u = 1) of month m for positions [lo, hi), given the rows of [hi, hi + len(after)) (empty when hi is the month end).
    Returns rows for [lo, hi + len(after)); row r holds F(lo + r, .) * exp(-erow[r])."""
    b = mend[m]; Sm = S[m]; L = len(nx); M = len(S); nA = after.shape[0]; tot = hi - lo + nA
    T = np.zeros((tot, Sm)); erow = np.zeros(tot); BIG = 1e200; LBIG = np.log(BIG)
    for r in range(nA):
        erow[hi - lo + r] = after_e[r]
        for s in range(Sm): T[hi - lo + r, s] = after[r, s]
    if nA > 0: e = after_e[0]
    else:
        e = lcfull[b] if b < L else 0.0
        if e == -np.inf: e = 0.0
    for i in range(hi - 1, lo - 1, -1):
        il = i - lo
        if i + 1 < b:
            f = np.exp(erow[il + 1] - e)
            for s in range(Sm): T[il, s] = f * T[il + 1, s]
        elif i + 1 < L:
            if lcfull[i + 1] > -np.inf: T[il, 0] = np.exp(lcfull[i + 1] - e) * Gfull[i + 1, 0]
        else:
            T[il, 0] = np.exp(-e)
        j = nx[i]; stb = st[i]; rb = rad[i]; wi = w[i]
        if j < b:
            f = wi * np.exp(erow[j - lo] - e); block = stb * rb; nq = Sm // block; jl = j - lo
            for q in range(nq):
                base = q * block
                for r in range(1, rb):
                    o = base + r * stb
                    for l in range(stb):
                        T[il, o + l] += f * T[jl, o - stb + l]
        elif stb < Sm:
            if j < L and mon[j] == m + 1 and lcfull[j] > -np.inf: T[il, stb] += wi * np.exp(lcfull[j] - e) * Gfull[j, 0]
            elif j >= L and m == M - 1: T[il, stb] += wi * np.exp(-e)
        mx = 0.0
        for s in range(Sm):
            if T[il, s] > mx: mx = T[il, s]
        if mx > BIG:
            for s in range(Sm): T[il, s] /= BIG
            e += LBIG
        erow[il] = e
    return T, erow

@njit(cache=False)
def _walk_seg(m, lo, hi, T, erow, nx, mon, st, rad, w, mend, FULL, Gfull, lcfull, cur_i, cur_s, out, nout, seed):
    """Advance every draw whose current position lies in [lo, hi) of month m until it leaves the segment."""
    np.random.seed(seed)
    b = mend[m]; L = len(nx); M = len(FULL)
    for d in range(len(cur_i)):
        i = cur_i[d]; s = cur_s[d]
        if i < lo or i >= hi: continue
        while i < hi:
            if s == 0:
                i = b; break
            il = i - lo; j = nx[i]; tv = 0.0; stb = st[i]
            if (s // stb) % rad[i] >= 1:
                if j < b:
                    tv = w[i] * T[j - lo, s - stb] * np.exp(erow[j - lo] - erow[il])
                elif s - stb == 0:
                    if j < L and mon[j] == m + 1: tv = w[i] * Gfull[j, 0] * np.exp(lcfull[j] - erow[il])
                    elif j >= L and m == M - 1: tv = w[i] * np.exp(-erow[il])
            sv = T[il + 1, s] * np.exp(erow[il + 1] - erow[il]) if i + 1 < b else 0.0
            p = tv / (tv + sv) if tv + sv > 0.0 else 0.0
            if np.random.random() < p:
                out[d, nout[d]] = i; nout[d] += 1; s -= stb; i = j
                if j >= b: break
            else:
                i += 1
        cur_i[d] = i
        if i >= b: cur_s[d] = FULL[mon[i]] if i < L else 0
        else: cur_s[d] = s

def sample_exact(P, ndraw, seed=0, seg=4000):
    """ndraw independent exact draws from Q_nu; returns an (ndraw, n) array of kept-position indices (sorted per draw).
    Each month is processed in segments of `seg` positions: one backward pass stores only the rows just after each segment boundary,
    and each segment is recomputed from its checkpoint when the draws walk through it (memory about (seg + R) x states per month)."""
    G, lcf = _backward(P['nx'], P['mon'], P['st'], P['rad'], P['h'], P['w'], P['S'], P['FULL'], 1, 0, P['R'])
    n = P['n']; R = P['R']; out = np.full((ndraw, n), -1, np.int64); nout = np.zeros(ndraw, np.int64)
    cur_i = np.zeros(ndraw, np.int64); cur_s = np.full(ndraw, P['FULL'][0], np.int64)
    args = (P['nx'], P['mon'], P['st'], P['rad'], P['w'], P['S'], P['mend'], G, lcf)
    for m in range(P['M']):
        a, b = int(P['mstart'][m]), int(P['mend'][m]); Sm = int(P['S'][m])
        bounds = list(range(a, b, seg)) + [b]; nseg = len(bounds) - 1
        ck = [None] * nseg; after = np.zeros((0, Sm)); after_e = np.zeros(0)
        for k in range(nseg - 1, -1, -1):
            lo, hi = bounds[k], bounds[k + 1]
            ck[k] = (after, after_e)
            Tm, er = _rows(m, lo, hi, after, after_e, *args)
            nA = min(R, b - lo); after = Tm[:nA].copy(); after_e = er[:nA].copy()
        for k in range(nseg):
            lo, hi = bounds[k], bounds[k + 1]
            Tm, er = _rows(m, lo, hi, ck[k][0], ck[k][1], *args)
            _walk_seg(m, lo, hi, Tm, er, P['nx'], P['mon'], P['st'], P['rad'], P['w'], P['mend'], P['FULL'], G, lcf,
                      cur_i, cur_s, out, nout, (seed * 1000003 + m) * 1009 + k)
    if np.any(nout != n): raise RuntimeError('sampler did not place every alert')
    return out

def check_draws(P, idx, C):
    """Constraint audit of exact draws: counts per (month, class) and spacing."""
    t = P['t'][idx]; bad_space = int((np.diff(t, axis=1) < C).sum())
    bad_count = 0
    for d in range(idx.shape[0]):
        cnt = np.zeros_like(P['NU']); np.add.at(cnt, (P['mon'][idx[d]], P['cls'][idx[d]]), 1)
        bad_count += int(np.any(cnt != P['NU']))
    return bad_space, bad_count
import itertools


def _enumerate(times, month, cls, nu, C):
    n = sum(nu.values()); out = []
    for comb in itertools.combinations(range(len(times)), n):
        if any(times[comb[k + 1]] - times[comb[k]] < C for k in range(n - 1)):
            continue
        cnt = {}
        for i in comb:
            cnt[(month[i], cls[i])] = cnt.get((month[i], cls[i]), 0) + 1
        if all(cnt.get(c, 0) == v for c, v in nu.items() if v > 0) and all(nu.get(c, 0) == v for c, v in cnt.items()):
            out.append(comb)
    return out


