# Programming sample, page 2: cell [15], exactly as printed (notebook 34 core, tidied).
# Imported by nb34_check_bruteforce.py. The full notebook is
# notebooks/34_Exact_Conditional_Null_Budget_Audit_SPOT_PREFLIGHT_v2.ipynb.

"""Exact conditional null for budget-constrained alerts (notebook 34 core, tidied).
Q_nu: uniform over alert sets A with spacing >= cooldown (time units) and exact counts
nu[month, class]. Positions are sorted times; each has a month index, a class, an
indicator h (pre-onset minute) and a weight w (1 = uniform). Statistic T = sum h(t)."""
import itertools
import numpy as np
from numba import njit
# ---------------------------------------------------------------- design preparation
def prepare(times, month, cls, h, nu, cooldown, w=None):
    """Drop positions whose (month, class) has zero count (they can never hold an
    alert), build radix states per month. nu: dict {(month_label, class): count}."""
    times, month, cls, h = (np.asarray(a, np.int64) for a in (times, month, cls, h))
    assert np.all(np.diff(times) > 0)
    w = np.ones(len(times)) if w is None else np.asarray(w, float)
    K = (int(max(cls.max() + 1, max(b for _, b in nu) + 1)) if nu
         else int(cls.max() + 1))                               # number of classes
    mlabels = sorted({m for (m, b), v in nu.items() if v > 0})  # months with alerts
    M, mid = len(mlabels), {m: k for k, m in enumerate(mlabels)}
    NU = np.zeros((M, K), np.int64)                 # alerts per (month, class)
    for (m, b), v in nu.items():
        if v > 0:
            NU[mid[m], b] = v
    ml = np.array(mlabels, np.int64)
    pos = np.searchsorted(ml, month)                    # month label -> month index
    pos_c = np.clip(pos, 0, max(M - 1, 0))
    keep = ((M > 0) & (pos < M) & (ml[pos_c] == month) if M
            else np.zeros(len(month), bool))
    if M:
        keep &= NU[pos_c, np.clip(cls, 0, K - 1)] > 0
    t, c, hh, ww = times[keep], cls[keep], h[keep], w[keep]
    mo = pos_c[keep].astype(np.int64)
    if M and not np.all(np.diff(mo) >= 0):
        raise ValueError('months must be increasing in time')
    L = len(t)
    mstart = np.searchsorted(mo, np.arange(M), 'left')
    mend = np.searchsorted(mo, np.arange(M), 'right')
    if np.any(mend <= mstart):
        raise ValueError('a month with a positive count has no admissible position')
    # State of a month = alerts still to place in each class, packed into one integer
    # with mixed radix (NU + 1): class b is the digit with stride STR[b]. FULL = start.
    RAD, STR = NU + 1, np.zeros((M, K), np.int64)
    S, FULL = np.ones(M, np.int64), np.zeros(M, np.int64)    # states per month, start
    for m in range(M):
        s = 1
        for b in range(K):
            if NU[m, b] > 0:
                STR[m, b] = s
                s *= RAD[m, b]
        S[m], FULL[m] = s, int((NU[m] * STR[m]).sum())
    st, rad = STR[mo, c], RAD[mo, c]
    nx = np.searchsorted(t, t + cooldown, 'left').astype(np.int64)  # next allowed position
    R = int((nx - np.arange(L)).max()) + 1 if L else 1        # look-ahead the DP keeps
    return dict(t=t, mon=mo, cls=c, h=hh, w=ww, st=st, rad=rad, nx=nx, S=S, FULL=FULL,
                mstart=mstart, mend=mend, M=M, L=L, R=R, NU=NU, keep=keep,
                mlabels=mlabels, n=int(NU.sum()))
# ---------------------------------------------------------------- backward pass
# F(i, s) = weighted number of ways to finish from position i in state s, as a
# polynomial in u (power of u = pre-onset catches). Skip i: F(i+1, s). Take i, if its
# class still has alerts: w_i * u^h_i * F(nx_i, s - one alert of that class).
# A month may only end in state 0. F(0, FULL) / sum of its coefficients = law of T.
@njit(cache=False)
def _pairs(D, mode, hi):
    """(destination bin, source bin) pairs for multiplying by u^hi (mode 0, saturating)
    or (1 + x)^hi (mode 1, truncated)."""
    bins = np.arange(D)
    if hi == 0:                                          # u^0: every bin stays put
        return np.stack((bins, bins), axis=1)
    if mode == 0:                                        # u^1: shift up, top bin absorbs
        return np.stack((np.minimum(bins + 1, D - 1), bins), axis=1)
    return np.concatenate((np.stack((bins, bins), axis=1),       # (1 + x): stay, or shift up
                           np.stack((bins[1:], bins[:-1]), axis=1)))
@njit(cache=False)
def _backward(nx, mon, st, rad, h, w, S, FULL, D, mode, R):
    """Backward recursion F(i, r; u). mode 0: saturating counter of T (bins 0..D-2
    exact, bin D-1 = '>= D-1'); mode 1: truncated polynomial in x = u - 1. Ring values
    share one running log scale e; Gfull[i] = value at the full state of month(i),
    with scale lcfull[i] (true value = Gfull * exp(lcfull))."""
    L, M = len(nx), len(S)
    Smax = max(1, S.max()) if M > 0 else 1
    # only rows i..i+R-1 are ever needed, so F lives in a ring buffer of R rows
    ring = np.zeros((R, D, Smax))
    Gfull, lcfull = np.zeros((L, D)), np.full(L, -np.inf)
    P0, P1 = _pairs(D, mode, 0), _pairs(D, mode, 1)
    e, BIG, LBIG = 0.0, 1e200, np.log(1e200)
    for i in range(L - 1, -1, -1):
        m = mon[i]
        Sm, cur = S[m], ring[i % R]            # states this month, row being filled
        # skip term
        if i + 1 < L and mon[i + 1] == m:
            cur[:, :Sm] = ring[(i + 1) % R][:, :Sm]
        else:                          # last position of the month: next month starts
            cur[:, :Sm] = 0.0
            if i + 1 < L:
                if lcfull[i + 1] > -np.inf:
                    cur[:, 0] = np.exp(lcfull[i + 1] - e) * Gfull[i + 1]
            else:
                cur[0, 0] = np.exp(-e)
        # take term: states whose digit for this class is >= 1 lose one alert (-stb)
        j, wi, stb, rb = nx[i], w[i], st[i], rad[i]
        PP = P1 if h[i] == 1 else P0
        if j < L and mon[j] == m:
            src, block = ring[j % R], stb * rb
            for p in range(PP.shape[0]):
                kd, ks = PP[p, 0], PP[p, 1]
                for q in range(Sm // block):
                    base = q * block
                    for r in range(1, rb):
                        o = base + r * stb
                        for l in range(stb):
                            cur[kd, o + l] += wi * src[ks, o - stb + l]
        elif stb < Sm:                     # the cooldown jumps into the next month
            ok, f = False, 0.0
            if j < L and mon[j] == m + 1 and lcfull[j] > -np.inf:
                ok, f = True, wi * np.exp(lcfull[j] - e)
            elif j >= L and m == M - 1:
                ok, f = True, wi * np.exp(-e)
            if ok:
                for p in range(PP.shape[0]):
                    kd, ks = PP[p, 0], PP[p, 1]
                    if j < L:
                        cur[kd, stb] += f * Gfull[j, ks]
                    elif ks == 0:                  # past the last position: done
                        cur[kd, stb] += f
        # occasional rescale of the whole ring (values only grow as i decreases)
        if (i & 15) == 0 or (i + 1 < L and mon[i + 1] != m):
            if cur[:, :Sm].max() > BIG:
                ring *= 1.0 / BIG
                e += LBIG
        Gfull[i], lcfull[i] = cur[:, FULL[m]], e
    return Gfull, lcfull
def exact_law(design, kcap):
    """Exact law of T under Q_nu: P(T = k) for k < kcap and P(T >= kcap) (last)."""
    G, lc = _backward(design['nx'], design['mon'], design['st'], design['rad'],
                      design['h'], design['w'], design['S'], design['FULL'],
                      int(kcap) + 1, 0, design['R'])
    v = G[0]
    if not v.sum() > 0:
        raise ValueError('no admissible configuration: the counts cannot be placed '
                         'under the cooldown')
    return v / v.sum(), lc[0] + np.log(v.sum())
def exact_moments(design):
    """Exact E[T] and Var[T] under Q_nu from one factorial-moment pass (mode 1, D = 3)."""
    G, lc = _backward(design['nx'], design['mon'], design['st'], design['rad'],
                      design['h'], design['w'], design['S'], design['FULL'],
                      3, 1, design['R'])
    z, f1, f2 = G[0]                       # 1, x, x^2 coefficients: sums of 1, T, C(T, 2)
    mean, e2 = f1 / z, 2 * f2 / z          # E[T(T-1)] = 2 E[C(T, 2)]
    return mean, e2 + mean - mean ** 2     # Var[T] = E[T(T-1)] + E[T] - E[T]^2
def _enumerate(times, month, cls, nu, cooldown):   # brute force: every admissible set
    n, out = sum(nu.values()), []
    for comb in itertools.combinations(range(len(times)), n):
        if any(times[comb[k + 1]] - times[comb[k]] < cooldown for k in range(n - 1)):
            continue
        cnt = {}
        for i in comb:
            cnt[(month[i], cls[i])] = cnt.get((month[i], cls[i]), 0) + 1
        if (all(cnt.get(c, 0) == v for c, v in nu.items() if v > 0)
                and all(nu.get(c, 0) == v for c, v in cnt.items())):
            out.append(comb)
    return out
# Not shown: log_count, an exact sampler, and the check of all of these against _enumerate
# on 567 random designs ("max law error 0.0e+00 ... PASS", also rerun on this tidied code).
