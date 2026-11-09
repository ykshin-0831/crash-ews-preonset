"""Notebook 39's odds-ratio kernel on the constructions of Appendix H.3 (manuscript Appendix H.7, "Checks").

The text states that the kernel "reproduces the constructions of Appendix H.3 exactly: OR_v = Gamma^n in the
design of Proposition 5 for n = 2-5 and C = 2-4, 10/3 and 7/2 in the single cell with gaps (n = 2; C = 3 and 4),
and Gamma(Gamma^n - 1)/[(Gamma - 1)n] in the contiguous single cell". This script runs the kernel on those
designs and compares its step odds ratio at t = 1 (the first decision, empty history) with these values.
It also reports notebook 39's own stored check against full enumeration
(results/design_constant_v1/1_kernel_check.csv: 208 random small designs, largest error 1.1e-15).

I copied the kernel below (prepare, _pairs, _backward, log_count, path_states, _or_paths, or_paths, _enumerate)
from notebooks/39_Design_Constant_SPOT.ipynb (section 1, kernel cell) and replaced numba's @njit with a no-op
decorator, so the functions run as plain Python; these small designs do not need numba.
"""
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def njit(*args, **kwargs):          # stands in for numba.njit: run the copied functions as plain Python
    return lambda f: f


# ================================================================ notebook 39 kernel (copy) ===========
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


def log_count(P):
    """log of the number (weighted) of admissible configurations."""
    G, lc = _backward(P['nx'], P['mon'], P['st'], P['rad'], P['h'], P['w'], P['S'], P['FULL'], 1, 0, P['R'])
    return lc[0] + np.log(G[0, 0])


@njit(cache=False)
def path_states(idx, nx, mon, FULL, st, L):
    """State (remaining counts of the month, radix code) of the configuration idx at every free decision point; -1 elsewhere."""
    PS = np.full(L, -1, np.int64); k = 0; n = len(idx); free = 0
    if L == 0:
        return PS
    cur = mon[0]; s = FULL[cur]
    for i in range(L):
        if mon[i] != cur:
            cur = mon[i]; s = FULL[cur]
        if i < free:
            continue
        PS[i] = s
        if k < n and idx[k] == i:
            s -= st[i]; free = nx[i]; k += 1
    return PS


@njit(cache=False)
def _or_paths(nx, mon, st, rad, W, S, FULL, R, PS, store):
    """Backward recursion of exactnull._backward (D = 1) for K weight columns at once (column 0 must be all ones).
    At every free decision point of every path (PS[p, i] >= 0) it reads take and skip values and records
    log OR_k = log(take_k / skip_k) - log(take_0 / skip_0). Decisions where the null forces the choice are counted, not used."""
    L = len(nx); M = len(S); K = W.shape[1]; NP = PS.shape[0]; Smax = 1
    for m in range(M):
        if S[m] > Smax: Smax = S[m]
    ring = np.zeros((R, K, Smax)); Gfull = np.zeros((L, K)); lcfull = np.full((L, K), -np.inf)
    e = np.zeros(K); BIG = 1e200; LBIG = np.log(BIG)
    mx = np.full((NP, K), -np.inf); mn = np.full((NP, K), np.inf)
    imx = np.full((NP, K), -1, np.int64); imn = np.full((NP, K), -1, np.int64)
    ndec = np.zeros(NP, np.int64); nforced = np.zeros(NP, np.int64)
    LOR = np.full((NP, L if store else 1, K), np.nan)
    tk = np.zeros(K); sk = np.zeros(K)
    for i in range(L - 1, -1, -1):
        m = mon[i]; Sm = S[m]; cur = ring[i % R]
        j = nx[i]; stb = st[i]; rb = rad[i]
        same_next = (i + 1 < L) and (mon[i + 1] == m)
        take_in = (j < L) and (mon[j] == m)
        for p in range(NP):
            s = PS[p, i]
            if s < 0:
                continue
            for k in range(K):
                if same_next:
                    sv = ring[(i + 1) % R, k, s]
                elif s == 0:
                    if i + 1 < L:
                        sv = np.exp(lcfull[i + 1, k] - e[k]) * Gfull[i + 1, k]
                    else:
                        sv = np.exp(-e[k])
                else:
                    sv = 0.0
                tv = 0.0
                if (s // stb) % rb >= 1:
                    if take_in:
                        tv = W[i, k] * ring[j % R, k, s - stb]
                    elif s == stb:
                        if j < L and mon[j] == m + 1:
                            tv = W[i, k] * np.exp(lcfull[j, k] - e[k]) * Gfull[j, k]
                        elif j >= L and m == M - 1:
                            tv = W[i, k] * np.exp(-e[k])
                tk[k] = tv; sk[k] = sv
            if tk[0] > 0.0 and sk[0] > 0.0:
                ndec[p] += 1
                base = np.log(tk[0]) - np.log(sk[0])
                for k in range(K):
                    lo = np.log(tk[k]) - np.log(sk[k]) - base
                    if store:
                        LOR[p, i, k] = lo
                    if lo > mx[p, k]:
                        mx[p, k] = lo; imx[p, k] = i
                    if lo < mn[p, k]:
                        mn[p, k] = lo; imn[p, k] = i
            else:
                nforced[p] += 1
        for k in range(K):
            if same_next:
                for s in range(Sm): cur[k, s] = ring[(i + 1) % R, k, s]
            else:
                for s in range(Sm): cur[k, s] = 0.0
                if i + 1 < L:
                    cur[k, 0] = np.exp(lcfull[i + 1, k] - e[k]) * Gfull[i + 1, k]
                else:
                    cur[k, 0] = np.exp(-e[k])
            wi = W[i, k]
            if take_in:
                block = stb * rb; nq = Sm // block; jr = j % R
                for q in range(nq):
                    bq = q * block
                    for r in range(1, rb):
                        o = bq + r * stb
                        for l in range(stb):
                            cur[k, o + l] += wi * ring[jr, k, o - stb + l]
            elif stb < Sm:
                if j < L and mon[j] == m + 1:
                    cur[k, stb] += wi * np.exp(lcfull[j, k] - e[k]) * Gfull[j, k]
                elif j >= L and m == M - 1:
                    cur[k, stb] += wi * np.exp(-e[k])
            if (i & 15) == 0 or not same_next:
                mxv = 0.0
                for s in range(Sm):
                    if cur[k, s] > mxv: mxv = cur[k, s]
                if mxv > BIG:
                    for rr in range(R):
                        for s in range(Smax): ring[rr, k, s] /= BIG
                    e[k] += LBIG
            Gfull[i, k] = cur[k, FULL[m]]; lcfull[i, k] = e[k]
    return mx, mn, imx, imn, ndec, nforced, Gfull, lcfull, LOR


def or_paths(P, W, paths, store=False):
    """paths: list of sorted kept-position index arrays (each an admissible configuration)."""
    PS = np.stack([path_states(np.asarray(a, np.int64), P['nx'], P['mon'], P['FULL'], P['st'], P['L']) for a in paths])
    return _or_paths(P['nx'], P['mon'], P['st'], P['rad'], np.ascontiguousarray(W, dtype=np.float64), P['S'], P['FULL'], P['R'], PS, store)


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


# ================================================================ end of the copied kernel ===============


def run_kernel(times, cls, nu, C, v):
    """One month; cls: class of each minute; nu: {class: alerts} (a class with 0 alerts holds none);
    v: weight of each minute. Returns OR_v at the first decision and the largest OR_v and 1/OR_v over
    every free decision point of every admissible configuration."""
    times = np.asarray(times); cls = np.asarray(cls)
    P = prepare(times, np.zeros(len(times), int), cls, np.zeros(len(times), int), {(0, c): k for c, k in nu.items()}, C)
    nu_idx = {(0, b): int(P['NU'][0, b]) for b in range(P['NU'].shape[1]) if P['NU'][0, b] > 0}
    paths = _enumerate(P['t'].tolist(), P['mon'].tolist(), P['cls'].tolist(), nu_idx, C)
    W = np.column_stack([np.ones(P['L']), np.asarray(v, float)[P['keep']]])
    mx, mn, imx, imn, nd, nf, G, lc, LOR = or_paths(P, W, paths, store=True)
    assert round(float(np.exp(lc[0, 0]) * G[0, 0])) == len(paths), 'kernel count differs from enumeration'
    first = LOR[:, 0, 1]
    assert np.allclose(first, first[0], rtol=0, atol=1e-14)   # t = 1 has an empty history on every path
    return np.exp(first[0]), np.exp(mx[:, 1].max()), np.exp(-mn[:, 1].min()), len(paths)


RESULTS = []


def check(label, got, want, tol=1e-12):
    err = abs(got / want - 1)
    RESULTS.append(err < tol)
    print(f'  {label}: kernel {got:.15g}, stated {want:.15g}, relative error {err:.1e}  {"MATCH" if err < tol else "DIFFERS"}')


GAMMAS = (1.5, 2.0, 3.0)
print('Proposition 5 design: J = (n-1)C + 2 minutes; minutes 2 and (n-1)C+1 in cell 1 (one alert), the rest in cell 0 (n-1)')
for n in range(2, 6):
    for C in range(2, 5):
        J = (n - 1) * C + 2; t = np.arange(1, J + 1)
        cls = np.isin(t, [2, (n - 1) * C + 1]).astype(int)
        for g in GAMMAS:
            v = np.where((t - 1) % C == 0, g, 1.0)                 # Gamma on A1 = {1, C+1, ..., (n-1)C+1}
            r0, rmax, rinv, k = run_kernel(t, cls, {0: n - 1, 1: 1}, C, v)
            assert k == 2
            check(f'n = {n}, C = {C}, Gamma = {g}: OR_v at t = 1 vs Gamma^n', r0, g ** n)
            check(f'   largest OR_v, 1/OR_v over all steps vs Gamma^n', max(rmax, rinv), g ** n)

print('\nOne cell with gaps: eligible minutes 1 and C, ..., 2C (n = 2); v = Gamma on 1 and between C and 2C')
for C, want in ((3, 10 / 3), (4, 7 / 2)):
    t = np.arange(1, 2 * C + 1)
    cls = np.where((t > 1) & (t < C), 1, 0)                         # minutes 2..C-1: a cell with no alert
    v = np.where((t == 1) | (t % C != 0), 2.0, 1.0)
    r0, *_ = run_kernel(t, cls, {0: 2, 1: 0}, C, v)
    check(f'C = {C}, Gamma = 2: OR_v at t = 1', r0, want)

print('\nContiguous single cell: 2n consecutive minutes, C = 2, v = Gamma on odd minutes')
for n in range(2, 6):
    t = np.arange(1, 2 * n + 1)
    for g in GAMMAS:
        r0, *_ = run_kernel(t, np.zeros(2 * n, int), {0: n}, 2, np.where(t % 2 == 1, g, 1.0))
        check(f'n = {n}, Gamma = {g}: OR_v at t = 1 vs Gamma(Gamma^n - 1)/((Gamma - 1)n)', r0, g * (g ** n - 1) / ((g - 1) * n))

KC = pd.read_csv(ROOT / 'results' / 'design_constant_v1' / '1_kernel_check.csv')
print(f'\nNotebook 39 stored kernel check (1_kernel_check.csv): {len(KC)} random small designs, largest |log OR| error '
      f'{KC.err.max():.1e}, {KC.decisions.sum()} decisions (text: 208 designs, 1.1e-15)  '
      f'{"MATCH" if len(KC) == 208 and f"{KC.err.max():.1e}" == "1.1e-15" else "DIFFERS"}')
print(f'\n{sum(RESULTS)} of {len(RESULTS)} construction checks match.')
