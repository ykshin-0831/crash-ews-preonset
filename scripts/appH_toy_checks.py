"""Exact checks, by enumeration, of the small-design values of Section 3.6 and Appendix H.

The script lists every admissible alert set of each small design and computes from that list the laws
Q_v(A) ∝ prod_{t in A} v(t), the step odds ratio OR_v(t, H) = E_Q[W | H, X_t = 1] / E_Q[W | H, X_t = 0]
and the worst cases. It checks the values that the text gives as fractions or closed forms in exact
rational arithmetic (fractions.Fraction); the exhaustive searches over weight vertices use floats whose
sums are exact integers (Gamma = 2) and confirm the maximum in Fraction arithmetic. The script reads no file.

Checks (Section 3.6 and Appendix H of the manuscript):
  H.1  least Lambda for the first gap design (n = 2, C = 3, Gamma = 2): 10/3 forward, 8/3 backward
  H.2  Proposition 4: Q_v(T >= 1) = 2G/(2G+1) and (G^2+G)/(G^2+G+1), their difference; the tilt understates
       a block's worst-case odds by G^(n-1)
  H.3  Cov(X1, X3) = 1/9; Proposition 5: OR_v = G^n at t = 1 and no step exceeds G^n over all vertex weights;
       without cooldown G (n completions of weight G against one of weight 1); gap design: one completion
       without t = 1, a fraction (C-1)/(C+n-2) of all-G completions, OR_v = G((C-1)G+1)/C for n = 2 (10/3);
       contiguous cell: n+1 admissible sets, OR_v = G(G^n - 1)/((G - 1)n), exhaustive search at G = 2
       (n = 2 up to 14 minutes: largest 3; n = 3 up to 13 minutes: largest 14/3); coupling bound 4 vs largest 3
  H.4 / (8)  Proposition 6: Q(T >= K) = 2^-K, p^S = (L/(1+L))^K, p^R = (G^n/(1+G^n))^K, and the example
       n = 2, K = 5, G = L = 1.15: 0.031, 0.044, 0.060
"""
from fractions import Fraction as Fr
from functools import lru_cache
from itertools import combinations

import numpy as np

RESULTS = []


def fmt(x):
    if isinstance(x, (list, tuple)):
        return '(' + ', '.join(fmt(y) for y in x) + ')'
    return str(x)


def check(label, got, want):
    """`want` is the value stated in the text, or the stated closed form evaluated exactly."""
    ok = got == want
    RESULTS.append(ok)
    print(f'  {label}: {fmt(got)}' + ('' if ok else f' (stated: {fmt(want)})') + f'  {"MATCH" if ok else "DIFFERS"}')


# ---------------------------------------------------------------- designs and enumeration
def admissible(pos, cell, nu, C):
    """All alert sets with exactly nu[c] alerts in each cell c and consecutive alerts at least C apart.
    pos: sorted minutes; cell: cell label of each minute (minutes of a cell with nu = 0 hold no alert)."""
    elig = [p for p, c in zip(pos, cell) if nu.get(c, 0) > 0]
    cof = dict(zip(pos, cell))
    out = []
    for A in combinations(elig, sum(nu.values())):
        if all(b - a >= C for a, b in zip(A, A[1:])) and all(sum(cof[a] == c for a in A) == k for c, k in nu.items()):
            out.append(A)
    return out


def steps(sets, order):
    """Unforced reachable steps (t, history) read in `order`; returns (t, history, C1, C0) with both non-empty."""
    out = []
    for i, t in enumerate(order):
        before = set(order[:i])
        groups = {}
        for A in sets:
            groups.setdefault(tuple(sorted(a for a in A if a in before)), []).append(A)
        for H, S in groups.items():
            S1 = [A for A in S if t in A]
            S0 = [A for A in S if t not in A]
            if S1 and S0:
                out.append((t, H, S1, S0))
    return out


def W(A, v):
    w = Fr(1)
    for a in A:
        w *= v.get(a, 1)
    return w


def odds_ratio(S1, S0, v):
    return (sum(W(A, v) for A in S1) / len(S1)) / (sum(W(A, v) for A in S0) / len(S0))


def least_lambda(sets, v, order):
    return max(max(r, 1 / r) for r in (odds_ratio(S1, S0, v) for _, _, S1, S0 in steps(sets, order)))


def vertex_search(sets, order, gamma=2, first_step_only=False):
    """max over v in {1, gamma}^eligible and over unforced steps of OR_v and of 1/OR_v (exact at the maximum)."""
    elig = sorted({a for A in sets for a in A})
    M = np.array([[a in A for A in sets] for a in elig], np.int64)                  # minutes x sets
    bits = (np.arange(2 ** len(elig))[:, None] >> np.arange(len(elig))) & 1         # vertices x minutes
    Wv = float(gamma) ** (bits @ M)                                                  # exact integers for gamma = 2
    st = steps(sets, order)[:1] if first_step_only else steps(sets, order)
    col = {A: j for j, A in enumerate(sets)}
    G1 = np.zeros((len(sets), len(st))); G0 = np.zeros((len(sets), len(st)))
    for k, (_, _, S1, S0) in enumerate(st):
        G1[[col[A] for A in S1], k] = 1 / len(S1); G0[[col[A] for A in S0], k] = 1 / len(S0)
    R = (Wv @ G1) / (Wv @ G0)                                                        # vertices x steps
    best = []
    for arr in (R, 1 / R):
        iv, ks = np.unravel_index(np.argmax(arr), arr.shape)
        v = {a: Fr(gamma) for a, b in zip(elig, bits[iv]) if b}
        r = odds_ratio(st[ks][2], st[ks][3], v)
        best.append(r if arr is R else 1 / r)
    return best                                                                      # [max OR, max 1/OR]


GAMMAS = (Fr(3, 2), Fr(2), Fr(3), Fr(7, 3))

# ---------------------------------------------------------------- Proposition 4 (H.2)
print('Proposition 4: four minutes, one cell with two alerts at least two apart, minute 1 marked')
pos = [1, 2, 3, 4]
S = admissible(pos, [0] * 4, {0: 2}, 2)
check('admissible sets', S, [(1, 3), (1, 4), (2, 4)])
for g in GAMMAS:
    tilt = sum(W(A, {1: g}) for A in S if 1 in A) / sum(W(A, {1: g}) for A in S)
    alt = sum(W(A, {1: g, 3: g}) for A in S if 1 in A) / sum(W(A, {1: g, 3: g}) for A in S)
    check(f'G = {g}: tilt, (G,1,G,1), difference',
          (tilt, alt, alt - tilt), (2 * g / (2 * g + 1), (g * g + g) / (g * g + g + 1), g * (g - 1) / ((2 * g + 1) * (g * g + g + 1))))
pX1 = Fr(sum(1 in A for A in S), len(S)); pX3 = Fr(sum(3 in A for A in S), len(S))
check('Cov(X1, X3) under Q (H.3)', Fr(sum(1 in A and 3 in A for A in S), len(S)) - pX1 * pX3, Fr(1, 9))


# ---------------------------------------------------------------- Proposition 5 (H.3)
def prop5_design(n, C):
    J = (n - 1) * C + 2
    pos = list(range(1, J + 1))
    cell = [1 if p in (2, (n - 1) * C + 1) else 0 for p in pos]
    return pos, admissible(pos, cell, {0: n - 1, 1: 1}, C)


print('\nProposition 5: J = (n-1)C + 2 positions, two cells; v = G on A1 = {1, C+1, ..., (n-1)C+1}')
for n in range(2, 6):
    for C in range(2, 5):
        pos, S = prop5_design(n, C)
        A1 = tuple(range(1, (n - 1) * C + 2, C)); A2 = tuple(a + 1 for a in A1)
        assert sorted(S) == sorted([A1, A2]), (n, C, S)
        got = [odds_ratio(*steps(S, pos)[0][2:], {a: g for a in A1}) for g in GAMMAS]
        check(f'n = {n}, C = {C}: only the two phases; OR_v at t = 1 for G in {[str(g) for g in GAMMAS]}', got, [g ** n for g in GAMMAS])
        if (n - 1) * C + 2 <= 12:                           # every step, every vertex of {1, 2}^J: nothing exceeds 2^n
            check(f'   n = {n}, C = {C}: max over vertices and steps of OR, 1/OR (G = 2)', vertex_search(S, pos), [Fr(2 ** n)] * 2)
print('Blocks of Proposition 6 (C = 2, marked position 2): tilt odds G vs whole-phase odds G^n, ratio G^(n-1)')
for n in range(2, 6):
    pos, S = prop5_design(n, 2)
    even, odd = tuple(range(2, 2 * n + 1, 2)), tuple(range(1, 2 * n, 2))
    assert sorted(S) == sorted([odd, even])
    for g in (Fr(3, 2), Fr(2)):
        o_tilt = W(even, {2: g}) / W(odd, {2: g})                       # odds of the even phase, v = G on position 2
        o_phase = W(even, {a: g for a in even}) / W(odd, {a: g for a in even})
        check(f'n = {n}, G = {g}: (tilt odds, phase odds, ratio)', (o_tilt, o_phase, o_phase / o_tilt), (g, g ** n, g ** (n - 1)))

print('\nWithout a cooldown: one cell of n+1 minutes holding n alerts, v = G on minute 1')
for n in range(2, 6):
    pos = list(range(1, n + 2)); S = admissible(pos, [0] * (n + 1), {0: n}, 1)
    _, _, S1, S0 = steps(S, pos)[0]
    check(f'n = {n}: completions with/without t = 1 and OR_v at t = 1 (G = 2, 3)',
          (len(S1), len(S0), [odds_ratio(S1, S0, {1: g}) for g in (Fr(2), Fr(3))]), (n, 1, [Fr(2), Fr(3)]))
    check(f'   n = {n}: max over vertices and steps (G = 2), Lemma H.2 bound G', vertex_search(S, pos), [Fr(2), Fr(2)])


def gap_design(n, C):
    pos = [1] + list(range(C, n * C + 1))
    v = lambda g: {p: g for p in pos if p == 1 or p % C != 0}          # 1 and minutes strictly between multiples of C
    return pos, admissible(pos, [0] * len(pos), {0: n}, C), v


print('\nOne cell with gaps: eligible minutes 1 and C, ..., nC; v = G on 1 and between multiples of C')
for n in range(2, 5):
    for C in range(2, 7):
        pos, S, v = gap_design(n, C)
        _, _, S1, S0 = steps(S, pos)[0]
        allG = Fr(sum(all(p in v(2) for p in A) for A in S1), len(S1))
        g = Fr(2)
        r = odds_ratio(S1, S0, v(g))
        line = (S0 == [tuple(range(C, n * C + 1, C))], allG, r >= g ** n * Fr(C - 1, C + n - 2))
        check(f'n = {n}, C = {C}: (single completion without t = 1, all-G share, OR >= G^n(C-1)/(C+n-2))',
              line, (True, Fr(C - 1, C + n - 2), True))
        if n == 2:
            check(f'   n = 2, C = {C}: OR_v at t = 1, G in (3/2, 2, 3)',
                  [odds_ratio(S1, S0, v(g)) for g in (Fr(3, 2), Fr(2), Fr(3))], [g * ((C - 1) * g + 1) / C for g in (Fr(3, 2), Fr(2), Fr(3))])
pos, S, v = gap_design(2, 3)
check('n = 2, C = 3, G = 2: OR_v at t = 1', odds_ratio(*steps(S, pos)[0][2:], v(Fr(2))), Fr(10, 3))
check('n = 2, C = 4, G = 2: OR_v at t = 1', odds_ratio(*steps(gap_design(2, 4)[1], gap_design(2, 4)[0])[0][2:], gap_design(2, 4)[2](Fr(2))), Fr(7, 2))
check('H.1: least Lambda with Q_v in S_Lambda, reading forward / backward',
      (least_lambda(S, v(Fr(2)), pos), least_lambda(S, v(Fr(2)), pos[::-1])), (Fr(10, 3), Fr(8, 3)))
_, _, S1, S0 = steps(S, pos)[0]
coupling = max(Fr(2) ** len(set(B) - set(S0[0])) for B in S1)                       # unique coupling: every B with S0[0]
check('H.4 remark: coupling bound on 1/OR_v at t = 1, and its largest value over vertices (G = 2)',
      (len(S0), coupling, vertex_search(S, pos, first_step_only=True)[1]), (1, Fr(4), Fr(3)))

print('\nOne cell of 2n consecutive minutes, C = 2, v = G on odd minutes')
for n in range(2, 7):
    pos = list(range(1, 2 * n + 1)); S = admissible(pos, [0] * (2 * n), {0: n}, 2)
    Ak = [tuple(range(1, 2 * k, 2)) + tuple(range(2 * k + 2, 2 * n + 1, 2)) for k in range(n + 1)]
    got = [odds_ratio(*steps(S, pos)[0][2:], {p: g for p in pos if p % 2}) for g in GAMMAS]
    check(f'n = {n}: sets are A_0..A_n; OR_v at t = 1', (sorted(S) == sorted(Ak), got),
          (True, [g * (g ** n - 1) / ((g - 1) * n) for g in GAMMAS]))
print('Exhaustive search, G = 2: contiguous single cell, every length, cooldown, unforced step and vertex')
for n, Jmax, want in ((2, 14, Fr(3)), (3, 13, Fr(14, 3))):
    best = Fr(0); arg = None
    for J in range(n, Jmax + 1):
        pos = list(range(1, J + 1))
        for C in range(1, (J - 1) // (n - 1) + 1):
            S = admissible(pos, [0] * J, {0: n}, C)
            if len(S) > 1:
                m = max(vertex_search(S, pos))
                if m > best:
                    best, arg = m, (J, C)
    check(f'n = {n}, up to {Jmax} minutes: largest max(OR, 1/OR) (attained at J, C = {arg})', best, want)


# ---------------------------------------------------------------- Proposition 6 (H.4) and equation (8)
def blocks(n, K):
    pos, cell, nu, h = [], [], {}, {}
    for j in range(K):                                   # block j: Proposition 5 design with C = 2, its own two cells
        off = j * (2 * n + 3)                            # separation: minutes off + 2n + 1, + 2 lie in a cell with nu = 0
        for p in range(1, 2 * n + 3):
            pos.append(off + p)
            cell.append((j, 1) if p in (2, 2 * n - 1) else (j, 0) if p <= 2 * n else 'sep')
            h[off + p] = int(p == 2)                     # block j's event: pre-onset set = its position 2
        nu[(j, 0)], nu[(j, 1)] = n - 1, 1
    nu['sep'] = 0
    return pos, admissible(pos, cell, nu, 2), h


def p_seq(pos, S, h, K, lam):
    """Worst case of P(T >= K) over the sequential model S_Lambda, by backward induction over histories."""
    @lru_cache(None)
    def V(i, live):
        if i == len(pos):
            return Fr(int(sum(h[a] for a in live[0]) >= K))
        S1 = tuple(A for A in live if pos[i] in A); S0 = tuple(A for A in live if pos[i] not in A)
        if not S1 or not S0:
            return V(i + 1, S1 or S0)
        o = Fr(len(S1), len(S0))
        return max(p * V(i + 1, S1) + (1 - p) * V(i + 1, S0) for p in (o * lam / (1 + o * lam), o / lam / (1 + o / lam)))
    return V(0, tuple(S))


def p_rosenbaum(S, h, K, gamma):
    """max over v in {1, gamma}^eligible of Q_v(T >= K) (vertex search; exact at the maximum)."""
    elig = sorted({a for A in S for a in A})
    M = np.array([[a in A for A in S] for a in elig], np.int64)
    hit = np.array([sum(h[a] for a in A) >= K for A in S])
    best, arg = -1.0, None
    for lo in range(0, 2 ** len(elig), 2 ** 16):
        bits = (np.arange(lo, min(lo + 2 ** 16, 2 ** len(elig)))[:, None] >> np.arange(len(elig))) & 1
        Wv = float(gamma) ** (bits @ M)
        q = Wv[:, hit].sum(1) / Wv.sum(1)
        if q.max() > best:
            best, arg = q.max(), bits[q.argmax()]
    v = {a: gamma for a, b in zip(elig, arg) if b}
    return sum(W(A, v) for A, k in zip(S, hit) if k) / sum(W(A, v) for A in S)


print('\nProposition 6: K blocks of the Proposition 5 design (C = 2), one marked minute per block')
for n, K, g in ((2, 1, Fr(2)), (2, 2, Fr(3, 2)), (2, 3, Fr(2)), (3, 2, Fr(2)), (2, 5, Fr(23, 20))):
    pos, S, h = blocks(n, K)
    q0 = Fr(sum(sum(h[a] for a in A) >= K for A in S), len(S))
    got = (len(S), q0, p_seq(pos, S, h, K, g), p_rosenbaum(S, h, K, g))
    check(f'n = {n}, K = {K}, G = L = {g}: (sets, Q(T>=K), p^S_L, p^R_G)', got,
          (2 ** K, Fr(1, 2 ** K), (g / (1 + g)) ** K, (g ** n / (1 + g ** n)) ** K))
    if (n, K) == (2, 5):
        check('   rounded: null tail, sequential, Rosenbaum-type', tuple(f'{float(x):.3f}' for x in got[1:]), ('0.031', '0.044', '0.060'))
        lam_eq = [x for x in (g ** 2 - Fr(1, 10 ** 6), g ** 2, g ** 2 + Fr(1, 10 ** 6))]
        check('   p^S_L >= p^R_G exactly when L >= G^n (L just below, at, above G^2)',
              [(l / (1 + l)) ** K >= got[3] for l in lam_eq], [False, True, True])

print(f'\n{sum(RESULTS)} of {len(RESULTS)} checks match.')
