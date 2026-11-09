# Programming sample, page 2: "New cell (added for this sample; run here on the code above)".
# Usage, from the repository root:  python programming_sample/nb34_check_bruteforce.py

# --- context from cell [15], so that this file runs on its own ----------------------
import numpy as np
from nb34_exact_null import prepare, exact_law, exact_moments, _enumerate
# --- the cell as printed -----------------------------------------------------------
# Added for this sample: the functions above on one small design, next to brute force.
# 16 candidate minutes in 2 months, two volatility classes, h = 1 marks pre-onset minutes.
times = np.arange(16)
month = np.repeat([0, 1], 8)
cls = np.array([1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0])
h = np.array([1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0])
nu = {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 1}          # alerts per (month, class)
design = prepare(times, month, cls, h, nu, cooldown=2)      # alerts >= 2 minutes apart
law_dp, log_z = exact_law(design, kcap=4)                   # P(T = 0..3), P(T >= 4)
mom_dp = exact_moments(design)                              # (E[T], Var[T]), mode-1 pass
alert_sets = _enumerate(times.tolist(), month.tolist(), cls.tolist(), nu, cooldown=2)
t_sets = np.array([h[list(a)].sum() for a in alert_sets])   # T of every admissible set
law_bf = np.bincount(t_sets, minlength=5)[:5] / len(alert_sets)   # 4 alerts, so T <= 4
mom_bf = (t_sets.mean(), t_sets.var())                      # ddof=0: variance over all sets
print(f'admissible alert sets: {len(alert_sets)} by enumeration, '
      f'{np.exp(log_z):.0f} by the DP')
print('P(T = 0, 1, 2, 3, >= 4) DP:', np.round(law_dp, 4), '  E[T], Var[T]:', np.round(mom_dp, 4))
print('                enumeration:', np.round(law_bf, 4), '               ', np.round(mom_bf, 4))
np.testing.assert_allclose(law_dp, law_bf, rtol=0, atol=1e-12)
np.testing.assert_allclose(mom_dp, mom_bf, rtol=1e-12)
# Printed output in the sample:
# admissible alert sets: 180 by enumeration, 180 by the DP
# P(T = 0, 1, 2, 3, >= 4) DP: [0.1667 0.3556 0.3222 0.1333 0.0222]   E[T], Var[T]: [1.4889 0.9832]
#                 enumeration: [0.1667 0.3556 0.3222 0.1333 0.0222]                 [1.4889 0.9832]
