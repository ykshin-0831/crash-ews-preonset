# Calendar placement and the exact null (Online Appendix I.7)

M0 issued 693 of its 800 development alerts on UTC days 1–15. The exact conditional null of notebook 34 matches M0's alert counts by month and volatility decile but lets each cell's alerts fall anywhere in the month. This check adds the half-month (days 1–15, days 16 onward) to the cells and recomputes the exact law of M0's breach-anchored pre-onset count.

| File | Contents |
|---|---|
| `half_month_null.py` | The check, in three modes: `describe`, `month_decile`, `half_month` |
| `exactnull_nb34.py` | The exact backward recursion of notebook 34 |
| `result_describe.json` | Shares of alerts, pre-onset minutes, eligible minutes and onsets on days 1–15 |
| `result_month_decile.json` | Exact null with month × decile cells (reproduces notebook 34) |
| `result_half_month.json` | Exact null with month × decile × half-month cells |

The check reads `results/design_constant_v1/39_design.npz` and needs numpy and numba. From the repository root:

```
python checks/half_month_null/half_month_null.py describe        # a few seconds
python checks/half_month_null/half_month_null.py month_decile    # about 15 minutes
python checks/half_month_null/half_month_null.py half_month      # about 30 minutes
```

With month × decile cells the exact null reproduces notebook 34 (mean 6.70, P(T ≥ 22) = 7.4 × 10⁻⁸). With half-month cells added, the null mean rises to 8.04 and P(T ≥ 22) = 1.2 × 10⁻⁶.

I ran this check on the development sample after registering the prospective test; it is exploratory.
