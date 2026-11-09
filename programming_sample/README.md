# Python programming sample

The two-page Python programming sample excerpts notebooks 18 and 34. Page 1 splits the AUC exactly into pre-onset and in-progress parts, with a block bootstrap that never re-ranks the scores; page 2 computes an exact null for budget-constrained alerts by dynamic programming and checks it against brute force. This folder holds the printed code split into files. The three new cells I wrote for the sample run here: two on result files stored in this repository, one on a 16-minute toy design.

| File | Part of the sample | Runs here |
|---|---|---|
| [`nb18_auc_decomposition_excerpt.py`](nb18_auc_decomposition_excerpt.py) | Page 1, cells [1], [2] and [4]: tidied excerpt of notebook 18 | No. It needs the one-minute Binance cache on Google Drive and helpers the sample does not show. The full notebook is `notebooks/18_BTC_Standard_Protocol_Inflation_SPOT.ipynb` |
| [`nb18_scores_around_onset.py`](nb18_scores_around_onset.py) | Page 1, new cell: scores around crash onset | Yes, from `results/standard_protocol_inflation_v1/` |
| [`nb34_exact_null.py`](nb34_exact_null.py) | Page 2, cell [15]: core of notebook 34, tidied | Imported by `nb34_check_bruteforce.py` |
| [`nb34_check_bruteforce.py`](nb34_check_bruteforce.py) | Page 2, first new cell: the exact null against brute force on a 16-minute design | Yes |
| [`nb34_m0_exact_null.py`](nb34_m0_exact_null.py) | Page 2, second new cell: the exact null for M0's 800 alerts | Yes, from `results/exact_conditional_null_v1/` |

Each file holds its printed cells exactly, after a short header. Where a cell relies on earlier cells, a marked block before it supplies what they defined (imports and three constants from cell [1], or the functions of cell [15]); a marked block after it saves any figure to `output/`.

From the repository root, these commands install the pinned versions and run the three files:

```
pip install -r requirements-check.txt
python programming_sample/nb34_check_bruteforce.py
python programming_sample/nb34_m0_exact_null.py
python programming_sample/nb18_scores_around_onset.py
```

`output/` holds the printed output and the figures of these runs, which I made on 10 November 2026 with Python 3.13 and the versions in `../requirements-check.txt`.
