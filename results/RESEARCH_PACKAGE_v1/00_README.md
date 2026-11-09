# BTC crash early-warning research package (v1)

Created 2026-10-29T14:19:40.052699+00:00.

**Research question.** Early-warning reliability under capacity constraints: can a capacity-limited alert system warn about BTC 4% intraday crashes before the decline starts, as opposed to detecting a decline already in progress?

**Reporting rule.** Early warning (alert before the 1% onset) and in-progress detection (alert after the onset but at least 30 min before the 4% breach) are reported separately. The composite "caught" is reference only.

## Confirmatory holdout (2018-2020, notebook 11)

- H1 early warning: 13 pre-onset hits vs null mean 11.4, one-sided p = 0.3532
- Composite (secondary): 36 caught, p = 9.999e-05
- H2 scheduler: calendar_vs_rolling30 Holm p = 1.000; calendar_vs_paced Holm p = 1.000; calendar_vs_token_bucket Holm p = 0.516
- H3 capacity: 103 of 163 misses had no monthly room

## Endpoints separated (notebook 12, primary circular-shift null)

| period | endpoint | observed | null mean | one-sided p |
|---|---|---|---|---|
| dev | early warning | 22 | 4.7 | 0.0001 |
| dev | in-progress detection | 17 | 3.3 | 0.0001 |
| dev | early warning (secondary) | 29 | 9.1 | 0.0003 |
| holdout | early warning | 13 | 11.4 | 0.3532 |
| holdout | in-progress detection | 23 | 7.7 | 0.0001 |
| holdout | early warning (secondary) | 22 | 21.6 | 0.5063 |

## Files

- `DEFINITIONS.json`
- `EXPERIMENTS.csv`
- `DATA_PROVENANCE.csv`
- `DATA_PROVENANCE_SUMMARY.csv`
- `RESULTS_COUNTS.csv`
- `RESULTS_TESTS.csv`
- `FILE_HASHES.csv`
- `PACKAGE_MANIFEST.json`
