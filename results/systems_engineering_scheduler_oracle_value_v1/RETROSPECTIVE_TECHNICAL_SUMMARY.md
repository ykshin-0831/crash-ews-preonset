# BTC Systems Engineering Research — Retrospective Findings (Auto-generated)

All 2023–Apr 2026 findings are EXPLORATORY; their periods were previously inspected.
Budget levels are experimental controls, not economically optimal choices. Oracle is clairvoyant, not implementable.

## Primary budget: B=240 annual alerts (experimental)

| mode         |   alarms |   caught |   pre_onset_caught |   false_alarms |   event_recall |   pre_onset_recall |
|:-------------|---------:|---------:|-------------------:|---------------:|---------------:|-------------------:|
| calendar     |      800 |       39 |                 22 |            761 |       0.46988  |          0.26506   |
| rolling30    |      791 |       21 |                  8 |            770 |       0.253012 |          0.0963855 |
| paced        |      789 |       28 |                 14 |            761 |       0.337349 |          0.168675  |
| token_bucket |      793 |       20 |                 10 |            773 |       0.240964 |          0.120482  |

## Paired primary outcome delta against calendar cap

| comparison               |   delta_alerts |   delta_caught |   delta_preonset |   paired_exact_p |   paired_preonset_p |
|:-------------------------|---------------:|---------------:|-----------------:|-----------------:|--------------------:|
| calendar_vs_paced        |            -11 |            -11 |               -8 |       0.0707555  |          0.0768127  |
| calendar_vs_rolling30    |             -9 |            -18 |              -14 |       0.00393317 |          0.00434351 |
| calendar_vs_token_bucket |             -7 |            -19 |              -12 |       0.00256321 |          0.0118179  |

## Interpretation reminders

- Alarm counts are compared as realized burdens; equality in 2022 calibration does not force equality in 2023+.
- No-alert remains an operational reference; 0 alerts and 0 events caught.
- Oracle can see future events and discard false alerts, so oracle gap is NOT attributable only to scheduling.
- Event-matching precision is not the probability of avoided 4-hour loss.
- All economic outcomes reflect simplified proxy execution, not actual trading returns.
- A development-period SHA-256 written today does NOT retroactively preregister earlier data exploration.
- Keep the 4% independent event catalog fixed; 3%/5% are separately labeled sensitivity analyses.