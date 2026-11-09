# Notebooks

Every notebook that uses crypto prices uses Binance spot one-minute klines (data.binance.vision); the "Other data" column lists further sources.

The "Cited in" column gives the sections of the main text (§), of the article's Appendices A and H and of Online Appendix I (OA) that cite each notebook; "—" means none of them. The table does not cover Online Appendices B–G.

| No. | Purpose | Results folder | Other data | Cited in |
|---|---|---|---|---|
| [01](../notebooks/01_BTCUSDT_77_Monthly_LeadTime_Cost_Colab_SPOT.ipynb) | Lead time and false-alarm cost; builds the one-minute spot cache that later notebooks read | (root of results/) |  | — |
| [02](../notebooks/02_BTC_Event_Definition_Eligibility_Audit_Colab_SPOT.ipynb) | Price-only event definition and eligibility audit | `event_definition_eligibility_audit` |  | — |
| [03](../notebooks/03_BTC_Gate_Ablation_PriceOnly_Colab_SPOT.ipynb) | Gate ablation with price-only events | `gate_ablation_price_only_v1` |  | — |
| [04](../notebooks/04_BTC_Systems_Research_Colab_SPOT.ipynb) | Scheduler, oracle and economic value of alerts | `systems_engineering_scheduler_oracle_value_v1` |  | — |
| [05](../notebooks/05_BTC_Causal_Alert_Selection_Colab_SPOT.ipynb) | Causal alert selection | `causal_alert_selection_v1` |  | — |
| [06](../notebooks/06_BTC_Final_Risk_Adaptive_Capacity_Colab_SPOT.ipynb) | Risk-adaptive alert capacity | `FINAL_risk_adaptive_alert_capacity_v1` |  | — |
| [07](../notebooks/07_BTC_2026_Validation_SourceFix_v2_SPOT.ipynb) | Frozen-policy replay, May to September 2026 | `FINAL_frozen_policy_May_Sep_2026_v2_source_verified` |  | §4.4 |
| [08](../notebooks/08_BTC_CrossVenue_Information_SPOT.ipynb) | Cross-venue information: does the futures market add warning information? | `crossvenue_information_v1` | Binance futures klines | — |
| [09](../notebooks/09_BTC_Capacity_Design_SPOT.ipynb) | Alert-capacity design under clustered crashes | `capacity_design_v1` |  | — |
| [10](../notebooks/10_MultiCoin_Shared_Budget_SPOT.ipynb) | Shared alert budget across five coins | `multicoin_shared_budget_v1` |  | §4.4 |
| [11](../notebooks/11_BTC_Holdout_2018_2020_Confirmatory_SPOT.ipynb) | One-shot confirmatory holdout, BTCUSDT spot, 2018-01-01 to 2020-12-31 | `HOLDOUT_2018_2020_confirmatory_v1` | Binance futures klines | §8.1 |
| [12](../notebooks/12_BTC_EarlyWarning_Reliability_SPOT.ipynb) | Pre-onset warning versus in-progress detection under capacity constraints; null-model audit | `early_warning_reliability_v1` |  | §8.1 |
| [13](../notebooks/13_BTC_Spot_vs_Futures_Same_Events.ipynb) | Spot and futures on the same events | `spot_futures_same_event_v1` | Binance futures klines | — |
| [14](../notebooks/14_Research_Package_Builder.ipynb) | Research package: definitions, provenance and file hashes | `RESEARCH_PACKAGE_v1` |  | — |
| [15](../notebooks/15_BTC_Detection_Value_SPOT.ipynb) | Value of in-progress detection: decline avoided by acting at the alert | `detection_value_v1` |  | — |
| [16](../notebooks/16_BTC_Methods_Definitions_SPOT.ipynb) | Detection that looks like prediction under other methods and event definitions | `methods_definitions_v1` |  | §3.5, 5.2, 8.3 |
| [17](../notebooks/17_MultiCoin_Holdout_2018_2019_Confirmatory_SPOT.ipynb) | One-shot multi-coin confirmatory test, ten coins, 2018 to 2019 | `HOLDOUT17_multicoin_2018_2019_v1` |  | §8.2, 8.3 |
| [18](../notebooks/18_BTC_Standard_Protocol_Inflation_SPOT.ipynb) | Inflation of the standard protocol: per-minute AUC split at onset | `standard_protocol_inflation_v1` |  | §5.1, 5.4, 6.2, 7 |
| [19](../notebooks/19_BTC_Horizon_Onset_Power_SPOT.ipynb) | Time scale, onset threshold and power | `horizon_onset_power_v1` |  | §7, 8.1 |
| [20](../notebooks/20_Contagion_Budget_LeadTime_SPOT.ipynb) | Altcoin warnings and BTC declines; other budgets; alert lead times | `contagion_capacity_leadtime_v1` |  | — |
| [21](../notebooks/21_BTC_Onset_Positioning_Volatility_SPOT.ipynb) | Onset definition, futures positioning data, volatility models | `onset_positioning_volatility_v1` | Binance futures metrics, funding | §5.2, 7 |
| [22](../notebooks/22_BTC_Peak_Window_Robustness_SPOT.ipynb) | Length of the prior-peak window in the crash definition | `peak_window_v1` |  | §5.2, 6.2, 7 |
| [23](../notebooks/23_BTC_Drawdown_Trigger_Tradeoff_SPOT.ipynb) | One-line drawdown trigger: standard AUC and loss trade-off | `drawdown_trigger_v1` |  | §5.1 |
| [24](../notebooks/24_BTC_Leverage_Liquidation_Value_SPOT.ipynb) | Alert value under leverage and forced liquidation | `leverage_value_v1` |  | — |
| [25](../notebooks/25_BTC_Leverage_MarkPrice_Funding_SPOT.ipynb) | Leverage value with futures prices, mark-price liquidation and funding | `leverage_markprice_v1` | Binance futures klines, mark price, funding | — |
| [26](../notebooks/26_BTC_Onset_Aware_Training_Budget_Null_SPOT.ipynb) | Training for pre-onset warning; budget-preserving random comparison | `onset_aware_training_v1` |  | §5.2, 7, 8.1, 8.3 |
| [27](../notebooks/27_BTC_Case_Oct2026_Binance_Incident_SPOT_v2.ipynb) | Case study: Binance security incident and Bitcoin decline, 29 to 30 October 2026 | `case_oct2026_v1` | Binance spot and REST API, Coinbase API | §4.4 |
| [28](../notebooks/28_BTC_Cooldown_Blind_Spots_SPOT.ipynb) | Where missed events were blocked: signal, judging window, cooldown, budget | `cooldown_blind_spots_v1` |  | §7 |
| [29](../notebooks/29_Ranking_Reversal_and_VIX_Baseline.ipynb) | Uncertainty of the ranking reversal; the same baselines on a daily VIX design | `ranking_vix_baseline_v1` | FRED, Cboe VIX, Yahoo Finance | §5.1, 6.3 |
| [29b](../notebooks/29b_VIX_Only_Rerun_v3.ipynb) | Part B of notebook 29 (VIX only) with a seeded LASSO | `ranking_vix_baseline_v1` | FRED, Cboe VIX, Yahoo Finance | — |
| [30a](../notebooks/30a_Label_Only_Audit_Crash_Labels.ipynb) | Label-only audit of common crash labels | `label_only_audit_v1` | FRED VIX (file of notebook 29), Yahoo Finance, French data library | — |
| [30b](../notebooks/30b_Label_Convention_Diagnostic_v2.ipynb) | Model-free diagnostic of reconstructed crash-label conventions | `label_convention_diagnostic_v2` | FRED VIX (file of notebook 29), Yahoo Finance, French data library | — |
| [31](../notebooks/31_Prospective_Test_Frozen_Systems_Dec2026_start.ipynb) | Preregistered prospective test of the frozen systems from December 2026 (hash mode and analysis mode) | `prospective_test_v1` |  | §9.4 |
| [32](../notebooks/32_Robustness_Checks_Reviewer_SPOT_v2.ipynb) | Robustness checks | `robustness_checks_v1` |  | §6.2, 6.3, 6.4, 6.5, 7 |
| [33](../notebooks/33_Shapley_Decomposition_Bootstrap_SPOT.ipynb) | Bootstrap intervals for the ranking-change decomposition (Proposition 3) | `shapley_decomposition_v1` |  | §6.3; App. A |
| [34](../notebooks/34_Exact_Conditional_Null_Budget_Audit_SPOT_PREFLIGHT_v2.ipynb) | Exact conditional version of the budget-preserving null | `exact_conditional_null_v1` |  | §7; App. H.5, App. H.7 |
| [35](../notebooks/35_Sequential_Sensitivity_Cooldown_SPOT_AUDITED_r2.ipynb) | Sequential sensitivity analysis for alert-timing bias under the cooldown | `sequential_sensitivity_v1` |  | §7; App. H.5, App. H.6, App. H.7 |
| [36](../notebooks/36_Review_R7_Checks_SPOT.ipynb) | Further checks: negative buffer, budget nulls for M1 and VPIN, event bootstrap, regime granularity, label-only grid | `review_r7_checks_v1` |  | §5.2, 5.4, 6.2, 7 |
| [37](../notebooks/37_Altcoin_Event_Level_SPOT.ipynb) | Altcoin backcast read by event: clusters, effect sizes, lead times, M0 versus M1 | `altcoin_event_level_v1` |  | §8.2 |
| [38](../notebooks/38_External_Equity_Indices.ipynb) | Daily crashes in ten equity indices: preregistered historical external validation | `external_equity_indices_v1` | Yahoo Finance | §8.4 |
| [38b](../notebooks/38b_External_Equity_Secondary.ipynb) | Two registered secondary outputs of notebook 38, computed after the lock | `external_equity_indices_v1b_secondary` |  | §8.4 |
| [39](../notebooks/39_Design_Constant_SPOT.ipynb) | Rosenbaum-type Γ and step-wise Λ on the design of M0 | `design_constant_v1` |  | App. H.7 |
| [40](../notebooks/40_Design_Groups_SPOT.ipynb) | Whether the design of M0 splits into the groups of Lemma H.3 | `design_groups_v1` |  | App. H.3, App. H.7 |
| [41](../notebooks/41_Complement_Fragility_SPOT.ipynb) | Complement-set intervals and fragility of the 24-hour ranking reversal | `complement_fragility_v1` |  | §6.5, 7 |
| [42](../notebooks/42_Placebo_Reference_SPOT.ipynb) | Placebo events: decomposition and reversal from event placement alone | `placebo_reference_v1` |  | §7 |
| [43](../notebooks/43_Altcoin_Dev_Period_SPOT.ipynb) | Ten altcoins, 2023-01 to 2026-04: decomposition and 24-hour reversal with frozen coin models | `altcoin_dev_period_v1` |  | §8.2 |
| [44](../notebooks/44_Altcoin_Dev_Period_LOCO.ipynb) | Leave-one-coin-out check of notebook 43 | `altcoin_dev_period_loco_v1` |  | §8.2 |
| [45](../notebooks/45_Event_Weighted_PreOnset_SPOT.ipynb) | Event-weighted pre-onset AUC | `event_weighted_preonset_v1` |  | §6.3, 7 |
| [46](../notebooks/46_Spectral_Onset_Incremental_SPOT.ipynb) | Incremental spectral information for pre-onset warning | `spectral_onset_incremental_v1` |  | §1, 7, 9.3; OA I.1, I.6 |
| [47](../notebooks/47_Nonlinear_Spectral_Interactions_SPOT.ipynb) | Nonlinear spectral interactions as a pre-onset signal | `spectral_nonlinear_interactions_v1` |  | §1, 7, 9.3; OA I.1, I.3, I.4, I.6 |
| [48](../notebooks/48_Onset_Spectral_Mechanism_Diagnostic_SPOT.ipynb) | Sources of the extra onset-anchored catches of HGB with spectral features | `spectral_onset_mechanism_diagnostic_v1` |  | §1, 7, 9.3; OA I.1, I.4 |
| [49](../notebooks/49_Spectral_Timing_Threshold_Counterfactual_Holdout_SPOT.ipynb) | HGB spectral warning mechanism: four experiments | `spectral_mechanism_four_tests_v1` |  | §1, 7, 9.3; OA I.1, I.4, I.5 |
| [50](../notebooks/50_Spectral_Tie_Timing_Policy_Development_SPOT.ipynb) | Spectral timing and score plateaus as alert policies | `spectral_tie_timing_policy_development_v1` |  | §1, 7, 9.3; OA I.1, I.2, I.5 |
| [51](../notebooks/51_CrossCoin_Leading_Stress_Spectral_SPOT.ipynb) | Lagged cross-coin downside stress and spectral warning | `crosscoin_leading_stress_spectral_v1` |  | §1, 4.3, 7, 9.3; OA I.1, I.2, I.3, I.6 |

Notebooks 29 and 29b write to the same folder. Notebooks 21, 22, 32 and 36 hold robustness checks that I ran in response to comments on drafts of the paper; their specification titles and the folder name `review_r7_checks_v1` keep the labels I gave those checks at the time. The [README](../README.md#registered-notebooks-31-and-38) gives the registered hash of notebook 31.

## Scripts and checks

| Paper item | Script | Output |
|---|---|---|
| Figure 1 | `scripts/fig1_schematic.py` | `scripts/output/fig1_schematic.png` |
| Section 3.6, Appendix H (small designs) | `scripts/appH_toy_checks.py` | printed |
| Appendix A (lead-time bound, budget check), Table 7 | `scripts/lead_feasibility_and_event_split.py` | `scripts/output/1_lead_feasibility.csv`, `2_budget_check.csv`, `3_event_split.csv` |
| Appendix H.3, H.7 (constructions) | `scripts/nb39_construction_checks.py` | printed |
| Section 8.2, Table B.9 | `scripts/altcoin_effect_size.py` | printed |
| Figure 5, Table 10 | `scripts/fig5_ladder.py` | `scripts/output/fig5_ladder.png` |
| Section 9.4 (precision of P1) | `scripts/p1_precision.py` | printed |
| Figure B.1, Table B.3 | `scripts/figB1_shapley_terms.py` | `scripts/output/figB1_shapley_terms.png` |
| Table B.19 | `scripts/loco44_summary.py` | printed |
| Online Appendix I.7 | `checks/half_month_null/half_month_null.py` | `checks/half_month_null/result_*.json` |

