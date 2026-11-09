# results/

This folder holds the stored outputs of the notebooks, which I copied from my Colab Drive folder; `../MANIFEST.sha256` lists their hashes.

Notebook 01 writes the eight files at this level (`*.csv`, `*.json`, `cost_comparison.png`), and each subfolder comes from the notebook listed below; [`../docs/NOTEBOOKS.md`](../docs/NOTEBOOKS.md) describes each notebook.

A `*_SPEC.json` file is the specification that a notebook wrote before its evaluation, and a `*_SPEC.replaced_<UTC time>.json` file is a specification that the notebook replaced at that time. A `*.lock` file records a one-shot evaluation (time and specification hash). The `*.joblib` files are frozen models, in `models/` except two in `external_equity_indices_v1/`. `RESEARCH_PACKAGE_v1/RESEARCH_PACKAGE_v1.zip` holds the other ten files of that folder.

| Folder | Notebook | Cited in |
|---|---|---|
| `altcoin_dev_period_loco_v1/` | 44 | §8.2 |
| `altcoin_dev_period_v1/` | 43 | §8.2 |
| `altcoin_event_level_v1/` | 37 | §8.2 |
| `capacity_design_v1/` | 09 | — |
| `case_oct2026_v1/` | 27 | §4.4 |
| `causal_alert_selection_v1/` | 05 | — |
| `complement_fragility_v1/` | 41 | §6.5, 7 |
| `contagion_capacity_leadtime_v1/` | 20 | — |
| `cooldown_blind_spots_v1/` | 28 | §7 |
| `crosscoin_leading_stress_spectral_v1/` | 51 | §1, 4.3, 7, 9.3; OA I.1, I.2, I.3, I.6 |
| `crossvenue_information_v1/` | 08 | — |
| `design_constant_v1/` | 39 | App. H.7 |
| `design_groups_v1/` | 40 | App. H.3, App. H.7 |
| `detection_value_v1/` | 15 | — |
| `drawdown_trigger_v1/` | 23 | §5.1 |
| `early_warning_reliability_v1/` | 12 | §8.1 |
| `event_definition_eligibility_audit/` | 02 | — |
| `event_weighted_preonset_v1/` | 45 | §6.3, 7 |
| `exact_conditional_null_v1/` | 34 | §7; App. H.5, App. H.7 |
| `external_equity_indices_v1/` | 38 | §8.4 |
| `external_equity_indices_v1b_secondary/` | 38b | §8.4 |
| `FINAL_frozen_policy_May_Sep_2026_v2_source_verified/` | 07 | §4.4 |
| `FINAL_risk_adaptive_alert_capacity_v1/` | 06 | — |
| `gate_ablation_price_only_v1/` | 03 | — |
| `HOLDOUT17_multicoin_2018_2019_v1/` | 17 | §8.2, 8.3 |
| `HOLDOUT_2018_2020_confirmatory_v1/` | 11 | §8.1 |
| `horizon_onset_power_v1/` | 19 | §7, 8.1 |
| `label_convention_diagnostic_v2/` | 30b | — |
| `label_only_audit_v1/` | 30a | — |
| `leverage_markprice_v1/` | 25 | — |
| `leverage_value_v1/` | 24 | — |
| `methods_definitions_v1/` | 16 | §3.5, 5.2, 8.3 |
| `multicoin_shared_budget_v1/` | 10 | §4.4 |
| `onset_aware_training_v1/` | 26 | §5.2, 7, 8.1, 8.3 |
| `onset_positioning_volatility_v1/` | 21 | §5.2, 7 |
| `peak_window_v1/` | 22 | §5.2, 6.2, 7 |
| `placebo_reference_v1/` | 42 | §7 |
| `prospective_test_v1/` | 31 | §9.4 |
| `ranking_vix_baseline_v1/` | 29, 29b | §5.1, 6.3 |
| `RESEARCH_PACKAGE_v1/` | 14 | — |
| `review_r7_checks_v1/` | 36 | §5.2, 5.4, 6.2, 7 |
| `robustness_checks_v1/` | 32 | §6.2, 6.3, 6.4, 6.5, 7 |
| `sequential_sensitivity_v1/` | 35 | §7; App. H.5, App. H.6, App. H.7 |
| `shapley_decomposition_v1/` | 33 | §6.3; App. A |
| `spectral_mechanism_four_tests_v1/` | 49 | §1, 7, 9.3; OA I.1, I.4, I.5 |
| `spectral_nonlinear_interactions_v1/` | 47 | §1, 7, 9.3; OA I.1, I.3, I.4, I.6 |
| `spectral_onset_incremental_v1/` | 46 | §1, 7, 9.3; OA I.1, I.6 |
| `spectral_onset_mechanism_diagnostic_v1/` | 48 | §1, 7, 9.3; OA I.1, I.4 |
| `spectral_tie_timing_policy_development_v1/` | 50 | §1, 7, 9.3; OA I.1, I.2, I.5 |
| `spot_futures_same_event_v1/` | 13 | — |
| `standard_protocol_inflation_v1/` | 18 | §5.1, 5.4, 6.2, 7 |
| `systems_engineering_scheduler_oracle_value_v1/` | 04 | — |
