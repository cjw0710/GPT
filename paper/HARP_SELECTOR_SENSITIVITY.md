# HARP-Select Threshold Sensitivity

This diagnostic evaluates fixed selector thresholds on the stored validation and test outcomes.
It is not used to retune the paper threshold. The manuscript rule remains fixed at `z=1.96`.

## Overall Sensitivity

| z | ESep selections | Macro test acc. | Mean oracle regret (pp) | Oracle-match rate |
|---:|---:|---:|---:|---:|
| 0 | 48/80 | 64.59 | +0.42 | 88.8 |
| 0.5 | 41/80 | 64.87 | +0.14 | 91.2 |
| 1 | 34/80 | 64.75 | +0.26 | 83.8 |
| 1.645 | 31/80 | 64.71 | +0.30 | 80.0 |
| 1.96 | 30/80 | 64.72 | +0.29 | 81.2 |
| 2.58 | 26/80 | 64.36 | +0.65 | 76.2 |

## Dataset Readout At z=1.96

| Dataset | ESep selections | Mean test acc. | Mean oracle regret (pp) | Oracle-match rate |
|---|---:|---:|---:|---:|
| actor | 0/10 | 35.08 | +0.78 | 40.0 |
| amazon-ratings | 0/10 | 45.43 | +0.75 | 30.0 |
| chameleon | 10/10 | 63.42 | +0.00 | 100.0 |
| cornell | 0/10 | 74.86 | +0.81 | 80.0 |
| roman-empire | 10/10 | 78.51 | +0.00 | 100.0 |
| squirrel | 10/10 | 46.48 | +0.00 | 100.0 |
| texas | 0/10 | 86.49 | +0.00 | 100.0 |
| wisconsin | 0/10 | 87.45 | +0.00 | 100.0 |

## Validation-Margin Calibration

`margin z` is `val_diff / standard_error`, where positive values favor HARP-ESep.

| Margin bin | Splits | Mean margin z | ESep test-win rate | Mean ESep-HARP test diff (pp) | Selected by z=1.96 |
|---|---:|---:|---:|---:|---:|
| <0 | 27 | -0.87 | 14.8 | -6.25 | 0.0 |
| [0, 1) | 19 | 0.36 | 42.1 | -2.46 | 0.0 |
| [1, 1.96) | 4 | 1.30 | 75.0 | +0.71 | 0.0 |
| >=1.96 | 30 | 4.01 | 100.0 | +6.70 | 100.0 |

## Interpretation

The fixed manuscript threshold selects HARP-ESep on 30/80 splits.
Lower thresholds increase ESep selections and can recover small positive branch advantages, but choosing such a threshold after seeing test outcomes would be post-hoc model selection.
The slightly higher macro average at lower thresholds is therefore diagnostic only; it is not used to change the manuscript rule.
Lower thresholds also risk selecting the no-self expert on regimes where WebKB-style self evidence is important.
The calibration table shows why the paper keeps the conservative rule: large positive validation margins correspond to stable ESep test wins, while small positive margins are mixed and should remain diagnostic rather than a source of test-tuned claims.
