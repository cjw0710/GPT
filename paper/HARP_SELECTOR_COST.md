# HARP-Select Training Cost Diagnostic

This diagnostic reads the recorded `elapsed_sec` values from the reported branch runs.
It does not benchmark new hardware runs and should be interpreted as artifact-local CPU wall-clock evidence.

## Dataset-Level Cost

| Dataset | HARP sec./split | ESep sec./split | Two-expert sec./split | Cost / HARP | ESep selections |
|---|---:|---:|---:|---:|---:|
| actor | 26.0 | 15.1 | 41.1 | 1.58x | 0/10 |
| amazon-ratings | 182.7 | 116.2 | 298.9 | 1.66x | 0/10 |
| chameleon | 20.7 | 28.1 | 48.8 | 2.38x | 10/10 |
| cornell | 4.4 | 0.9 | 5.3 | 1.24x | 0/10 |
| roman-empire | 83.7 | 90.0 | 173.7 | 2.11x | 10/10 |
| squirrel | 86.8 | 92.5 | 179.2 | 2.18x | 10/10 |
| texas | 4.4 | 1.3 | 5.7 | 1.34x | 0/10 |
| wisconsin | 5.6 | 1.4 | 7.0 | 1.25x | 0/10 |

## Aggregate Readout

- Macro mean HARP time: 51.8 seconds per split.
- Macro mean ESep time: 43.2 seconds per split.
- Macro mean two-expert HARP-Select training time: 95.0 seconds per split.
- Macro mean overhead versus self-loop HARP: 1.72x.
- Across all 80 selector runs, the artifact records 2.11 wall-clock hours for the two experts combined.

## Interpretation

HARP-Select is an auditable benchmark method, not an efficiency claim.
Its selection rule requires both branches to be trained before the validation decision, so deployment-oriented versions should use shared encoders, early branch screening, or distillation.
The cost is still modest enough for the reported fixed-split study: WebKB rows are lightweight, while the external datasets and Squirrel dominate the wall-clock total.
