# HARP-Select Validation-Calibrated Routing

This note records a candidate method diagnostic, not a final main-manuscript
claim. HARP-Select addresses the observed specialization boundary:

- self-loop HARP-GNN is stronger on Texas, Wisconsin, and Cornell;
- no-self HARP-ESep is much stronger on Chameleon and Squirrel; and
- neither HARP-Adaptive nor HARP-Blend reliably learns that routing end to end.

## Selection Rule

HARP-Select trains HARP-GNN and HARP-ESep independently with the same split and
optimization budget. It selects HARP-ESep only when its validation advantage is
larger than a conservative normal-approximation uncertainty threshold:

```text
select HARP-ESep iff
val(ESep) - val(HARP) > 1.96 * sqrt(
    val(HARP) * (1 - val(HARP)) / n_val
  + val(ESep) * (1 - val(ESep)) / n_val
)
```

Otherwise, it retains HARP-GNN. The multiplier `1.96` is fixed by the
conventional two-sided 95% normal critical value. Test labels are not used in
the selection rule. The independence approximation is conservative when the two
validation accuracies are positively correlated.

Per-split inputs and decisions are stored in
`results/harp_select_diagnostics.csv`. The selected result rows are stored in
`results/harp_select.csv`.

## Ten-Split Results

| Dataset | HARP-GNN | HARP-ESep | HARP-Select | ESep selections |
|---|---:|---:|---:|---:|
| Texas | 86.49 +/- 4.59 | 75.41 +/- 4.84 | 86.49 +/- 4.59 | 0/10 |
| Wisconsin | 87.45 +/- 3.23 | 82.16 +/- 4.57 | 87.45 +/- 3.23 | 0/10 |
| Cornell | 74.86 +/- 4.60 | 68.92 +/- 5.87 | 74.86 +/- 4.60 | 0/10 |
| Actor | 35.08 +/- 1.36 | 35.62 +/- 1.09 | 35.08 +/- 1.36 | 0/10 |
| Chameleon | 55.90 +/- 2.49 | 63.42 +/- 2.24 | 63.42 +/- 2.24 | 10/10 |
| Squirrel | 36.66 +/- 1.66 | 46.48 +/- 1.10 | 46.48 +/- 1.10 | 10/10 |
| Roman-Empire | 75.76 +/- 0.63 | 78.51 +/- 0.57 | 78.51 +/- 0.57 | 10/10 |
| Amazon-Ratings | 45.43 +/- 0.71 | 45.94 +/- 0.48 | 45.43 +/- 0.71 | 0/10 |

On Roman-Empire, HARP-ESep improves over HARP-GNN by 2.75 percentage points,
wins all 10 fixed splits, and is significant under the paired t-test
(`p < 0.001`) and exact sign-flip test after Holm correction (`p = 0.0039`).
The frozen HARP-Select rule chooses HARP-ESep on all 10 splits.

On Amazon-Ratings, HARP-ESep is higher by 0.50 percentage points in mean
accuracy, but the difference is not significant (`p = 0.172`). Its per-split
validation advantages do not exceed the fixed uncertainty threshold, so
HARP-Select conservatively retains HARP-GNN on all 10 splits. This incurs a mean
oracle regret of 0.75 percentage points and exposes a real limitation of
single-split confidence routing when branch differences are small.

Across the original six datasets, the unweighted macro average rises from 62.74
for HARP-GNN to 65.63 for HARP-Select. The strongest implemented non-HARP
baseline on each dataset has a corresponding macro average of 65.32. This macro
average is a descriptive summary, not a significance claim.

## Threshold Sensitivity

The companion diagnostic `paper/HARP_SELECTOR_SENSITIVITY.md` evaluates several
fixed selector thresholds on the stored validation and test outcomes. This
analysis is not used to retune the manuscript threshold: the paper rule remains
fixed at `z=1.96`.

At `z=1.96`, HARP-Select chooses HARP-ESep on 30/80 splits, obtains 64.72 macro
mean test accuracy over the eight selector datasets, and has 0.29 percentage
points of mean oracle regret. The high-margin bin
`val_diff / standard_error >= 1.96` contains 30 splits, and HARP-ESep is the
better test branch in all of them. Lower thresholds can recover small positive
branch advantages, but adopting such thresholds after seeing test outcomes would
be post-hoc model selection. They are therefore reported only as diagnostics.

## Training Cost

The companion diagnostic `paper/HARP_SELECTOR_COST.md` reads the recorded
`elapsed_sec` values from the reported branch runs. On the eight selector
datasets, the macro mean two-expert training cost is 95.0 seconds per split,
which is 1.72x the self-loop HARP expert on average. Across all 80 selector
runs, the two experts account for 2.11 artifact-local wall-clock hours.

This supports a bounded cost claim: HARP-Select is reasonable for an auditable
fixed-split benchmark study, but it is not an efficiency method. A deployment
version should use shared encoders, early branch screening, or distillation.

## Paired Evidence

HARP-Select versus the strongest implemented non-HARP baseline:

| Dataset | Baseline | Difference | Paired t-test |
|---|---|---:|---:|
| Texas | MixHop | +1.89 pp | 0.132 |
| Wisconsin | H2GCN | +1.96 pp | 0.085 |
| Cornell | MLP | -1.62 pp | 0.405 |
| Actor | LINKX | -0.59 pp | 0.245 |
| Chameleon | H2GCN | -0.79 pp | 0.395 |
| Squirrel | LINKX | +1.01 pp | 0.273 |

No row is significant at `p < 0.05`. Exact paired sign-flip tests with Holm
correction also yield no significant rows. Bootstrap intervals, win/tie/loss
counts, Wilcoxon tests, and adjusted p-values are stored in
`results/harp_select_robust_tests.csv`.

## Interpretation

HARP-Select removes the two significant Chameleon/Squirrel deficits of the
original HARP-GNN evaluation while retaining its Texas/Wisconsin behavior.
Unlike HARP-Adaptive, its decisions align with the observed specialist boundary
on the original 60 fixed splits. On the two external datasets, it correctly
detects the large, stable HARP-ESep advantage on Roman-Empire and remains
conservative on the smaller, non-significant Amazon-Ratings difference.

This is promising evidence for a confidence-aware specialist formulation, but
it is not yet sufficient for a top-conference main claim:

1. It trains two full models and therefore increases training cost.
2. The 95% normal threshold remains conservative when a branch has a small but
   stable advantage, as observed on Amazon-Ratings.
3. Stronger official-code heterophily baselines remain missing on both the
   original and external datasets.
4. The three binary critical-heterophily datasets require ROC-AUC support before
   they can be added under their official protocol.
5. The current result establishes competitive balance, not statistically
   significant superiority.

The next go/no-go step is to compare the frozen selector and both specialists
against stronger official implementations on Roman-Empire and Amazon-Ratings,
then add ROC-AUC support for Minesweeper, Tolokers, and Questions.
