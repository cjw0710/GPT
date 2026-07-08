# HARP-GNN Scientific Audit

This report is generated from the current CSV artifacts. It is meant to keep the paper narrative aligned with the actual evidence before submission.

## Executive Readout

- HARP-GNN is best on 2/12 reported benchmark rows and within 1 percentage point of the best model on 1 additional row.
- Paired tests currently show 0 significant positive HARP-GNN margins and 2 significant negative margins.
- The strongest defensible claim is not state of the art; it is that low/high-pass residual fusion is useful on WebKB-like heterophily and easy to audit.
- The main scientific risk is external validity: Chameleon and Squirrel favor H2GCN/LINKX by significant margins, and Planetoid favors established low-pass or simplified propagation baselines.
- Binary ROC-AUC support is implemented; Minesweeper and Tolokers have complete 10-split branch comparisons, while Questions remains smoke-only.

## Top-Conference Readiness Gate

Status: The submission-format artifact is internally consistent, but the current scientific evidence is not yet strong enough for an AAAI main-track claim of broad competitive superiority.

Decision distinction:

- Format/readiness status: green. The paper source, compiled PDF, checklist, packages, and verifiers are synchronized.
- Scientific competitiveness status: red-amber. The current result set is useful and reproducible, but it does not yet clear a top-conference evidence bar.

Evidence trigger: HARP-GNN is best on 2/12 rows, within 1 pp on 1 additional row, has 0 significant positive paired margins, and has 2 significant negative paired margins.

| Priority | Risk | Current evidence | Required action before treating the draft as AAAI-main competitive |
|---|---|---|---|
| P0 | Competitive evidence below main-track bar | No significant positive paired margins; significant deficits on Chameleon and Squirrel | Either improve the method or re-scope the contribution to a narrower, explicitly diagnostic claim |
| P0 | Baseline coverage is incomplete for a heterophily paper | Implemented baselines omit official-code FAGCN, BernNet, and newer 2025--2026 heterophily methods | Add license-compatible official or carefully reproduced baselines, with the same fixed splits and audit rows |
| P0 | External strong-baseline coverage is incomplete | Roman-Empire and Amazon-Ratings include the implemented non-HARP suite, but still omit official-code FAGCN, BernNet, and newer 2025--2026 baselines | Add license-compatible official or carefully reproduced newer baselines on the external datasets before making broad competitiveness claims |
| P1 | Binary critical-heterophily selector calibration is incomplete | ROC-AUC support and complete Minesweeper/Tolokers branch comparisons are present, but Questions remains smoke-only and no ROC-AUC-specific selector exists | Finish Questions if claiming the full binary suite, and add a ROC-AUC-specific calibration before applying HARP-Select to binary datasets |
| P1 | Homophily fallback is weak | Synthetic high-homophily and Planetoid checks favor low-pass or simplified propagation baselines | Add an adaptive low-pass fallback, regularizer, or dataset-level branch prior and re-run Planetoid/synthetic checks |
| P1 | Gate signal claim is fragile | HARP-NoSignal is close to the full model and better on Cornell | Treat local feature variation as a hypothesis, or redesign the gate around learned branch representations |
| P1 | Statistical support remains thin | Bootstrap intervals, Wilcoxon tests, exact sign-flip tests, Holm correction, and fixed-threshold sensitivity are available for HARP-Select, but each dataset still has only 10 fixed splits and no significant positive margin | Add external datasets, more repeated seeds where valid, and keep the sensitivity diagnostic synchronized with regenerated selector outputs |
| P2 | Presentation polish can still improve | The framework figure and official PDF are present, but scientific framing is conservative | Keep visual polish, but prioritize stronger evidence over cosmetic changes |

Go/no-go rule for the next revision:

- Do not call the draft AAAI-main ready until it either obtains significant positive evidence against strong heterophily baselines on at least several benchmark rows, or reframes itself as a focused diagnostic artifact with a clearly limited claim.
- Keep the manuscript free of state-of-the-art, broad superiority, and significant-gain language unless regenerated results support those claims.

## Benchmark Landscape

### Synthetic Homophily Sweep

| Dataset | HARP-GNN | Best model | Best acc. | HARP rank | Gap to best (pp) |
|---|---:|---|---:|---:|---:|
| synthetic_h020 | 47.04 +/- 2.25 | MLP | 48.70 +/- 2.10 | 2/5 | -1.67 |
| synthetic_h050 | 58.33 +/- 4.81 | APPNP | 62.78 +/- 3.47 | 4/5 | -4.44 |
| synthetic_h080 | 87.96 +/- 1.70 | GCN | 98.89 +/- 1.47 | 4/5 | -10.93 |

### Planetoid Citation Checks

| Dataset | HARP-GNN | Best model | Best acc. | HARP rank | Gap to best (pp) |
|---|---:|---|---:|---:|---:|
| citeseer | 68.36 +/- 1.55 | GCN | 71.80 +/- 0.27 | 4/5 | -3.44 |
| cora | 78.56 +/- 0.79 | APPNP | 82.92 +/- 0.29 | 4/5 | -4.36 |
| pubmed | 63.40 +/- 2.63 | SGC | 77.24 +/- 0.29 | 3/5 | -13.84 |

### WebKB Heterophily

| Dataset | HARP-GNN | Best model | Best acc. | HARP rank | Gap to best (pp) |
|---|---:|---|---:|---:|---:|
| cornell | 74.86 +/- 4.60 | MLP | 76.49 +/- 3.83 | 4/9 | -1.62 |
| texas | 86.49 +/- 4.59 | HARP-GNN | 86.49 +/- 4.59 | 1/9 | +0.00 |
| wisconsin | 87.45 +/- 3.23 | HARP-GNN | 87.45 +/- 3.23 | 1/9 | +0.00 |

### Larger Geom-GCN Heterophily

| Dataset | HARP-GNN | Best model | Best acc. | HARP rank | Gap to best (pp) |
|---|---:|---|---:|---:|---:|
| actor | 35.08 +/- 1.36 | LINKX | 35.67 +/- 1.22 | 3/9 | -0.59 |
| chameleon | 55.90 +/- 2.49 | H2GCN | 64.21 +/- 2.66 | 6/9 | -8.31 |
| squirrel | 36.66 +/- 1.66 | LINKX | 45.48 +/- 2.33 | 8/9 | -8.82 |

## Paired-Split Evidence

### WebKB

| Dataset | Baseline | HARP-GNN | Baseline | Diff (pp) | p-value | Interpretation |
|---|---|---:|---:|---:|---:|---|
| cornell | MLP | 74.86 | 76.49 | -1.62 | 0.405 | negative, not significant |
| texas | MixHop | 86.49 | 84.59 | +1.89 | 0.132 | positive, not significant |
| wisconsin | H2GCN | 87.45 | 85.49 | +1.96 | 0.0848 | positive, not significant |

### Actor/Chameleon/Squirrel

| Dataset | Baseline | HARP-GNN | Baseline | Diff (pp) | p-value | Interpretation |
|---|---|---:|---:|---:|---:|---|
| actor | LINKX | 35.08 | 35.67 | -0.59 | 0.245 | negative, not significant |
| chameleon | H2GCN | 55.90 | 64.21 | -8.31 | 7.31e-06 | negative significant |
| squirrel | LINKX | 36.66 | 45.48 | -8.82 | 6.11e-07 | negative significant |

## Mechanistic Evidence

### Ablation Signals

- WebKB average HARP-GNN accuracy: 82.93.
- HARP-Low average: 63.09; HARP-High average: 77.60. This supports the residual high-pass branch as the main useful component.
- HARP-NoSignal average: 82.66. The current handcrafted feature-variation gate signal should remain a bounded claim.
- Scalar-gate variants average 77.25 and 77.30, below the feature-wise gate on WebKB.

### Efficiency and Capacity

- HARP-GNN has 331.9K trainable parameters on WebKB, close to MixHop (328.1K) and larger than H2GCN (126.5K).
- CPU time per WebKB split is 4.77s for HARP-GNN, 2.37s for MixHop, and 0.73s for H2GCN.
- The paper can claim modest absolute runtime on WebKB, but not speed superiority.

## Structural HARP Diagnostics

This is a bounded two-seed diagnostic, not a primary manuscript claim.
HARP-X appends a LINKX-style sparse adjacency branch to the residual low/high-pass HARP fusion.
HARP-SGate uses the same sparse adjacency evidence to condition the low/high-pass gate instead of concatenating it into the classifier.
HARP-ESep instead separates ego and no-self-neighbor evidence before applying hidden-space residual HARP filtering.
HARP-Adaptive learns a node-wise selector between the original self-loop HARP branch and the no-self HARP-ESep branch.
HARP-Blend is a more conservative graph-level logit mixture with auxiliary branch supervision.

| Dataset | HARP-GNN | HARP-X | HARP-SGate | HARP-ESep | HARP-Adaptive | HARP-Blend | Best diagnostic model | Best structural gap (pp) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| actor | 35.39 +/- 1.30 | 35.53 +/- 0.28 | 35.30 +/- 0.60 | 35.76 +/- 1.63 | 36.58 +/- 0.84 | 35.95 +/- 0.42 | HARP-Adaptive | +0.00 |
| chameleon | 55.37 +/- 0.78 | 56.91 +/- 2.33 | 56.03 +/- 1.71 | 66.12 +/- 2.95 | 57.35 +/- 2.64 | 55.92 +/- 0.62 | HARP-ESep | +0.00 |
| cornell | 74.32 +/- 1.91 | 75.68 +/- 0.00 | 70.27 +/- 3.82 | 68.92 +/- 13.38 | 71.62 +/- 1.91 | 75.68 +/- 7.64 | H2GCN | -1.35 |
| squirrel | 37.13 +/- 1.15 | 39.10 +/- 0.82 | 35.54 +/- 1.09 | 47.31 +/- 0.75 | 40.20 +/- 3.60 | 36.60 +/- 0.95 | LINKX | -0.00 |
| texas | 86.49 +/- 3.82 | 85.14 +/- 1.91 | 86.49 +/- 3.82 | 75.68 +/- 0.00 | 77.03 +/- 1.91 | 83.78 +/- 3.82 | HARP-GNN | +0.00 |
| wisconsin | 84.31 +/- 2.77 | 79.41 +/- 1.39 | 82.35 +/- 5.55 | 77.45 +/- 6.93 | 78.43 +/- 5.55 | 81.37 +/- 6.93 | HARP-GNN | -1.96 |

Interpretation: HARP-X and HARP-SGate argue against simple adjacency late fusion or structure-conditioned gates as top-conference fixes.
HARP-ESep is the first promising structural candidate on the larger heterophily rows, and the separate full 10-split run below confirms that signal.
The first two adaptive follow-ups do not close the gap: HARP-Adaptive over-selects ESep on WebKB and under-selects it on Chameleon/Squirrel, while HARP-Blend is stabler on WebKB but too conservative on the larger heterophily rows.

## HARP-ESep Full Split Candidate

HARP-ESep is a 10-split candidate run on Actor, Chameleon, and Squirrel. It is not yet the main manuscript model because it damages WebKB in the two-seed diagnostic, but it directly addresses the larger-heterophily P0 risk.

| Dataset | HARP-ESep | Best model with candidate | Best acc. | HARP-ESep rank | Gap to best (pp) |
|---|---:|---|---:|---:|---:|
| actor | 35.62 +/- 1.09 | LINKX | 35.67 +/- 1.22 | 2/10 | -0.05 |
| chameleon | 63.42 +/- 2.24 | H2GCN | 64.21 +/- 2.66 | 2/10 | -0.79 |
| squirrel | 46.48 +/- 1.10 | HARP-ESep | 46.48 +/- 1.10 | 1/10 | +0.00 |

Paired tests against the strongest non-HARP-ESep baseline:

| Dataset | Baseline | HARP-ESep | Baseline | Diff (pp) | p-value |
|---|---|---:|---:|---:|---:|
| actor | LINKX | 35.62 | 35.67 | -0.05 | 0.935 |
| chameleon | H2GCN | 63.42 | 64.21 | -0.79 | 0.395 |
| squirrel | LINKX | 46.48 | 45.48 | +1.01 | 0.273 |

Interpretation: HARP-ESep turns the previous large significant deficits on Chameleon and Squirrel into statistically non-significant gaps against the strongest implemented baselines, and is best on Squirrel in mean accuracy.
The remaining scientific issue is adaptivity: the same no-self ego-separated design hurts WebKB, and the first node-wise/graph-level selectors did not reliably choose the right branch. A publishable main method needs a stronger structural prior or branch-selection calibration.

## HARP-Select Validation-Calibrated Candidate

HARP-Select trains self-loop HARP and no-self HARP-ESep independently, then selects HARP-ESep only when its validation advantage exceeds a fixed `1.96 * SE` normal-approximation threshold. Test labels are not used for routing.

| Dataset | HARP-GNN | HARP-ESep | HARP-Select | ESep splits | Oracle regret (pp) |
|---|---:|---:|---:|---:|---:|
| actor | 35.08 +/- 1.36 | 35.62 +/- 1.09 | 35.08 +/- 1.36 | 0/10 | 0.78 |
| amazon-ratings | 45.43 +/- 0.71 | 45.94 +/- 0.48 | 45.43 +/- 0.71 | 0/10 | 0.75 |
| chameleon | 55.90 +/- 2.49 | 63.42 +/- 2.24 | 63.42 +/- 2.24 | 10/10 | 0.00 |
| cornell | 74.86 +/- 4.60 | 68.92 +/- 5.87 | 74.86 +/- 4.60 | 0/10 | 0.81 |
| roman-empire | 75.76 +/- 0.63 | 78.51 +/- 0.57 | 78.51 +/- 0.57 | 10/10 | 0.00 |
| squirrel | 36.66 +/- 1.66 | 46.48 +/- 1.10 | 46.48 +/- 1.10 | 10/10 | 0.00 |
| texas | 86.49 +/- 4.59 | 75.41 +/- 4.84 | 86.49 +/- 4.59 | 0/10 | 0.00 |
| wisconsin | 87.45 +/- 3.23 | 82.16 +/- 4.57 | 87.45 +/- 3.23 | 0/10 | 0.00 |

Paired HARP-Select evidence against the strongest implemented non-HARP baseline:

| Dataset | Baseline | HARP-Select | Baseline | Diff (pp) | p-value |
|---|---|---:|---:|---:|---:|
| actor | LINKX | 35.08 | 35.67 | -0.59 | 0.245 |
| chameleon | H2GCN | 63.42 | 64.21 | -0.79 | 0.395 |
| cornell | MLP | 74.86 | 76.49 | -1.62 | 0.405 |
| squirrel | LINKX | 46.48 | 45.48 | +1.01 | 0.273 |
| texas | MixHop | 86.49 | 84.59 | +1.89 | 0.132 |
| wisconsin | H2GCN | 87.45 | 85.49 | +1.96 | 0.085 |

Robustness readout: exact paired sign-flip tests with Holm correction produce 0 significant rows at `p < 0.05`.

Interpretation on the original suite: HARP-Select preserves the original HARP results on Texas, Wisconsin, Cornell, and Actor while routing every Chameleon/Squirrel split to HARP-ESep. It removes the original significant Chameleon/Squirrel deficits, but does not establish statistically significant superiority against the strongest implemented baselines.

Threshold-sensitivity diagnostic:

- At the frozen manuscript threshold `z=1.96`, the selector chooses HARP-ESep on 30/80 splits, with macro mean accuracy 64.72 and mean oracle regret 0.29 percentage points.
- The high validation-margin bin (`val_diff / SE >= 1.96`) contains 30 splits and has ESep test-win rate 100.0%.
- Lower fixed thresholds are reported only as diagnostics; the paper threshold is not retuned after seeing test outcomes.

Training-cost diagnostic:

- Recorded CPU wall-clock times give macro mean two-expert training cost 95.0 seconds per split and macro overhead 1.72x relative to the self-loop HARP expert.
- Across all 80 selector runs, the two branches account for 2.11 artifact-local wall-clock hours.
- This supports the paper's cost boundary: HARP-Select is an auditable benchmark method, not an efficiency claim.

External fixed-rule validation on the ICLR 2023 critical-heterophily benchmark:

| Dataset | HARP-GNN | HARP-ESep | Diff (pp) | p-value | HARP-Select ESep splits |
|---|---:|---:|---:|---:|---:|
| amazon-ratings | 45.43 | 45.94 | +0.50 | 0.172 | 0/10 |
| roman-empire | 75.76 | 78.51 | +2.75 | <0.001 | 10/10 |

External interpretation: Roman-Empire provides the first significant positive branch result: HARP-ESep wins all 10 splits, and the frozen selector routes all 10 splits to it (Holm-adjusted exact sign-flip p=0.0039).
Amazon-Ratings shows the conservative boundary: HARP-ESep has a small non-significant mean advantage, but no split exceeds the fixed confidence threshold, so HARP-Select retains HARP-GNN and incurs modest oracle regret.
The candidate now has external validation, but still lacks strong official-code baselines on Roman-Empire/Amazon-Ratings. The binary ROC-AUC path is implemented for branch comparisons, while selector calibration for ROC-AUC datasets remains future work.

## Binary Critical-Heterophily ROC-AUC

The binary critical-heterophily path evaluates ROC-AUC rather than accuracy. The current manuscript reports complete 10-split Minesweeper and Tolokers branch comparisons; Questions remains smoke-only and is not a main claim.

| Dataset | Metric | HARP-GNN | HARP-ESep | Diff (pp) | W/T/L | Paired p | Sign-flip Holm p | Status |
|---|---|---:|---:|---:|---:|---:|---:|---|
| minesweeper | ROC-AUC | 88.18 +/- 0.84 | 89.64 +/- 0.67 | +1.45 | 10/0/0 | <0.001 | 0.004 | reported positive branch evidence |
| tolokers | ROC-AUC | 82.79 +/- 0.83 | 79.22 +/- 0.70 | -3.56 | 0/0/10 | <0.001 | 0.004 | reported negative branch evidence |

Interpretation: Minesweeper supports the ego-separated branch under ROC-AUC, while Tolokers gives equally strong evidence in the opposite direction. These are branch-specialization results, not HARP-Select routing claims. The current selector threshold is derived from validation accuracy uncertainty and should not be reused for ROC-AUC without a separate calibration argument.
The smoke file covers minesweeper, questions, tolokers with one split per branch and is treated as loader/protocol evidence only.
The full binary branch trace currently contains 40 rows (minesweeper: 10 seeds, tolokers: 10 seeds).

## Claim Boundaries for the Current Draft

Safe claims:

- HARP-GNN gives a compact residual polynomial formulation with low-pass and high-pass bases.
- On WebKB, HARP-GNN is strongest among the implemented baselines on Texas and Wisconsin, but paired margins are not significant at p < 0.05.
- Learned WebKB filters and gates allocate substantial mass to first-order high-pass residual evidence.
- As a separate candidate diagnostic, HARP-Select uses validation-only confidence routing to remove the original significant Chameleon/Squirrel deficits without creating a significant superiority claim.
- On binary Minesweeper under ROC-AUC, HARP-ESep has a complete 10-split positive branch comparison against HARP-GNN; on Tolokers, the direction reverses and HARP-GNN is better. Neither row extends to a HARP-Select routing claim.
- The artifact includes coverage checks, implementation invariants, manuscript/package checks, and final submission-readiness validation.

Claims to avoid:

- Do not claim state-of-the-art performance.
- Do not claim significant WebKB gains.
- Do not claim speed superiority.
- Do not present Planetoid as a tuned citation benchmark result.
- Do not cite Questions as a main binary critical-heterophily result until its full ROC-AUC run is complete.

## Next Scientific Moves

1. Add strong official-code baselines on Roman-Empire and Amazon-Ratings under the same fixed masks.
2. Finish Questions under ROC-AUC if claiming the full binary suite, then design a ROC-AUC-specific calibration rule before applying HARP-Select to binary datasets.
3. Investigate distillation, shared encoders, or early branch screening to reduce the measured two-specialist training cost.
4. Improve the low-pass fallback so the model does not lose as much on homophilous/citation graphs.
5. Keep the frozen-threshold sensitivity and calibration diagnostic synchronized, without selecting the threshold on test performance.
