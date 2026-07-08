# Reviewer Risk Audit for HARP-Select

This note is an anonymous, reviewer-facing risk register for the current
AAAI-style draft. It is not a rebuttal and should not be submitted as author
discussion. Its purpose is to keep the manuscript, supplement, and code
artifact aligned with the questions reviewers are likely to ask.

## One-Sentence Positioning

HARP-Select is best framed as an auditable residual-polynomial expert-routing
framework for heterophilous graphs, not as a universal state-of-the-art
heterophily architecture.

## Likely Reviewer Concerns

| Concern | Current answer in the draft/artifact | Residual risk | Rebuttal seed |
|---|---|---|---|
| Novelty looks incremental over high-pass or spectral GNNs | Main paper explains the residual polynomial low/high-pass pair; supplement derives the filter response \(p_H(\lambda)=(1-\lambda)\sum_k\beta_k\lambda^{k-1}\). | Reviewers may still view the branch family as simple. | Emphasize the contribution is the auditable branch-specialization and frozen validation routing protocol, not only a new filter basis. |
| Selector may be hidden test-time model selection | Main paper states the rule uses validation labels only; supplement logs the margin, threshold, selected branch, oracle branch, and regret. | The normal-approximation threshold is heuristic and accuracy-specific. | Point to fixed threshold, split-level logs, no test-label access, and the explicit caveat that ROC-AUC datasets are branch evidence only. |
| Evidence is mixed and not broadly superior | Main paper and scientific audit explicitly avoid state-of-the-art claims; robust tests show no significant positive row on the original six datasets. | AAAI reviewers may still expect stronger wins. | Reframe the result as removing the significant Chameleon/Squirrel deficits while exposing where expert routing fails. |
| Baselines are not complete enough | Implemented reported baselines include MLP, GCN, SGC, APPNP, MixHop, GPR-GNN, H2GCN-style, and LINKX; the artifact also includes a local FAGCN-style smoke scaffold. | Official-code FAGCN, BernNet, and newer 2025--2026 heterophily baselines remain missing. | Acknowledge this as the main limitation; avoid leaderboard language; present Roman-Empire as branch evidence against the implemented suite only. |
| Binary critical-heterophily treatment is confusing | Main paper reports complete Minesweeper and Tolokers branch comparisons under ROC-AUC; Questions remains smoke-only, with a deferred full-run recipe under `configs/deferred/`. | The selector is not calibrated for ROC-AUC, so binary rows cannot be HARP-Select routing claims. | State that Minesweeper/Tolokers demonstrate branch specialization in opposite directions and motivate metric-specific routing, not a selector claim. |
| Framework figure may look like a pipeline rather than a method | Figure 1 shows parallel expert training, validation evidence, frozen routing, and locked test readout. | Some reviewers may want an algorithm box or pseudocode. | Supplement includes the algorithmic rule; a rebuttal can restate the four-step procedure and point to logged CSV columns. |
| Runtime doubles because both experts are trained | Main paper reports two-expert CPU cost and says this is not an efficiency claim. | Deployment-minded reviewers may dislike the cost. | Argue the artifact targets auditability; distillation/shared encoders are a natural extension, not hidden in current claims. |
| Gate signal may be weak | Ablations show HARP-NoSignal is close and sometimes better, so the paper treats feature variation as a bounded hypothesis. | The handcrafted signal may not be publishable as a central novelty. | Keep the central claim on residual branch specialization and routing, not on the handcrafted gate. |
| Reproducibility could be questioned | Artifact includes CSVs, configs, generated tables, checks for implementation invariants, reported results, claim boundaries, and package synchronization. | Raw public datasets are not bundled; reviewers must run loaders. | Point to the supplementary artifact README and result-coverage audit; raw datasets are public and intentionally not vendored. |

## Claim Boundaries To Preserve

- Safe: residual low/high-pass experts provide a compact and inspectable filter family.
- Safe: HARP-Select removes the original significant Chameleon/Squirrel deficits by routing to HARP-ESep on those splits.
- Safe: Roman-Empire gives significant positive branch evidence for HARP-ESep and the frozen selector routes all splits to it.
- Safe: Minesweeper and Tolokers show opposite ROC-AUC branch preferences, supporting branch specialization rather than a universal no-self claim.
- Safe: the deferred Questions full-run config is a reproducibility recipe, not reported evidence.
- Safe: FAGCN-style smoke results show that the local signed-edge baseline scaffold runs; they do not close the official-code baseline gap.
- Unsafe: state-of-the-art performance.
- Unsafe: broad superiority over heterophily baselines.
- Unsafe: ROC-AUC HARP-Select routing claims before metric-specific calibration.
- Unsafe: Questions as a reported binary critical-heterophily result before the full run is complete.
- Unsafe: treating the FAGCN-style smoke scaffold as official FAGCN evidence.

## Minimum Pre-Submission Checklist

1. Run `python scripts\verify_submission_readiness.py`.
2. Confirm the main PDF still has seven technical pages and references start on page 8.
3. Confirm `paper\HARP_GNN_AAAI2027_supplementary_artifact.zip` is the artifact uploaded for code/data.
4. Confirm no GitHub URL appears in the paper or supplement unless the submission system explicitly allows it.
5. Select the AAAI primary topic as machine learning representation learning or graph/probabilistic models, depending on the final OpenReview taxonomy wording.
6. Use the TL;DR: `HARP-Select routes between self-loop and ego-separated residual polynomial GNN experts using validation-only confidence evidence for heterophilous graphs.`

## If Space Opens In The Main Paper

Priority additions, in order:

1. A short algorithm box for HARP-Select.
2. One sentence connecting \(p_H(\lambda)\) to the supplement's filter-response derivation.
3. A compact note that Tolokers reverses Minesweeper, making branch specialization the core message.
4. A stronger explicit limitation that missing official-code 2025--2026 baselines preclude leaderboard claims.
