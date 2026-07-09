# HARP-Select Artifact Manifest

This manifest describes the files that make up the current reproducible HARP-Select research artifact for the AAAI 2027 main-track draft. It separates primary reported artifacts from auxiliary diagnostics and tuning traces.

## Primary Manuscript Artifacts

- `paper/main.tex`: Main AAAI-format manuscript source.
- `paper/references.bib`, `paper/references_additions.bib`: Bibliography files used by the manuscript.
- `paper/aaai2027.sty`, `paper/aaai2027.bst`: Official AAAI 2027 style files copied from the author kit.
- `paper/HARP_GNN_AAAI2027_official_compile.pdf`: Latest compiled official-template PDF.
- `paper/HARP_GNN_AAAI2027_official_page1.png` through `paper/HARP_GNN_AAAI2027_official_page8.png`: Rendered page previews of the latest official-template PDF.
- `paper/ReproducibilityChecklist_HARP_GNN.tex`, `paper/ReproducibilityChecklist_HARP_GNN.pdf`: Filled reproducibility checklist generated from the official author-kit template.
- `paper/AAAI27_SUBMISSION_REQUIREMENTS.md`: Local mapping from the official AAAI-27 submission instructions to the current review artifacts and automated checks.
- `paper/SCIENTIFIC_AUDIT.md`: Generated scientific audit summarizing current safe claims, unsafe claims, paired-split evidence, top-conference readiness risks, efficiency tradeoffs, and recommended next scientific moves from the CSV outputs.
- `paper/HARP_X_DIAGNOSTIC.md`: Bounded two-seed diagnostic note for HARP-X, HARP-SGate, HARP-ESep, HARP-Adaptive, and HARP-Blend structural variants. This is not a primary manuscript benchmark claim.
- `paper/HARP_SELECT_DIAGNOSTIC.md`: Validation-calibrated HARP/HARP-ESep specialist-routing note with ten-split results, paired tests, robust statistics, and explicit claim boundaries for the main HARP-Select method.
- `paper/HARP_SELECTOR_SENSITIVITY.md`: Fixed-threshold sensitivity and validation-margin calibration diagnostic for HARP-Select. This is not used to retune the manuscript threshold.
- `paper/HARP_SELECTOR_COST.md`: Recorded wall-clock training-cost diagnostic for the two-expert selector.
- `paper/HARP_GNN_AAAI2027_submission_package.zip`: Validated package containing the official PDF plus a minimal LaTeX source tree that recompiles in a clean staging directory.
- `paper/HARP_GNN_AAAI2027_supplementary_artifact.zip`: Validated code/results reproducibility artifact containing source code, configs, result CSVs, generated paper artifacts, and verification scripts.

## One-Command Rebuild

Run this from the repository root to rebuild reported tables, figures, audits, verifier checks, the filled reproducibility checklist, the official PDF, page previews, the validated submission package, and the supplementary reproducibility artifact from existing CSV outputs:

```powershell
.\scripts\rebuild_reported_artifacts.ps1
```

Run this to skip LaTeX compilation and only refresh generated report artifacts and checks:

```powershell
.\scripts\rebuild_reported_artifacts.ps1 -SkipCompile
```

The rebuild script does not retrain models. It assumes the CSV files under `results/` already exist.

Run this to rebuild only the current PDF/source submission package and validate the included source tree:

```powershell
.\scripts\build_submission_package.ps1 -SkipCompile
```

Run this to regenerate the filled reproducibility checklist from the official author-kit checklist template:

```powershell
.\scripts\build_reproducibility_checklist.ps1
```

Run this to rebuild only the supplementary code/results reproducibility artifact and validate it with the included Python verifiers:

```powershell
.\scripts\build_supplementary_artifact.ps1
```

Run this to verify the final PDF, LaTeX logs, page previews, checklist, source submission package, and supplementary reproducibility artifact:

```powershell
python scripts\verify_submission_readiness.py
```

Run this to regenerate the scientific audit report from the current CSV outputs:

```powershell
python scripts\generate_scientific_audit.py
```

Run this to verify that manuscript claims stay within the current evidence boundary:

```powershell
python scripts\verify_top_conference_claims.py
```

## Core Source Code

- `src/harp_gnn/data.py`: Dataset loaders for synthetic graphs, Planetoid, WebKB, and larger Geom-GCN format datasets.
- `src/harp_gnn/models.py`: Baseline models and HARP-GNN variants, including projected propagation, fixed-feature propagation caches, ego-separated HARP, node-wise adaptive HARP/ESep selection, and graph-level HARP/ESep logit blending.
- `src/harp_gnn/training.py`: Training loop, early stopping, and accuracy/ROC-AUC metric collection.
- `src/harp_gnn/run_experiment.py`: Multi-dataset, multi-model, multi-seed experiment runner.
- `src/harp_gnn/utils.py`: Sparse adjacency normalization, CSR-backed sparse conversion utilities, and tie-aware binary ROC-AUC.

## Reported Experiment Configs

These configs correspond to results that are reported directly in the manuscript or used by manuscript diagnostics:

- `configs/synthetic_sweep.yaml`: Controlled contextual SBM homophily sweep.
- `configs/planetoid_all.yaml`: Planetoid Cora/CiteSeer/PubMed citation checks.
- `configs/webkb.yaml`: WebKB Texas/Wisconsin/Cornell main heterophily benchmark.
- `configs/webkb_ablation.yaml`: WebKB low/high-pass and signal ablations.
- `configs/webkb_scalar_gate.yaml`: WebKB gate-granularity ablations.
- `configs/webkb_harp_diagnostics.yaml`: HARP filter and gate diagnostics.
- `configs/geom_gcn_large.yaml`: Actor/Chameleon/Squirrel 10-split larger Geom-GCN benchmarks.
- `configs/geom_gcn_harp_esep.yaml`: Full 10-split HARP-ESep candidate run on Actor/Chameleon/Squirrel.
- `configs/webkb_harp_esep.yaml`: Full 10-split HARP-ESep candidate run on Texas/Wisconsin/Cornell.
- `configs/critical_heterophily_harp.yaml`: Full 10-split HARP/HARP-ESep external validation on Roman-Empire and Amazon-Ratings.
- `configs/critical_heterophily_smoke.yaml`: Split-0 loader and training smoke test for the external datasets.
- `configs/critical_heterophily_binary_smoke.yaml`: Split-0 binary critical-heterophily ROC-AUC loader and training smoke test for Minesweeper, Tolokers, and Questions.
- `configs/harp_x_diagnostic.yaml`: Two-seed diagnostic for HARP residual fusion models augmented, conditioned, or ego-separated with structural evidence.

Additional smoke and development configs remain under `configs/` and are covered by the result audit when their output files are present.
The in-progress full binary ROC-AUC run lives under `configs/in_progress/critical_heterophily_binary_harp.yaml` so the main result audit does not treat unfinished Tolokers and Questions rows as reported claims.
The per-dataset continuation configs `configs/in_progress/critical_heterophily_binary_tolokers.yaml` and `configs/in_progress/critical_heterophily_binary_questions.yaml` write to the same full-run CSV while allowing the remaining binary ROC-AUC datasets to be resumed independently.

## Primary Result CSVs

- `results/synthetic_sweep.csv`: Synthetic homophily sweep results.
- `results/planetoid_all.csv`: Active corrected Planetoid results after the CSR sparse-conversion fix.
- `results/webkb.csv`: WebKB main benchmark results.
- `results/webkb_ablation.csv`: WebKB branch/signal ablation results.
- `results/webkb_scalar_gate.csv`: WebKB scalar-gate ablation results.
- `results/webkb_filter_weights.csv`: Aggregated learned HARP filter weights.
- `results/webkb_gate_diagnostics.csv`: Aggregated learned HARP gate diagnostics.
- `results/webkb_parameters.csv`: Parameter-count summary.
- `results/webkb_runtime.csv`: Runtime summary.
- `results/geom_gcn_large.csv`: Actor/Chameleon/Squirrel 10-split results.
- `results/geom_gcn_large_dataset_stats.csv`: Larger Geom-GCN dataset statistics.
- `results/geom_gcn_large_paired_tests.csv`: Larger Geom-GCN paired tests against the strongest non-HARP baseline.
- `results/geom_gcn_harp_esep.csv`: Full 10-split HARP-ESep candidate results on Actor/Chameleon/Squirrel.
- `results/geom_gcn_large_with_harp_esep.csv`: Merged larger Geom-GCN result table including HARP-ESep for candidate ranking and paired tests.
- `results/geom_gcn_harp_esep_paired_tests.csv`: Paired tests comparing HARP-ESep against the strongest implemented non-HARP-ESep baseline.
- `results/harp_x_diagnostic.csv`: Two-seed HARP-X, HARP-SGate, HARP-ESep, HARP-Adaptive, and HARP-Blend diagnostic results used only for method-risk triage.
- `results/webkb_harp_esep.csv`: Full 10-split WebKB HARP-ESep specialist results.
- `results/harp_select_diagnostics.csv`: Per-split validation margins, uncertainty thresholds, routing decisions, and oracle regret for HARP-Select.
- `results/harp_select.csv`: Selected HARP-Select result rows.
- `results/harp_select_with_baselines.csv`: HARP-Select merged with the six-dataset implemented baseline results.
- `results/harp_select_paired_tests.csv`: Paired t-tests against the strongest implemented non-HARP baseline.
- `results/harp_select_robust_tests.csv`: Bootstrap intervals, Wilcoxon tests, exact sign-flip tests, and Holm corrections.
- `results/harp_select_threshold_sensitivity.csv`: Dataset-level fixed-threshold sensitivity summary for HARP-Select.
- `results/harp_select_threshold_sensitivity_overall.csv`: Overall fixed-threshold sensitivity summary across the eight selector datasets.
- `results/harp_select_margin_calibration.csv`: Validation-margin bins and corresponding test-branch calibration diagnostics.
- `results/harp_select_training_cost.csv`: Dataset-level recorded training-cost summary for the two HARP-Select experts.
- `results/harp_select_training_cost_per_split.csv`: Per-split recorded wall-clock cost rows used to build the cost summary.
- `results/critical_heterophily_harp.csv`: Full Roman-Empire and Amazon-Ratings HARP/HARP-ESep results.
- `results/critical_heterophily_harp_esep_paired_tests.csv`: External paired branch comparisons.
- `results/critical_heterophily_harp_esep_robust_tests.csv`: External bootstrap, Wilcoxon, exact sign-flip, and Holm statistics.
- `results/critical_heterophily_dataset_stats.csv`: External dataset statistics and fixed split sizes.
- `results/critical_heterophily_binary_smoke.csv`: Binary ROC-AUC smoke-test rows for Minesweeper, Tolokers, and Questions.
- `results/critical_heterophily_binary_dataset_stats.csv`: Binary critical-heterophily dataset statistics and fixed split sizes.
- `results/critical_heterophily_binary_minesweeper.csv`: Complete 10-split Minesweeper ROC-AUC branch comparison.
- `results/critical_heterophily_binary_minesweeper_paired_tests.csv`: Minesweeper paired branch comparison.
- `results/critical_heterophily_binary_minesweeper_robust_tests.csv`: Minesweeper bootstrap, Wilcoxon, exact sign-flip, and Holm statistics.
- `results/critical_heterophily_binary_complete_paired_tests.csv`: Complete-only binary ROC-AUC paired branch table generated from datasets with all 10 paired splits available.
- `results/critical_heterophily_binary_complete_robust_tests.csv`: Complete-only binary ROC-AUC bootstrap, exact sign-flip, and Holm statistics.
- `results/critical_heterophily_binary_harp.csv`: Partial full binary ROC-AUC run trace. Minesweeper is complete, Tolokers currently has 4/10 paired splits, and Questions remains smoke-only; Tolokers and Questions are not manuscript claims in this revision.
- `results/result_audit.csv`: Coverage audit for all experiment configs.

Traceability files:

- `results/planetoid_all_pre_csr_fix.csv`: Pre-fix Planetoid snapshot retained for traceability.
- `results/*tuning*.csv`, `results/*smoke*.csv`: Diagnostic, smoke, and tuning traces. These are not primary manuscript claims unless explicitly summarized in `paper/main.tex`.

## Generated Manuscript Tables and Figures

Tables read directly by `paper/main.tex`:

- `paper/tables/webkb_dataset_stats.tex`
- `paper/tables/synthetic_results.tex`
- `paper/tables/planetoid_results.tex`
- `paper/tables/webkb_results.tex`
- `paper/tables/geom_gcn_large_results.tex`
- `paper/tables/geom_gcn_large_paired_tests.tex`
- `paper/tables/geom_gcn_harp_esep_results.tex`
- `paper/tables/geom_gcn_harp_esep_paired_tests.tex`
- `paper/tables/webkb_paired_tests.tex`
- `paper/tables/webkb_ablation_results.tex`
- `paper/tables/webkb_scalar_gate_results.tex`
- `paper/tables/webkb_filter_weights.tex`
- `paper/tables/webkb_gate_diagnostics.tex`
- `paper/tables/webkb_parameters.tex`
- `paper/tables/webkb_runtime.tex`
- `paper/tables/harp_x_diagnostic_results.tex`
- `paper/tables/webkb_harp_esep_results.tex`
- `paper/tables/harp_select_results.tex`
- `paper/tables/harp_select_paired_tests.tex`
- `paper/tables/harp_select_robust_tests.tex`
- `paper/tables/harp_select_threshold_sensitivity.tex`
- `paper/tables/harp_select_training_cost.tex`
- `paper/tables/critical_heterophily_dataset_stats.tex`
- `paper/tables/critical_heterophily_harp_results.tex`
- `paper/tables/critical_heterophily_harp_esep_paired_tests.tex`
- `paper/tables/critical_heterophily_harp_esep_robust_tests.tex`
- `paper/tables/critical_heterophily_binary_minesweeper_paired_tests.tex`

Figures read directly by `paper/main.tex`:

- `paper/figures/harp_select_framework.pdf`
- `paper/figures/harp_framework.pdf`
- `paper/figures/selector_audit.pdf`
- `paper/figures/selector_sensitivity.pdf`
- `paper/figures/synthetic_accuracy.png`

Additional binary ROC-AUC support tables retained in the paper artifact:

- `paper/tables/critical_heterophily_binary_smoke_results.tex`
- `paper/tables/critical_heterophily_binary_dataset_stats.tex`
- `paper/tables/critical_heterophily_binary_minesweeper_results.tex`
- `paper/tables/critical_heterophily_binary_minesweeper_robust_tests.tex`
- `paper/tables/critical_heterophily_binary_complete_paired_tests.tex`
- `paper/tables/critical_heterophily_binary_complete_robust_tests.tex`

Audit table:

- `paper/tables/result_audit.tex`

Scientific audit:

- `paper/SCIENTIFIC_AUDIT.md`

## Verification Scripts

- `scripts/check_results_coverage.py`: Checks one config/result pair for missing, extra, and duplicate rows.
- `scripts/audit_results.py`: Audits every YAML config under `configs/` and writes CSV/LaTeX coverage summaries.
- `scripts/verify_implementation.py`: Checks sparse conversion fidelity, projected HARP equivalence, fixed-feature propagation caches, and tie-aware binary ROC-AUC on deterministic toy tensors.
- `scripts/verify_manuscript_integrity.py`: Checks LaTeX inputs, figure files, table/figure labels, references, citation keys, anonymization, and AAAI submission source markers.
- `scripts/verify_reported_results.py`: Checks core reported values, paired tests, coverage, generated table text, and manuscript language smoke checks.
- `scripts/verify_top_conference_claims.py`: Checks that the manuscript avoids unsupported state-of-the-art, broad-superiority, significant-gain, or readiness claims under the current paired-test evidence.
- `scripts/verify_submission_readiness.py`: Checks final PDF/checklist/package artifacts, LaTeX logs, page previews, zip cleanliness, and whether packed sources/PDFs are synchronized with the current manuscript.
- `scripts/generate_scientific_audit.py`: Generates `paper/SCIENTIFIC_AUDIT.md` from current result CSVs for claim-boundary and reviewer-risk review.
- `scripts/build_binary_critical_tables.py`: Builds complete-only binary critical-heterophily ROC-AUC paired and robust tables from the full run trace.
- `scripts/plot_framework.py`: Generates the HARP-GNN framework diagram used in the Method section.
- `scripts/plot_harp_select_framework.py`: Generates the HARP-Select two-expert routing framework diagram used in the Method section.
- `scripts/plot_selector_audit.py`: Generates the split-level validation-surplus and held-out branch-advantage audit figure from selector diagnostics.
- `scripts/plot_selector_sensitivity.py`: Generates the frozen-threshold sensitivity plot from the aggregate selector sweep.
- `scripts/analyze_selector_sensitivity.py`: Generates HARP-Select fixed-threshold sensitivity CSVs, a LaTeX summary table, and the standalone selector-sensitivity diagnostic note.
- `scripts/analyze_selector_cost.py`: Generates HARP-Select recorded training-cost CSVs, a LaTeX summary table, and the standalone selector-cost diagnostic note.
- `scripts/check_sparse_conversion.py`: Verifies the CSR-backed sparse conversion path against direct SciPy sparse semantics.
- `scripts/fill_reproducibility_checklist.py`: Generates `paper/ReproducibilityChecklist_HARP_GNN.tex` from the official checklist template.
- `scripts/build_reproducibility_checklist.ps1`: Generates and compiles the filled reproducibility checklist.
- `scripts/compile_paper.ps1`: Compiles `paper/main.tex` with the official AAAI LaTeX style and refreshes official PDF page previews.
- `scripts/render_official_pdf_pages.ps1`: Renders `paper/HARP_GNN_AAAI2027_official_compile.pdf` into page-level PNG previews using the bundled TinyTeX Ghostscript.
- `scripts/build_submission_package.ps1`: Builds `paper/HARP_GNN_AAAI2027_submission_package.zip` with the official PDF and minimal LaTeX source tree, then validates that the staged source recompiles.
- `scripts/build_supplementary_artifact.ps1`: Builds `paper/HARP_GNN_AAAI2027_supplementary_artifact.zip` with code, configs, result CSVs, generated paper artifacts, and validation checks.
- `scripts/rebuild_reported_artifacts.ps1`: Regenerates all reported artifacts from CSV outputs, runs verification, and compiles the PDF unless `-SkipCompile` is supplied.

## Current Verification Status

The current artifact passes:

```powershell
python scripts\audit_results.py --config-dir configs --output-csv results\result_audit.csv --output-tex paper\tables\result_audit.tex
python scripts\verify_implementation.py
python scripts\verify_manuscript_integrity.py
python scripts\verify_reported_results.py
python scripts\generate_scientific_audit.py
python scripts\verify_top_conference_claims.py
.\scripts\build_reproducibility_checklist.ps1
.\scripts\compile_paper.ps1
.\scripts\build_submission_package.ps1 -SkipCompile
.\scripts\build_supplementary_artifact.ps1
python scripts\verify_submission_readiness.py
```

The latest compile produces an 8-page official-template PDF with seven pages of technical content and references beginning on page 8, eight page preview PNGs, a filled 2-page reproducibility checklist PDF, a validated submission package zip, and a validated supplementary reproducibility artifact. The LaTeX logs have no undefined citations, undefined references, fatal errors, or overfull boxes. The remaining warnings are underfull box messages from page layout.

This is a submission-format readiness statement, not a scientific go decision for AAAI main track. The generated scientific audit is the active source for top-conference risk and claim-boundary review.

## Known Scientific Boundaries

The current manuscript is intentionally conservative:

- HARP-GNN is strongest among the implemented baselines on Texas and Wisconsin, but paired WebKB margins are not significant at `p < 0.05`.
- Cornell remains negative relative to the strongest non-HARP baselines.
- On Actor, HARP-GNN is close to LINKX and the paired deficit is not significant.
- On Chameleon and Squirrel, HARP-GNN significantly trails H2GCN/LINKX.
- Planetoid results are treated as citation-loader and training-pipeline checks rather than a tuned citation-network benchmark claim.
- The binary critical-heterophily ROC-AUC support is now implemented, and Minesweeper has a complete 10-split branch comparison. Tolokers currently has 4/10 paired full-run splits and Questions remains smoke-only, so neither should be cited as a main result yet.
- The HARP-Select routing rule is calibrated for validation accuracy in the current manuscript and is not applied to ROC-AUC datasets.
- Future benchmark work should add official-code baselines such as FAGCN, BernNet, and stronger 2025--2026 heterophily methods where license-compatible.
