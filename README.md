# HARP-Select: Anonymous Reproducibility Artifact

This repository contains the anonymized code, configs, generated results, paper
source, and verification scripts for an AAAI 2027 main-track submission draft:

**HARP-Select: Confidence-Calibrated Residual Polynomial Experts for Heterophilous Graphs**

The central idea is to treat heterophilous graph learning as a recorded choice
between two residual polynomial experts:

- **HARP-GNN**, a self-loop residual polynomial expert.
- **HARP-ESep**, an ego-separated no-self residual polynomial expert.
- **HARP-Select**, a validation-calibrated selector that switches to HARP-ESep
  only when its validation advantage exceeds a fixed uncertainty threshold.

The selector uses validation labels only. Test labels never enter the routing
decision.

## Artifact Status

This artifact includes:

- PyTorch implementation without PyG dependency.
- Loaders for Planetoid, WebKB/Geom-GCN, and critical-heterophily benchmarks.
- Implemented baselines: MLP, GCN, SGC, APPNP, MixHop, GPR-GNN, H2GCN-style
  separation, LINKX, HARP-GNN, HARP-ESep, and HARP-Select.
- Fixed-split experiment configs and CSV outputs for all reported tables.
- LaTeX table/figure generation scripts.
- AAAI-format anonymous manuscript source.
- Reproducibility checklist source/PDF.
- Verification scripts for implementation invariants, result coverage,
  reported numbers, manuscript integrity, claim boundaries, and package
  readiness.

Raw public datasets are not bundled. The loaders download or read public
benchmark files under `data/` when experiments are rerun.

## Main Reported Evidence

The current manuscript intentionally makes bounded rather than leaderboard
claims.

- On Roman-Empire, HARP-Select routes all splits to HARP-ESep and exceeds the
  strongest implemented non-HARP baseline, H2GCN, by `+0.79` accuracy points
  with 10/10 split wins and Holm-corrected exact sign-flip `p=0.008`.
- On Amazon-Ratings, LINKX is stronger than HARP-Select by `0.91` points,
  exposing that the current expert library lacks a LINKX-like adjacency expert.
- On Minesweeper under ROC-AUC, HARP-ESep improves over HARP-GNN by `+1.45`
  points with 10/10 split wins.
- On Tolokers under ROC-AUC, the direction reverses: HARP-ESep is `-3.56`
  points below HARP-GNN with 0/10 split wins.
- On the original six heterophily datasets, HARP-Select removes the large
  significant Chameleon/Squirrel deficits of the self-loop expert but does not
  make an unsupported universal superiority claim.

These mixed results are deliberate: the contribution is an auditable
specialist-and-routing formulation, not a claim that one filter wins everywhere.

## Quick Start

Create an environment and install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Run fast implementation checks:

```powershell
python scripts\verify_implementation.py
```

Verify all reported numbers against the included CSV files:

```powershell
python scripts\verify_reported_results.py
```

Run the full artifact rebuild and verification pipeline from the included CSVs:

```powershell
.\scripts\rebuild_reported_artifacts.ps1 -SkipCompile
```

Run final readiness checks after the PDF and packages are built:

```powershell
python scripts\verify_submission_readiness.py
```

## Rebuild Tables From CSVs

```powershell
python scripts\summarize_results.py --input results\planetoid_all.csv --output paper\tables\planetoid_results.tex
python scripts\summarize_results.py --input results\webkb.csv --output paper\tables\webkb_results.tex
python scripts\summarize_results.py --input results\geom_gcn_large.csv --output paper\tables\geom_gcn_large_results.tex
python scripts\build_harp_selector.py
python scripts\build_critical_baseline_tables.py
python scripts\build_binary_critical_tables.py
```

## Rerun Core Experiments

The full CPU run is intentionally reproducible but not tiny. Use `--resume` to
skip rows already present in the configured CSV output.

```powershell
python -m src.harp_gnn.run_experiment --config configs\webkb.yaml --resume
python -m src.harp_gnn.run_experiment --config configs\geom_gcn_large.yaml --resume
python -m src.harp_gnn.run_experiment --config configs\critical_heterophily_harp.yaml --resume
python -m src.harp_gnn.run_experiment --config configs\critical_heterophily_baselines.yaml --resume
python -m src.harp_gnn.run_experiment --config configs\critical_heterophily_binary_harp.yaml --resume
```

Then regenerate reported artifacts:

```powershell
.\scripts\rebuild_reported_artifacts.ps1 -SkipCompile
```

## Compile the Paper

If a bundled TinyTeX installation is present under `tools/`, the compile script
uses it. Otherwise it falls back to `pdflatex` and `bibtex` on `PATH`.

```powershell
.\scripts\compile_paper.ps1
```

The official compiled PDF is written to:

```text
paper/HARP_GNN_AAAI2027_official_compile.pdf
```

## Repository Layout

```text
configs/      Experiment configs.
paper/        Anonymous AAAI manuscript source, figures, tables, and checks.
results/      CSV outputs and generated statistical summaries.
scripts/      Experiment, table-generation, packaging, and verification tools.
src/harp_gnn/ Dataset loaders, model definitions, training loop, and runner.
```

## Anonymous GitHub Use

For double-blind review, submit the frozen Code and Data ZIP through the official
submission system. Do not cite a mutable external GitHub URL in the paper unless
the conference instructions explicitly permit it. See
`ANONYMITY_AND_RELEASE.md` for anonymous GitHub mirroring guidance.

## License

The code in this artifact is released under the MIT License. See `LICENSE`.
