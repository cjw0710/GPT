# HARP-GNN AAAI 2027 Reproducibility Artifact

This supplementary artifact contains the code, configs, generated result CSVs,
generated manuscript tables/figures, and verification scripts for the current
HARP-GNN AAAI 2027 draft.

The package also includes paper\HARP_GNN_AAAI2027_supplementary_material.pdf,
which gives extended tables, diagnostics, and reproducibility details that do
not fit in the main seven-page manuscript.

## Quick Checks

`powershell
python scripts\verify_implementation.py
python scripts\verify_manuscript_integrity.py
python scripts\verify_reported_results.py
python scripts\generate_scientific_audit.py
python scripts\verify_top_conference_claims.py
`

## Rebuild Reported Tables From Included CSVs

`powershell
python scripts\summarize_results.py --input results\synthetic_sweep.csv --output paper\tables\synthetic_results.tex
python scripts\summarize_results.py --input results\planetoid_all.csv --output paper\tables\planetoid_results.tex
python scripts\summarize_results.py --input results\webkb.csv --output paper\tables\webkb_results.tex
python scripts\summarize_results.py --input results\geom_gcn_large.csv --output paper\tables\geom_gcn_large_results.tex
python scripts\build_binary_critical_tables.py
.\scripts\compile_supplementary_material.ps1
`

Raw datasets are not bundled. Dataset loaders download or read public benchmark
files under data/ when experiments are run from the full repository.
