$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $Root
python -m src.harp_gnn.run_experiment --config configs\synthetic_sweep.yaml
python scripts\summarize_results.py --input results\synthetic_sweep.csv --output paper\tables\synthetic_results.tex
