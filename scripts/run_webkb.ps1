$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $Root
python -m src.harp_gnn.run_experiment --config configs\webkb.yaml
python scripts\summarize_results.py --input results\webkb.csv --output paper\tables\webkb_results.tex
