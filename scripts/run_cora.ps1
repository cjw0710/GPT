$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location -LiteralPath $Root
python -m src.harp_gnn.run_experiment --config configs\cora.yaml
python scripts\summarize_results.py --input results\cora.csv --output paper\tables\cora_results.tex
