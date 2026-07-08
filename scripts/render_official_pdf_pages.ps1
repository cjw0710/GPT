param(
    [string]$PdfPath,
    [string]$OutputPattern,
    [int]$Resolution = 150
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$PaperDir = Join-Path $Root "paper"

if (-not $PdfPath) {
    $PdfPath = Join-Path $PaperDir "HARP_GNN_AAAI2027_official_compile.pdf"
}
if (-not $OutputPattern) {
    $OutputPattern = Join-Path $PaperDir "HARP_GNN_AAAI2027_official_page%d.png"
}

$PdfPath = (Resolve-Path -LiteralPath $PdfPath).Path
$OutputDir = Split-Path -Parent $OutputPattern
$ResolvedOutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
$ResolvedPaperDir = (Resolve-Path -LiteralPath $PaperDir).Path

if ($ResolvedOutputDir -ne $ResolvedPaperDir) {
    throw "Refusing to write page previews outside the paper directory: $ResolvedOutputDir"
}

$GhostscriptBase = Join-Path $Root "tools\TinyTeX\portable\TinyTeX\tlpkg\tlgs"
$Ghostscript = Join-Path $GhostscriptBase "bin\gswin64c.exe"
if (-not (Test-Path -LiteralPath $Ghostscript)) {
    throw "Ghostscript executable was not found at $Ghostscript"
}

$SearchPath = @(
    (Join-Path $GhostscriptBase "Resource\Init"),
    (Join-Path $GhostscriptBase "Resource"),
    (Join-Path $GhostscriptBase "lib"),
    (Join-Path $GhostscriptBase "kanji")
) -join ";"

Get-ChildItem -LiteralPath $ResolvedPaperDir -Filter "HARP_GNN_AAAI2027_official_page*.png" -File |
    Remove-Item -Force

& $Ghostscript "-I$SearchPath" -q -dSAFER -dBATCH -dNOPAUSE -sDEVICE=png16m "-r$Resolution" "-sOutputFile=$OutputPattern" $PdfPath

$Generated = Get-ChildItem -LiteralPath $ResolvedPaperDir -Filter "HARP_GNN_AAAI2027_official_page*.png" -File |
    Sort-Object Name

if (-not $Generated) {
    throw "Ghostscript finished but no page previews were generated."
}

Write-Host "Rendered official AAAI PDF page previews:" $Generated.Count
