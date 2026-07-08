param(
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PaperDir = Join-Path $Root "paper"
$tinyTexBin = Join-Path $Root "tools\TinyTeX\portable\TinyTeX\bin\windows"
$pdflatex = Join-Path $tinyTexBin "pdflatex.exe"

if (-not (Test-Path -LiteralPath $pdflatex)) {
    throw "pdflatex.exe was not found at $pdflatex"
}

Push-Location $Root
try {
    python scripts\fill_reproducibility_checklist.py
    if ($LASTEXITCODE -ne 0) {
        throw "fill_reproducibility_checklist.py failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}

if ($SkipCompile) {
    return
}

$env:PATH = "$tinyTexBin;$env:PATH"
Push-Location $PaperDir
try {
    & $pdflatex -interaction=nonstopmode -halt-on-error ReproducibilityChecklist_HARP_GNN.tex
    if ($LASTEXITCODE -ne 0) {
        throw "pdflatex failed with exit code $LASTEXITCODE"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $PaperDir "ReproducibilityChecklist_HARP_GNN.pdf"))) {
        throw "Reproducibility checklist PDF was not produced."
    }
    Write-Host "Compiled reproducibility checklist PDF:" (Join-Path $PaperDir "ReproducibilityChecklist_HARP_GNN.pdf")
}
finally {
    Pop-Location
}
