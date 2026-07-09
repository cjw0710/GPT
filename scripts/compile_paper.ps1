param(
    [string]$PaperDir = "",
    [string]$TinyTexBin = "",
    [switch]$SkipPreview
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($PaperDir)) {
    $PaperDir = Join-Path $Root "paper"
}
$BuildDir = Join-Path $Root "build\official_compile"
$SourceDir = Join-Path $BuildDir "source"

if ([string]::IsNullOrWhiteSpace($TinyTexBin)) {
    $BundledTinyTexBin = Join-Path $Root "tools\TinyTeX\portable\TinyTeX\bin\windows"
    if (Test-Path -LiteralPath (Join-Path $BundledTinyTexBin "pdflatex.exe")) {
        $TinyTexBin = $BundledTinyTexBin
    }
}

if (-not [string]::IsNullOrWhiteSpace($TinyTexBin)) {
    $pdflatex = Join-Path $TinyTexBin "pdflatex.exe"
    $bibtex = Join-Path $TinyTexBin "bibtex.exe"
    if (-not (Test-Path -LiteralPath $pdflatex)) {
        throw "pdflatex.exe was not found at $pdflatex"
    }
    if (-not (Test-Path -LiteralPath $bibtex)) {
        throw "bibtex.exe was not found at $bibtex"
    }
    $env:PATH = "$TinyTexBin;$env:PATH"
}
else {
    $pdflatexCommand = Get-Command pdflatex -ErrorAction SilentlyContinue
    $bibtexCommand = Get-Command bibtex -ErrorAction SilentlyContinue
    if ($null -eq $pdflatexCommand -or $null -eq $bibtexCommand) {
        throw "No bundled TinyTeX or system pdflatex/bibtex found. Install TeX Live/TinyTeX or pass -TinyTexBin."
    }
    $pdflatex = $pdflatexCommand.Source
    $bibtex = $bibtexCommand.Source
}

$ResolvedBuildDir = if (Test-Path -LiteralPath $BuildDir) { (Resolve-Path -LiteralPath $BuildDir).Path } else { $BuildDir }
$ResolvedRoot = (Resolve-Path -LiteralPath $Root).Path
if ($ResolvedBuildDir -notlike (Join-Path $ResolvedRoot "build") + "*") {
    throw "Refusing to clean unexpected build directory: $ResolvedBuildDir"
}
if (Test-Path -LiteralPath $BuildDir) {
    Remove-Item -LiteralPath $BuildDir -Recurse -Force
}
New-Item -ItemType Directory -Path $SourceDir -Force | Out-Null

foreach ($File in @("main.tex", "aaai2027.sty", "aaai2027.bst")) {
    Copy-Item -LiteralPath (Join-Path $PaperDir $File) -Destination (Join-Path $SourceDir $File) -Force
}
Get-ChildItem -LiteralPath $PaperDir -Filter "*.bib" -File |
    Copy-Item -Destination $SourceDir -Force
Copy-Item -LiteralPath (Join-Path $PaperDir "tables") -Destination (Join-Path $SourceDir "tables") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $PaperDir "figures") -Destination (Join-Path $SourceDir "figures") -Recurse -Force

function Invoke-LaTeXStep {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    & $Command
    if ($LASTEXITCODE -ne 0) {
        throw "$Name failed with exit code $LASTEXITCODE"
    }
}

Push-Location $SourceDir
try {
    Invoke-LaTeXStep "pdflatex pass 1" {
        & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    }
    Invoke-LaTeXStep "bibtex" {
        & $bibtex main
    }
    Invoke-LaTeXStep "pdflatex pass 2" {
        & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    }
    Invoke-LaTeXStep "pdflatex pass 3" {
        & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    }
    Invoke-LaTeXStep "pdflatex pass 4" {
        & $pdflatex -interaction=nonstopmode -halt-on-error main.tex
    }

    $CompiledPdf = Join-Path $SourceDir "main.pdf"
    Copy-Item -LiteralPath $CompiledPdf -Destination (Join-Path $PaperDir "HARP_GNN_AAAI2027_official_compile.pdf") -Force
    Copy-Item -LiteralPath $CompiledPdf -Destination (Join-Path $PaperDir "main.pdf") -Force
    Copy-Item -LiteralPath $CompiledPdf -Destination (Join-Path $PaperDir "main_latest.pdf") -Force
    Copy-Item -LiteralPath (Join-Path $SourceDir "main.log") -Destination (Join-Path $PaperDir "main.log") -Force
    Copy-Item -LiteralPath (Join-Path $SourceDir "main.aux") -Destination (Join-Path $PaperDir "main.aux") -Force
    Copy-Item -LiteralPath (Join-Path $SourceDir "main.bbl") -Destination (Join-Path $PaperDir "main.bbl") -Force
    Write-Host "Compiled official AAAI PDF:" (Join-Path $PaperDir "HARP_GNN_AAAI2027_official_compile.pdf")

    if (-not $SkipPreview) {
        & (Join-Path $PSScriptRoot "render_official_pdf_pages.ps1") `
            -PdfPath (Join-Path $PaperDir "HARP_GNN_AAAI2027_official_compile.pdf")
    }
}
finally {
    Pop-Location
}
