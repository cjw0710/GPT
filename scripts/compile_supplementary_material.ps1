param(
    [string]$PaperDir = "",
    [string]$TinyTexBin = ""
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
if ([string]::IsNullOrWhiteSpace($PaperDir)) {
    $PaperDir = Join-Path $Root "paper"
}
$BuildDir = Join-Path $Root "build\supplementary_material_compile"
$SourceDir = Join-Path $BuildDir "source"

if ([string]::IsNullOrWhiteSpace($TinyTexBin)) {
    $BundledTinyTexBin = Join-Path $Root "tools\TinyTeX\portable\TinyTeX\bin\windows"
    if (Test-Path -LiteralPath (Join-Path $BundledTinyTexBin "pdflatex.exe")) {
        $TinyTexBin = $BundledTinyTexBin
    }
}

if (-not [string]::IsNullOrWhiteSpace($TinyTexBin)) {
    $pdflatex = Join-Path $TinyTexBin "pdflatex.exe"
    if (-not (Test-Path -LiteralPath $pdflatex)) {
        throw "pdflatex.exe was not found at $pdflatex"
    }
    $env:PATH = "$TinyTexBin;$env:PATH"
}
else {
    $pdflatexCommand = Get-Command pdflatex -ErrorAction SilentlyContinue
    if ($null -eq $pdflatexCommand) {
        throw "No bundled TinyTeX or system pdflatex found. Install TeX Live/TinyTeX or pass -TinyTexBin."
    }
    $pdflatex = $pdflatexCommand.Source
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

Copy-Item -LiteralPath (Join-Path $PaperDir "supplementary_material.tex") -Destination (Join-Path $SourceDir "supplementary_material.tex") -Force
Copy-Item -LiteralPath (Join-Path $PaperDir "tables") -Destination (Join-Path $SourceDir "tables") -Recurse -Force
Copy-Item -LiteralPath (Join-Path $PaperDir "figures") -Destination (Join-Path $SourceDir "figures") -Recurse -Force

Push-Location $SourceDir
try {
    & $pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
    if ($LASTEXITCODE -ne 0) {
        throw "pdflatex pass 1 failed with exit code $LASTEXITCODE"
    }
    & $pdflatex -interaction=nonstopmode -halt-on-error supplementary_material.tex
    if ($LASTEXITCODE -ne 0) {
        throw "pdflatex pass 2 failed with exit code $LASTEXITCODE"
    }
    Copy-Item -LiteralPath (Join-Path $SourceDir "supplementary_material.pdf") -Destination (Join-Path $PaperDir "HARP_GNN_AAAI2027_supplementary_material.pdf") -Force
    Copy-Item -LiteralPath (Join-Path $SourceDir "supplementary_material.log") -Destination (Join-Path $PaperDir "supplementary_material.log") -Force
    Write-Host "Compiled supplementary material PDF:" (Join-Path $PaperDir "HARP_GNN_AAAI2027_supplementary_material.pdf")
}
finally {
    Pop-Location
}
