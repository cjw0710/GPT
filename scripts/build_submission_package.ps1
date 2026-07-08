param(
    [string]$OutputPath,
    [switch]$SkipCompile,
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PaperDir = Join-Path $Root "paper"
$BuildRoot = Join-Path $Root "build"
$WorkDir = Join-Path $BuildRoot "submission_package"
$StageDir = Join-Path $WorkDir "package"
$SourceDir = Join-Path $StageDir "source"
$ValidationDir = Join-Path $WorkDir "validation_source"

if (-not $OutputPath) {
    $OutputPath = Join-Path $PaperDir "HARP_GNN_AAAI2027_submission_package.zip"
}

function Assert-UnderPath {
    param(
        [string]$Path,
        [string]$Parent
    )
    $FullParent = [System.IO.Path]::GetFullPath($Parent).TrimEnd('\')
    $FullPath = [System.IO.Path]::GetFullPath($Path).TrimEnd('\')
    if ($FullPath -ne $FullParent -and -not $FullPath.StartsWith("$FullParent\", [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to operate outside intended directory. Path=$FullPath Parent=$FullParent"
    }
}

function Copy-RequiredFile {
    param(
        [string]$RelativePath,
        [string]$DestinationRoot
    )
    $SourcePath = Join-Path $PaperDir $RelativePath
    if (-not (Test-Path -LiteralPath $SourcePath)) {
        throw "Missing paper dependency: $RelativePath"
    }
    $DestinationPath = Join-Path $DestinationRoot $RelativePath
    $DestinationParent = Split-Path -Parent $DestinationPath
    New-Item -ItemType Directory -Force -Path $DestinationParent | Out-Null
    Copy-Item -LiteralPath $SourcePath -Destination $DestinationPath -Force
}

if (-not $SkipCompile) {
    & (Join-Path $PSScriptRoot "compile_paper.ps1")
}

$PdfPath = Join-Path $PaperDir "HARP_GNN_AAAI2027_official_compile.pdf"
if (-not (Test-Path -LiteralPath $PdfPath)) {
    throw "Missing compiled official PDF: $PdfPath"
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
Assert-UnderPath -Path $WorkDir -Parent $BuildRoot
if (Test-Path -LiteralPath $WorkDir) {
    $ResolvedWorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
    Assert-UnderPath -Path $ResolvedWorkDir -Parent $BuildRoot
    Remove-Item -LiteralPath $ResolvedWorkDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $SourceDir | Out-Null

$MainText = Get-Content -LiteralPath (Join-Path $PaperDir "main.tex") -Raw -Encoding UTF8
$Inputs = [regex]::Matches($MainText, "\\input\{([^}]+)\}") | ForEach-Object { $_.Groups[1].Value }
$Figures = [regex]::Matches($MainText, "\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}") | ForEach-Object { $_.Groups[1].Value }
$BibliographyFiles = [regex]::Matches($MainText, "\\bibliography\{([^}]+)\}") |
    ForEach-Object { $_.Groups[1].Value.Split(",") } |
    ForEach-Object { $_.Trim() } |
    Where-Object { $_ } |
    ForEach-Object { if ($_.EndsWith(".bib")) { $_ } else { "$_.bib" } }
$CoreFiles = @("main.tex", "aaai2027.sty", "aaai2027.bst") + $BibliographyFiles

foreach ($File in $CoreFiles) {
    Copy-RequiredFile -RelativePath $File -DestinationRoot $SourceDir
}
foreach ($File in ($Inputs + $Figures | Sort-Object -Unique)) {
    Copy-RequiredFile -RelativePath $File -DestinationRoot $SourceDir
}

$PackagePdf = Join-Path $StageDir "HARP_GNN_AAAI2027_official_compile.pdf"
Copy-Item -LiteralPath $PdfPath -Destination $PackagePdf -Force

if (-not $SkipValidation) {
    Copy-Item -LiteralPath $SourceDir -Destination $ValidationDir -Recurse -Force
    & (Join-Path $PSScriptRoot "compile_paper.ps1") -PaperDir $ValidationDir -SkipPreview

    $ValidationPdf = Join-Path $ValidationDir "HARP_GNN_AAAI2027_official_compile.pdf"
    if (-not (Test-Path -LiteralPath $ValidationPdf)) {
        throw "Validation compile did not produce a PDF."
    }
}

$GeneratedAt = Get-Date -Format "yyyy-MM-dd HH:mm:ss zzz"
$Manifest = @"
HARP-GNN AAAI 2027 submission package
Generated: $GeneratedAt

Contents:
- HARP_GNN_AAAI2027_official_compile.pdf
- source/main.tex
- source/references*.bib
- source/aaai2027.sty
- source/aaai2027.bst
- source/tables/*.tex used by main.tex
- source/figures/*.png used by main.tex

Validation:
- Source compile validation: $(-not $SkipValidation)
- Current manuscript verifier: run `python scripts\verify_manuscript_integrity.py` from the project root.
"@
$ManifestPath = Join-Path $StageDir "PACKAGE_MANIFEST.txt"
$Manifest | Set-Content -LiteralPath $ManifestPath -Encoding UTF8

$OutputParent = Split-Path -Parent $OutputPath
if (-not $OutputParent) {
    $OutputParent = (Get-Location).Path
    $OutputPath = Join-Path $OutputParent $OutputPath
}
New-Item -ItemType Directory -Force -Path $OutputParent | Out-Null
$OutputPath = [System.IO.Path]::GetFullPath($OutputPath)
Assert-UnderPath -Path $OutputPath -Parent $PaperDir
if (Test-Path -LiteralPath $OutputPath) {
    Remove-Item -LiteralPath $OutputPath -Force
}

Compress-Archive -Path (Join-Path $StageDir "*") -DestinationPath $OutputPath -Force

$Package = Get-Item -LiteralPath $OutputPath
Write-Host "Built AAAI submission package:" $Package.FullName
Write-Host "Package size:" $Package.Length "bytes"
