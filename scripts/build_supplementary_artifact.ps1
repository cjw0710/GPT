param(
    [string]$OutputPath,
    [switch]$SkipValidation
)

$ErrorActionPreference = "Stop"

$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$PaperDir = Join-Path $Root "paper"
$BuildRoot = Join-Path $Root "build"
$WorkDir = Join-Path $BuildRoot "supplementary_artifact"
$StageDir = Join-Path $WorkDir "package"
$ArtifactRootName = "HARP_GNN_AAAI2027_reproducibility_artifact"
$ArtifactRoot = Join-Path $StageDir $ArtifactRootName

if (-not $OutputPath) {
    $OutputPath = Join-Path $PaperDir "HARP_GNN_AAAI2027_supplementary_artifact.zip"
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

function Copy-FileIntoArtifact {
    param(
        [string]$RelativePath
    )
    $SourcePath = Join-Path $Root $RelativePath
    if (-not (Test-Path -LiteralPath $SourcePath)) {
        throw "Missing artifact input: $RelativePath"
    }
    $DestinationPath = Join-Path $ArtifactRoot $RelativePath
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $DestinationPath) | Out-Null
    Copy-Item -LiteralPath $SourcePath -Destination $DestinationPath -Force
}

function Copy-TreeFilesIntoArtifact {
    param(
        [string]$RelativeDir,
        [string[]]$Include
    )
    $SourceDir = Join-Path $Root $RelativeDir
    if (-not (Test-Path -LiteralPath $SourceDir)) {
        throw "Missing artifact directory: $RelativeDir"
    }
    $AllowedExtensions = $Include | ForEach-Object { $_ -replace '^\*', '' }
    Get-ChildItem -LiteralPath $SourceDir -Recurse -File |
        Where-Object {
            $_.FullName -notlike "*\__pycache__\*" -and
            ($AllowedExtensions -contains $_.Extension)
        } |
        ForEach-Object {
            $RootPrefix = $Root.TrimEnd('\') + '\'
            $FullName = [System.IO.Path]::GetFullPath($_.FullName)
            if (-not $FullName.StartsWith($RootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing to copy file outside repository root: $FullName"
            }
            $RelPath = $FullName.Substring($RootPrefix.Length)
            Copy-FileIntoArtifact -RelativePath $RelPath
        }
}

New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
Assert-UnderPath -Path $WorkDir -Parent $BuildRoot
if (Test-Path -LiteralPath $WorkDir) {
    $ResolvedWorkDir = (Resolve-Path -LiteralPath $WorkDir).Path
    Assert-UnderPath -Path $ResolvedWorkDir -Parent $BuildRoot
    Remove-Item -LiteralPath $ResolvedWorkDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $ArtifactRoot | Out-Null

Copy-FileIntoArtifact -RelativePath "README.md"
Copy-FileIntoArtifact -RelativePath ".gitignore"
Copy-FileIntoArtifact -RelativePath "LICENSE"
Copy-FileIntoArtifact -RelativePath "ANONYMITY_AND_RELEASE.md"
Copy-FileIntoArtifact -RelativePath "requirements.txt"
Copy-TreeFilesIntoArtifact -RelativeDir "src" -Include @("*.py")
Copy-TreeFilesIntoArtifact -RelativeDir "scripts" -Include @("*.py", "*.ps1")
Copy-TreeFilesIntoArtifact -RelativeDir "configs" -Include @("*.yaml")
Copy-TreeFilesIntoArtifact -RelativeDir "results" -Include @("*.csv")
Copy-TreeFilesIntoArtifact -RelativeDir "paper\tables" -Include @("*.tex")
Copy-TreeFilesIntoArtifact -RelativeDir "paper\figures" -Include @("*.png")

foreach ($PaperFile in @(
    "paper\main.tex",
    "paper\references.bib",
    "paper\references_additions.bib",
    "paper\aaai2027.sty",
    "paper\aaai2027.bst",
    "paper\ARTIFACT_MANIFEST.md",
    "paper\AAAI27_SUBMISSION_REQUIREMENTS.md",
    "paper\SCIENTIFIC_AUDIT.md",
    "paper\HARP_X_DIAGNOSTIC.md",
    "paper\HARP_SELECT_DIAGNOSTIC.md",
    "paper\HARP_SELECTOR_SENSITIVITY.md",
    "paper\HARP_SELECTOR_COST.md",
    "paper\HARP_GNN_AAAI2027_official_compile.pdf",
    "paper\ReproducibilityChecklist.tex",
    "paper\ReproducibilityChecklist_HARP_GNN.tex",
    "paper\ReproducibilityChecklist_HARP_GNN.pdf"
)) {
    Copy-FileIntoArtifact -RelativePath $PaperFile
}

$Readme = @"
# HARP-GNN AAAI 2027 Reproducibility Artifact

This supplementary artifact contains the code, configs, generated result CSVs,
generated manuscript tables/figures, and verification scripts for the current
HARP-GNN AAAI 2027 draft.

## Quick Checks

```powershell
python scripts\verify_implementation.py
python scripts\verify_manuscript_integrity.py
python scripts\verify_reported_results.py
python scripts\generate_scientific_audit.py
python scripts\verify_top_conference_claims.py
```

## Rebuild Reported Tables From Included CSVs

```powershell
python scripts\summarize_results.py --input results\synthetic_sweep.csv --output paper\tables\synthetic_results.tex
python scripts\summarize_results.py --input results\planetoid_all.csv --output paper\tables\planetoid_results.tex
python scripts\summarize_results.py --input results\webkb.csv --output paper\tables\webkb_results.tex
python scripts\summarize_results.py --input results\geom_gcn_large.csv --output paper\tables\geom_gcn_large_results.tex
python scripts\build_binary_critical_tables.py
```

Raw datasets are not bundled. Dataset loaders download or read public benchmark
files under `data/` when experiments are run from the full repository.
"@
$Readme | Set-Content -LiteralPath (Join-Path $ArtifactRoot "SUPPLEMENTARY_ARTIFACT_README.md") -Encoding UTF8

if (-not $SkipValidation) {
    Push-Location $ArtifactRoot
    try {
        python scripts\verify_implementation.py
        if ($LASTEXITCODE -ne 0) {
            throw "verify_implementation.py failed with exit code $LASTEXITCODE"
        }
        python scripts\verify_manuscript_integrity.py
        if ($LASTEXITCODE -ne 0) {
            throw "verify_manuscript_integrity.py failed with exit code $LASTEXITCODE"
        }
        python scripts\verify_reported_results.py
        if ($LASTEXITCODE -ne 0) {
            throw "verify_reported_results.py failed with exit code $LASTEXITCODE"
        }
        python scripts\verify_anonymous_artifact.py --root .
        if ($LASTEXITCODE -ne 0) {
            throw "verify_anonymous_artifact.py failed with exit code $LASTEXITCODE"
        }
        python scripts\generate_scientific_audit.py
        if ($LASTEXITCODE -ne 0) {
            throw "generate_scientific_audit.py failed with exit code $LASTEXITCODE"
        }
        python scripts\verify_top_conference_claims.py
        if ($LASTEXITCODE -ne 0) {
            throw "verify_top_conference_claims.py failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

Get-ChildItem -LiteralPath $ArtifactRoot -Recurse -Directory -Filter "__pycache__" |
    Remove-Item -Recurse -Force
Get-ChildItem -LiteralPath $ArtifactRoot -Recurse -File -Filter "*.pyc" |
    Remove-Item -Force

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
Write-Host "Built supplementary reproducibility artifact:" $Package.FullName
Write-Host "Package size:" $Package.Length "bytes"
