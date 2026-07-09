param(
    [switch]$SkipCompile
)

$ErrorActionPreference = "Stop"

$Root = Resolve-Path (Join-Path $PSScriptRoot "..")

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command
    )

    Write-Host ""
    Write-Host "[step] $Name"
    & $Command
}

Push-Location $Root
try {
    Invoke-Step "Synthetic results table" {
        python scripts\summarize_results.py --input results\synthetic_sweep.csv --output paper\tables\synthetic_results.tex
    }
    Invoke-Step "Synthetic accuracy figure" {
        python scripts\plot_results.py --input results\synthetic_sweep.csv --output paper\figures\synthetic_accuracy.png
    }
    Invoke-Step "HARP-GNN framework figure" {
        python scripts\plot_framework.py --output paper\figures\harp_framework.png
    }
    Invoke-Step "HARP-Select framework figure" {
        python scripts\plot_harp_select_framework.py --output paper\figures\harp_select_framework.png
    }
    Invoke-Step "HARP-Select split-level audit figure" {
        python scripts\plot_selector_audit.py --input results\harp_select_diagnostics.csv --output paper\figures\selector_audit.png
    }
    Invoke-Step "HARP-Select threshold sensitivity figure" {
        python scripts\plot_selector_sensitivity.py --input results\harp_select_threshold_sensitivity_overall.csv --output paper\figures\selector_sensitivity.png
    }
    Invoke-Step "Planetoid results table" {
        python scripts\summarize_results.py --input results\planetoid_all.csv --output paper\tables\planetoid_results.tex
    }
    Invoke-Step "WebKB dataset statistics" {
        python scripts\summarize_datasets.py --config configs\webkb.yaml --output paper\tables\webkb_dataset_stats.tex --csv-output results\webkb_dataset_stats.csv
    }
    Invoke-Step "WebKB results table" {
        python scripts\summarize_results.py --input results\webkb.csv --output paper\tables\webkb_results.tex
    }
    Invoke-Step "WebKB paired tests" {
        python scripts\paired_tests.py --input results\webkb.csv --output paper\tables\webkb_paired_tests.tex --csv-output results\webkb_paired_tests.csv
    }
    Invoke-Step "WebKB ablation table" {
        python scripts\summarize_results.py --input results\webkb_ablation.csv --output paper\tables\webkb_ablation_results.tex
    }
    Invoke-Step "WebKB scalar-gate table" {
        python scripts\summarize_results.py --input results\webkb_scalar_gate.csv --output paper\tables\webkb_scalar_gate_results.tex
    }
    Invoke-Step "WebKB filter weights" {
        python scripts\summarize_filter_weights.py --input results\webkb.csv --output paper\tables\webkb_filter_weights.tex --csv-output results\webkb_filter_weights.csv
    }
    Invoke-Step "WebKB gate diagnostics" {
        python scripts\summarize_gate_diagnostics.py --input results\webkb.csv --output paper\tables\webkb_gate_diagnostics.tex --csv-output results\webkb_gate_diagnostics.csv
    }
    Invoke-Step "WebKB parameter counts" {
        python scripts\summarize_parameters.py --config configs\webkb.yaml --output paper\tables\webkb_parameters.tex --csv-output results\webkb_parameters.csv --raw-output results\webkb_parameters_raw.csv
    }
    Invoke-Step "WebKB runtime table" {
        python scripts\summarize_runtime.py --input results\webkb.csv --output paper\tables\webkb_runtime.tex --csv-output results\webkb_runtime.csv
    }
    Invoke-Step "Larger Geom-GCN dataset statistics" {
        python scripts\summarize_datasets.py --config configs\geom_gcn_large.yaml --output paper\tables\geom_gcn_large_dataset_stats.tex --csv-output results\geom_gcn_large_dataset_stats.csv
    }
    Invoke-Step "Larger Geom-GCN results table" {
        python scripts\summarize_results.py --input results\geom_gcn_large.csv --output paper\tables\geom_gcn_large_results.tex
    }
    Invoke-Step "Larger Geom-GCN paired tests" {
        python scripts\paired_tests.py --input results\geom_gcn_large.csv --output paper\tables\geom_gcn_large_paired_tests.tex --csv-output results\geom_gcn_large_paired_tests.csv
    }
    Invoke-Step "HARP-ESep full Geom-GCN table" {
        python scripts\summarize_results.py --input results\geom_gcn_harp_esep.csv --output paper\tables\geom_gcn_harp_esep_results.tex
    }
    Invoke-Step "Merged Geom-GCN HARP-ESep results" {
        python scripts\merge_harp_esep_results.py --base results\geom_gcn_large.csv --esep results\geom_gcn_harp_esep.csv --output results\geom_gcn_large_with_harp_esep.csv
    }
    Invoke-Step "HARP-ESep paired tests" {
        python scripts\paired_tests.py --input results\geom_gcn_large_with_harp_esep.csv --target harp_esep --output paper\tables\geom_gcn_harp_esep_paired_tests.tex --csv-output results\geom_gcn_harp_esep_paired_tests.csv
    }
    Invoke-Step "Critical heterophily HARP/ESep table" {
        python scripts\summarize_results.py --input results\critical_heterophily_harp.csv --output paper\tables\critical_heterophily_harp_results.tex
    }
    Invoke-Step "Critical heterophily HARP/ESep paired tests" {
        python scripts\paired_tests.py --input results\critical_heterophily_harp.csv --target harp_esep --baselines harp --output paper\tables\critical_heterophily_harp_esep_paired_tests.tex --csv-output results\critical_heterophily_harp_esep_paired_tests.csv
    }
    Invoke-Step "Critical heterophily robust paired statistics" {
        python scripts\paired_robustness.py --input results\critical_heterophily_harp.csv --assignments results\critical_heterophily_harp_esep_paired_tests.csv --output paper\tables\critical_heterophily_harp_esep_robust_tests.tex --csv-output results\critical_heterophily_harp_esep_robust_tests.csv
    }
    Invoke-Step "Binary critical heterophily complete-only ROC-AUC tables" {
        python scripts\build_binary_critical_tables.py
    }
    Invoke-Step "HARP-Select calibrated routing" {
        python scripts\build_harp_selector.py
    }
    Invoke-Step "External critical-heterophily non-HARP baselines" {
        python scripts\build_critical_baseline_tables.py
    }
    Invoke-Step "HARP-Select threshold sensitivity" {
        python scripts\analyze_selector_sensitivity.py
    }
    Invoke-Step "HARP-Select training cost" {
        python scripts\analyze_selector_cost.py
    }
    Invoke-Step "HARP-Select paired tests" {
        python scripts\paired_tests.py --input results\harp_select_with_baselines.csv --target harp_select --baselines mlp gcn sgc appnp mixhop gprgnn h2gcn linkx --output paper\tables\harp_select_paired_tests.tex --csv-output results\harp_select_paired_tests.csv
    }
    Invoke-Step "HARP-Select robust paired statistics" {
        python scripts\paired_robustness.py --input results\harp_select_with_baselines.csv --assignments results\harp_select_paired_tests.csv --output paper\tables\harp_select_robust_tests.tex --csv-output results\harp_select_robust_tests.csv
    }
    Invoke-Step "HARP-X diagnostic table" {
        python scripts\summarize_results.py --input results\harp_x_diagnostic.csv --output paper\tables\harp_x_diagnostic_results.tex
    }
    Invoke-Step "Result coverage audit" {
        python scripts\audit_results.py --config-dir configs --output-csv results\result_audit.csv --output-tex paper\tables\result_audit.tex
    }
    Invoke-Step "Implementation verifier" {
        python scripts\verify_implementation.py
    }
    Invoke-Step "Manuscript integrity verifier" {
        python scripts\verify_manuscript_integrity.py
    }
    Invoke-Step "Reported-result verifier" {
        python scripts\verify_reported_results.py
    }
    Invoke-Step "Scientific audit report" {
        python scripts\generate_scientific_audit.py
    }
    Invoke-Step "Top-conference claim verifier" {
        python scripts\verify_top_conference_claims.py
    }
    Invoke-Step "Reproducibility checklist" {
        if ($SkipCompile) {
            .\scripts\build_reproducibility_checklist.ps1 -SkipCompile
        }
        else {
            .\scripts\build_reproducibility_checklist.ps1
        }
    }

    if (-not $SkipCompile) {
        Invoke-Step "Official AAAI PDF compile" {
            .\scripts\compile_paper.ps1
        }
        Invoke-Step "Supplementary material PDF compile" {
            .\scripts\compile_supplementary_material.ps1
        }
        Invoke-Step "AAAI submission package" {
            .\scripts\build_submission_package.ps1 -SkipCompile
        }
        Invoke-Step "Supplementary reproducibility artifact" {
            .\scripts\build_supplementary_artifact.ps1
        }
        Invoke-Step "Submission readiness verifier" {
            python scripts\verify_submission_readiness.py
        }
    }

    Write-Host ""
    Write-Host "[done] Reported artifacts rebuilt successfully."
}
finally {
    Pop-Location
}
