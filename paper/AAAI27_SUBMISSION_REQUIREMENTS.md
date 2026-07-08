# AAAI-27 Submission Requirements Check

Source: https://aaai.org/conference/aaai/aaai-27/submission-instructions/

Checked on: 2026-07-07

This checklist maps the AAAI-27 submission instructions to the current HARP-Select review artifacts. It is an internal compliance aid; the official AAAI page remains authoritative.

## Deadlines From Official Page

- OpenReview paper submission opens: 2026-07-21.
- Abstract deadline: 2026-07-28 at 11:59 PM UTC-12.
- Full paper deadline: 2026-07-31 at 11:59 PM UTC-12.
- Supplementary material and code deadline: 2026-09-24 at 11:59 PM UTC-12.

## Format And Anonymity

| Requirement | Current Status | Evidence |
|---|---|---|
| AAAI two-column style from AAAI-27 author kit | Pass | `paper/main.tex` uses `\usepackage[submission]{aaai2027}` and the local `aaai2027.sty`/`.bst` files. |
| Trouble-free high-resolution PDF | Pass | `scripts/verify_submission_readiness.py` checks LaTeX logs, PDF page previews, and package synchronization. |
| US Letter 8.5 in x 11 in PDF | Pass | Rendered official page previews are 1275 x 1650 pixels at 150 DPI, matching US Letter. |
| Type 1 or TrueType fonts | Pass | `paper/main.log` records embedded Type 1 `.pfb` fonts and no bitmap/Type 3 font generation. |
| Double-blind review anonymity | Pass | Manuscript uses anonymous author/affiliation fields and the verifier checks submission markers. |
| Acknowledgements omitted for review | Pass | `scripts/verify_submission_readiness.py` blocks acknowledgement sections/environments. |
| Up to 7 pages of technical content plus references/checklist | Pass | Current official PDF is 8 total pages: pages 1-7 are technical content, and references begin on page 8 as verified by the LaTeX log marker. |

## Files To Submit Or Retain

| Item | Official Requirement | Current Artifact |
|---|---|---|
| Main review paper | PDF required at submission time | `paper/HARP_GNN_AAAI2027_official_compile.pdf` |
| Reproducibility checklist | Required and made available to reviewers | `paper/ReproducibilityChecklist_HARP_GNN.pdf` and `.tex` |
| Supplementary material/code/data | Due by 2026-09-24 at 11:59 PM UTC-12 when relevant | `paper/HARP_GNN_AAAI2027_supplementary_artifact.zip` |
| Source files | Required if accepted for publication, not for initial review submission | `paper/HARP_GNN_AAAI2027_submission_package.zip` is maintained as an internal clean-source validation package. |

## Current Caveats

- The format/readiness checks pass, but the scientific audit still marks the current evidence as red-amber for AAAI main-track competitiveness.
- Binary critical-heterophily results include complete 10-split Minesweeper and Tolokers ROC-AUC branch comparisons. Questions remains smoke-only and is not a main manuscript claim.
- Author-specific and OpenReview-form requirements are not locally verifiable. Before submission, manually confirm author limits, author registration, abstract metadata, subject areas, conflicts of interest, and any OpenReview checklist fields.
- Before submission, rerun:

```powershell
python scripts\verify_submission_readiness.py
```
