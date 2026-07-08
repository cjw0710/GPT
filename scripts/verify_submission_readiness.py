from __future__ import annotations

import argparse
import re
import struct
import zipfile
from pathlib import Path

from verify_top_conference_claims import ClaimVerificationError
from verify_top_conference_claims import run_checks as run_top_conference_claim_checks


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
BUILD = ROOT / "build"

OFFICIAL_PDF = PAPER / "HARP_GNN_AAAI2027_official_compile.pdf"
SUBMISSION_ZIP = PAPER / "HARP_GNN_AAAI2027_submission_package.zip"
SUPPLEMENTARY_ZIP = PAPER / "HARP_GNN_AAAI2027_supplementary_artifact.zip"
CHECKLIST_TEX = PAPER / "ReproducibilityChecklist_HARP_GNN.tex"
CHECKLIST_PDF = PAPER / "ReproducibilityChecklist_HARP_GNN.pdf"
SUPPLEMENTARY_MATERIAL_TEX = PAPER / "supplementary_material.tex"
SUPPLEMENTARY_MATERIAL_PDF = PAPER / "HARP_GNN_AAAI2027_supplementary_material.pdf"
MAIN_TEX = PAPER / "main.tex"
MAX_TECHNICAL_PAGES = 7
MIN_SUPPLEMENTARY_PAGES = 4
PREVIEW_DPI = 150
LETTER_PREVIEW_SIZE = (1275, 1650)


class ReadinessError(AssertionError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _read_text(path: Path) -> str:
    _require(path.exists(), f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def _read_bytes(path: Path) -> bytes:
    _require(path.exists(), f"Missing file: {path}")
    return path.read_bytes()


def _assert_file(path: Path, min_bytes: int = 1) -> None:
    _require(path.exists(), f"Missing file: {path}")
    _require(path.is_file(), f"Not a file: {path}")
    size = path.stat().st_size
    _require(size >= min_bytes, f"File is unexpectedly small: {path} ({size} bytes)")


def _zip_names(path: Path) -> list[str]:
    _assert_file(path, min_bytes=100)
    with zipfile.ZipFile(path) as archive:
        bad_member = archive.testzip()
        _require(bad_member is None, f"Corrupt zip member in {path.name}: {bad_member}")
        return [name.replace("\\", "/") for name in archive.namelist()]


def _zip_read(path: Path, member: str) -> bytes:
    with zipfile.ZipFile(path) as archive:
        return archive.read(member)


def _latest_page_count_from_log(path: Path) -> int:
    text = _read_text(path)
    matches = list(re.finditer(r"Output written on .*?\((\d+) pages?,", text))
    _require(matches, f"Could not find PDF page count in log: {path}")
    return int(matches[-1].group(1))


def _references_start_page_from_log(path: Path) -> int:
    text = _read_text(path)
    matches = re.findall(r"HARP_REFERENCES_START_PAGE=(\d+)", text)
    _require(matches, f"Could not find references-start page marker in log: {path}")
    return int(matches[-1])


def _png_dimensions(path: Path) -> tuple[int, int]:
    data = _read_bytes(path)
    _require(data.startswith(b"\x89PNG\r\n\x1a\n"), f"Not a PNG file: {path}")
    _require(len(data) >= 24, f"PNG file is truncated: {path}")
    return struct.unpack(">II", data[16:24])


def _brace_args(command: str, tex: str) -> list[str]:
    return re.findall(rf"\\{command}\s*(?:\[[^\]]*\]\s*)?\{{([^{{}}]+)\}}", tex)


def _bibliography_files(tex: str) -> list[Path]:
    files: list[Path] = []
    for body in _brace_args("bibliography", tex):
        for name in body.split(","):
            name = name.strip()
            if not name:
                continue
            rel = name if name.endswith(".bib") else f"{name}.bib"
            files.append(PAPER / rel)
    _require(files, "Missing bibliography command")
    return files


def check_core_artifacts() -> list[str]:
    _assert_file(MAIN_TEX, min_bytes=1000)
    _assert_file(PAPER / "references.bib", min_bytes=1000)
    for bib_path in _bibliography_files(_read_text(MAIN_TEX)):
        _assert_file(bib_path, min_bytes=100)
    _assert_file(PAPER / "aaai2027.sty", min_bytes=1000)
    _assert_file(PAPER / "aaai2027.bst", min_bytes=1000)
    _assert_file(OFFICIAL_PDF, min_bytes=100_000)
    _assert_file(SUBMISSION_ZIP, min_bytes=100_000)
    _assert_file(SUPPLEMENTARY_ZIP, min_bytes=100_000)
    _assert_file(CHECKLIST_TEX, min_bytes=1000)
    _assert_file(CHECKLIST_PDF, min_bytes=10_000)
    _assert_file(SUPPLEMENTARY_MATERIAL_TEX, min_bytes=1000)
    _assert_file(SUPPLEMENTARY_MATERIAL_PDF, min_bytes=100_000)
    return ["primary paper, checklist, and package artifacts exist with plausible sizes"]


def check_manuscript_source() -> list[str]:
    tex = _read_text(MAIN_TEX)
    required = [
        r"\documentclass[letterpaper]{article}",
        r"\usepackage[submission]{aaai2027}",
        "Anonymous Authors",
        "Anonymous Affiliation",
    ]
    missing = [item for item in required if item not in tex]
    _require(not missing, "Manuscript is missing submission markers: " + ", ".join(missing))
    bib_names = [path.stem for path in _bibliography_files(tex)]
    _require("references" in bib_names, "Main bibliography file 'references.bib' is not included")

    banned = [
        "TODO",
        "FIXME",
        "TBD",
        "Lorem ipsum",
        "dummy result",
        "fake result",
        "Future extensions should",
        "comparison set should be expanded",
        "toward a full AAAI 2027 submission package",
    ]
    found = [item for item in banned if item in tex]
    _require(not found, "Manuscript still contains draft wording: " + ", ".join(found))
    ack_patterns = [
        r"\\section\*?\{Acknowledg",
        r"\\begin\{ack",
        r"\\acks\{",
    ]
    acknowledgements = [pattern for pattern in ack_patterns if re.search(pattern, tex, re.IGNORECASE)]
    _require(
        not acknowledgements,
        "Review submission must omit acknowledgements; found patterns: " + ", ".join(acknowledgements),
    )
    return ["manuscript source has submission markers and no blocked draft wording"]


def check_latex_logs() -> list[str]:
    logs = [
        PAPER / "main.log",
        CHECKLIST_TEX.with_suffix(".log"),
        PAPER / "supplementary_material.log",
        BUILD / "submission_package" / "validation_source" / "main.log",
    ]
    blocking_patterns = [
        re.compile(r"undefined citations", re.IGNORECASE),
        re.compile(r"undefined references", re.IGNORECASE),
        re.compile(r"Citation .* undefined", re.IGNORECASE),
        re.compile(r"Reference .* undefined", re.IGNORECASE),
        re.compile(r"Rerun to get citations", re.IGNORECASE),
        re.compile(r"Label\(s\) may have changed", re.IGNORECASE),
        re.compile(r"Fatal error", re.IGNORECASE),
        re.compile(r"Emergency stop", re.IGNORECASE),
        re.compile(r"Undefined control sequence", re.IGNORECASE),
        re.compile(r"Missing \$ inserted", re.IGNORECASE),
        re.compile(r"Overfull \\hbox", re.IGNORECASE),
        re.compile(r"Overfull \\vbox", re.IGNORECASE),
    ]

    problems: list[str] = []
    underfull_count = 0
    for path in logs:
        text = _read_text(path)
        for line_number, line in enumerate(text.splitlines(), start=1):
            if "Underfull" in line:
                underfull_count += 1
            for pattern in blocking_patterns:
                if pattern.search(line):
                    problems.append(f"{path}:{line_number}: {line.strip()}")

    _require(not problems, "Blocking LaTeX log messages found:\n" + "\n".join(problems))
    return [f"LaTeX logs have no blocking warnings/errors; underfull messages only ({underfull_count})"]


def check_supplementary_material() -> list[str]:
    tex = _read_text(SUPPLEMENTARY_MATERIAL_TEX)
    _require("Anonymous Authors" in tex, "Supplementary material should remain anonymized")
    _require("github.com" not in tex.lower(), "Supplementary material must not cite a mutable GitHub URL")
    _require(
        "Supplementary Material for" in tex,
        "Supplementary material should have an explicit title",
    )
    _require(
        "Filter-Response Interpretation" in tex and "p_H(\\lambda)" in tex,
        "Supplementary material should retain the residual filter-response derivation",
    )
    page_count = _latest_page_count_from_log(PAPER / "supplementary_material.log")
    _require(
        page_count >= MIN_SUPPLEMENTARY_PAGES,
        f"Supplementary material has only {page_count} pages; expected at least {MIN_SUPPLEMENTARY_PAGES}",
    )
    return [f"supplementary material PDF compiles and remains anonymous ({page_count} pages)"]


def check_pdf_previews() -> list[str]:
    page_count = _latest_page_count_from_log(PAPER / "main.log")
    validation_page_count = _latest_page_count_from_log(
        BUILD / "submission_package" / "validation_source" / "main.log"
    )
    references_start_page = _references_start_page_from_log(PAPER / "main.log")
    validation_references_start_page = _references_start_page_from_log(
        BUILD / "submission_package" / "validation_source" / "main.log"
    )
    expected_references_start_page = MAX_TECHNICAL_PAGES + 1
    _require(
        references_start_page == expected_references_start_page,
        f"References start on page {references_start_page}; expected page {expected_references_start_page} "
        f"after {MAX_TECHNICAL_PAGES} technical pages",
    )
    _require(
        page_count == validation_page_count,
        f"Paper and submission-source validation page counts differ: {page_count} vs {validation_page_count}",
    )
    _require(
        references_start_page == validation_references_start_page,
        f"Paper and submission-source references-start pages differ: "
        f"{references_start_page} vs {validation_references_start_page}",
    )
    _require(
        page_count >= references_start_page,
        f"Official PDF has {page_count} pages, but references are marked as starting on page {references_start_page}",
    )

    previews = sorted(PAPER.glob("HARP_GNN_AAAI2027_official_page*.png"))
    _require(len(previews) == page_count, f"Expected {page_count} page previews, found {len(previews)}")
    for preview in previews:
        _assert_file(preview, min_bytes=10_000)
        width, height = _png_dimensions(preview)
        _require(
            (width, height) == LETTER_PREVIEW_SIZE,
            f"{preview.name} renders to {width}x{height}, expected {LETTER_PREVIEW_SIZE[0]}x{LETTER_PREVIEW_SIZE[1]} "
            f"for US Letter at {PREVIEW_DPI} DPI",
        )
    return [
        f"official PDF renders {page_count} total pages with references starting on page {references_start_page} "
        f"after {MAX_TECHNICAL_PAGES} technical pages (US Letter at {PREVIEW_DPI} DPI)"
    ]


def check_pdf_font_format() -> list[str]:
    log_text = _read_text(PAPER / "main.log")
    type1_fonts = re.findall(r"fonts/type1/[^<>\s]+\.pfb", log_text, flags=re.IGNORECASE)
    _require(type1_fonts, "LaTeX log does not show embedded Type 1 .pfb fonts")
    blocked_patterns = [
        re.compile(r"\.pk\b", re.IGNORECASE),
        re.compile(r"mktexpk", re.IGNORECASE),
        re.compile(r"Type ?3", re.IGNORECASE),
        re.compile(r"bitmap", re.IGNORECASE),
    ]
    problems = []
    for line_number, line in enumerate(log_text.splitlines(), start=1):
        for pattern in blocked_patterns:
            if pattern.search(line):
                problems.append(f"{PAPER / 'main.log'}:{line_number}: {line.strip()}")
    _require(not problems, "PDF font log suggests bitmap or Type 3 fonts:\n" + "\n".join(problems))
    return [f"PDF font log shows Type 1 .pfb fonts and no bitmap/Type 3 font generation ({len(type1_fonts)} fonts)"]


def check_checklist() -> list[str]:
    text = _read_text(CHECKLIST_TEX)
    marker = "% The questions start here"
    _require(marker in text, "Could not locate checklist question marker")
    formal_body = text.split(marker, 1)[1]
    placeholders = formal_body.count("Type your response here")
    _require(placeholders == 0, f"Checklist still has {placeholders} formal placeholders")
    return ["filled reproducibility checklist has no formal placeholders"]


def check_top_conference_claims() -> list[str]:
    try:
        run_top_conference_claim_checks()
    except ClaimVerificationError as exc:
        raise ReadinessError(f"Top-conference claim verification failed: {exc}") from exc
    return ["top-conference claim verifier passes against current evidence"]


def check_submission_zip() -> list[str]:
    names = _zip_names(SUBMISSION_ZIP)
    name_set = set(names)
    tex = _read_text(MAIN_TEX)
    bibliography_members = {f"source/{path.name}" for path in _bibliography_files(tex)}
    figure_members = {f"source/{rel}" for rel in _brace_args("includegraphics", tex)}
    required = {
        "HARP_GNN_AAAI2027_official_compile.pdf",
        "PACKAGE_MANIFEST.txt",
        "source/main.tex",
        "source/aaai2027.sty",
        "source/aaai2027.bst",
    } | bibliography_members | figure_members
    missing = sorted(required - name_set)
    _require(not missing, "Submission package missing required members: " + ", ".join(missing))

    forbidden_suffixes = (".aux", ".log", ".blg", ".out", ".toc", ".zip")
    forbidden_roots = ("build/", "data/", "results/", "scripts/", "src/", "tools/")
    bad_members = [
        name
        for name in names
        if name.endswith(forbidden_suffixes) or name.startswith(forbidden_roots)
    ]
    _require(not bad_members, "Submission package contains forbidden members: " + ", ".join(bad_members[:20]))

    _require(
        _zip_read(SUBMISSION_ZIP, "source/main.tex") == _read_bytes(MAIN_TEX),
        "Submission package source/main.tex is stale relative to paper/main.tex",
    )
    _require(
        _zip_read(SUBMISSION_ZIP, "HARP_GNN_AAAI2027_official_compile.pdf") == _read_bytes(OFFICIAL_PDF),
        "Submission package PDF is stale relative to the current official PDF",
    )
    return [f"submission zip is valid, minimal, and synchronized ({len(names)} members)"]


def check_supplementary_zip() -> list[str]:
    names = _zip_names(SUPPLEMENTARY_ZIP)
    root = "HARP_GNN_AAAI2027_reproducibility_artifact/"
    _require(any(name.startswith(root) for name in names), "Supplementary zip has no artifact root folder")

    required = {
        root + "README.md",
        root + "LICENSE",
        root + "ANONYMITY_AND_RELEASE.md",
        root + "requirements.txt",
        root + "SUPPLEMENTARY_ARTIFACT_README.md",
        root + "src/harp_gnn/models.py",
        root + "src/harp_gnn/utils.py",
        root + "scripts/verify_implementation.py",
        root + "scripts/verify_manuscript_integrity.py",
        root + "scripts/verify_reported_results.py",
        root + "scripts/verify_submission_readiness.py",
        root + "scripts/verify_top_conference_claims.py",
        root + "scripts/compile_supplementary_material.ps1",
        root + "scripts/generate_scientific_audit.py",
        root + "scripts/merge_harp_esep_results.py",
        root + "scripts/plot_framework.py",
        root + "scripts/plot_harp_select_framework.py",
        root + "configs/geom_gcn_large.yaml",
        root + "configs/geom_gcn_harp_esep.yaml",
        root + "configs/harp_x_diagnostic.yaml",
        root + "results/result_audit.csv",
        root + "results/geom_gcn_harp_esep.csv",
        root + "results/geom_gcn_large_with_harp_esep.csv",
        root + "results/geom_gcn_harp_esep_paired_tests.csv",
        root + "results/harp_x_diagnostic.csv",
        root + "results/harp_select_threshold_sensitivity.csv",
        root + "results/harp_select_threshold_sensitivity_overall.csv",
        root + "results/harp_select_margin_calibration.csv",
        root + "results/harp_select_training_cost.csv",
        root + "results/harp_select_training_cost_per_split.csv",
        root + "paper/main.tex",
        root + "paper/supplementary_material.tex",
        root + "paper/AAAI27_SUBMISSION_REQUIREMENTS.md",
        root + "paper/references_additions.bib",
        root + "paper/SCIENTIFIC_AUDIT.md",
        root + "paper/HARP_X_DIAGNOSTIC.md",
        root + "paper/HARP_SELECTOR_SENSITIVITY.md",
        root + "paper/HARP_SELECTOR_COST.md",
        root + "paper/tables/geom_gcn_harp_esep_results.tex",
        root + "paper/tables/geom_gcn_harp_esep_paired_tests.tex",
        root + "paper/tables/harp_x_diagnostic_results.tex",
        root + "paper/tables/harp_select_threshold_sensitivity.tex",
        root + "paper/tables/harp_select_training_cost.tex",
        root + "paper/figures/harp_framework.png",
        root + "paper/figures/harp_select_framework.png",
        root + "paper/HARP_GNN_AAAI2027_official_compile.pdf",
        root + "paper/HARP_GNN_AAAI2027_supplementary_material.pdf",
        root + "paper/ReproducibilityChecklist_HARP_GNN.pdf",
    }
    name_set = set(names)
    missing = sorted(required - name_set)
    _require(not missing, "Supplementary artifact missing required members: " + ", ".join(missing))

    bad_members: list[str] = []
    for name in names:
        inner = name[len(root) :] if name.startswith(root) else name
        if (
            "__pycache__" in inner
            or inner.endswith((".pyc", ".aux", ".log", ".zip"))
            or inner.startswith(("build/", "data/", "tools/"))
        ):
            bad_members.append(name)
    _require(not bad_members, "Supplementary artifact contains forbidden members: " + ", ".join(bad_members[:20]))

    _require(
        _zip_read(SUPPLEMENTARY_ZIP, root + "paper/main.tex") == _read_bytes(MAIN_TEX),
        "Supplementary artifact paper/main.tex is stale relative to paper/main.tex",
    )
    _require(
        _zip_read(SUPPLEMENTARY_ZIP, root + "paper/HARP_GNN_AAAI2027_official_compile.pdf")
        == _read_bytes(OFFICIAL_PDF),
        "Supplementary artifact PDF is stale relative to the current official PDF",
    )
    _require(
        _zip_read(SUPPLEMENTARY_ZIP, root + "paper/supplementary_material.tex")
        == _read_bytes(SUPPLEMENTARY_MATERIAL_TEX),
        "Supplementary artifact supplementary_material.tex is stale relative to paper/supplementary_material.tex",
    )
    _require(
        _zip_read(SUPPLEMENTARY_ZIP, root + "paper/HARP_GNN_AAAI2027_supplementary_material.pdf")
        == _read_bytes(SUPPLEMENTARY_MATERIAL_PDF),
        "Supplementary artifact supplementary material PDF is stale relative to the current supplementary PDF",
    )
    return [f"supplementary artifact zip is clean and synchronized ({len(names)} members)"]


def run_checks() -> list[str]:
    messages: list[str] = []
    messages.extend(check_core_artifacts())
    messages.extend(check_manuscript_source())
    messages.extend(check_latex_logs())
    messages.extend(check_supplementary_material())
    messages.extend(check_pdf_previews())
    messages.extend(check_pdf_font_format())
    messages.extend(check_checklist())
    messages.extend(check_top_conference_claims())
    messages.extend(check_submission_zip())
    messages.extend(check_supplementary_zip())
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify final AAAI submission artifact readiness.")
    parser.add_argument("--quiet", action="store_true", help="Only print failures.")
    args = parser.parse_args()

    try:
        messages = run_checks()
    except ReadinessError as exc:
        raise SystemExit(f"[failed] {exc}") from exc

    if not args.quiet:
        for message in messages:
            print(f"[ok] {message}")
        print("[ok] submission readiness verification passed")


if __name__ == "__main__":
    main()
