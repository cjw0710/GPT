from __future__ import annotations

import argparse
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN_TEX = PAPER / "main.tex"


def _read(path: Path) -> str:
    if not path.exists():
        raise AssertionError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def _strip_comments(tex: str) -> str:
    lines = []
    for line in tex.splitlines():
        match = re.search(r"(?<!\\)%", line)
        lines.append(line[: match.start()] if match else line)
    return "\n".join(lines)


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
            path = PAPER / rel
            if not path.exists():
                raise AssertionError(f"Missing bibliography file: {rel}")
            files.append(path)
    if not files:
        raise AssertionError("Missing bibliography command")
    return files


def check_included_artifacts(tex: str) -> list[str]:
    checked: list[Path] = []
    for rel in _brace_args("input", tex):
        if rel.startswith("|"):
            continue
        path = PAPER / rel
        if not path.exists():
            raise AssertionError(f"Missing LaTeX input target: {rel}")
        if path.stat().st_size == 0:
            raise AssertionError(f"Empty LaTeX input target: {rel}")
        checked.append(path)

    for rel in _brace_args("includegraphics", tex):
        path = PAPER / rel
        if not path.exists():
            raise AssertionError(f"Missing figure target: {rel}")
        if path.stat().st_size == 0:
            raise AssertionError(f"Empty figure target: {rel}")
        checked.append(path)

    return [f"included artifacts exist and are non-empty ({len(checked)} files)"]


def check_labels_and_refs(tex: str) -> list[str]:
    labels = set(_brace_args("label", tex))
    refs = set()
    for command in ["ref", "pageref", "autoref"]:
        refs.update(_brace_args(command, tex))

    missing_labels = sorted(ref for ref in refs if ref not in labels)
    if missing_labels:
        raise AssertionError("References without matching labels: " + ", ".join(missing_labels))

    unreferenced = sorted(label for label in labels if label.startswith(("tab:", "fig:")) and label not in refs)
    if unreferenced:
        raise AssertionError("Table/figure labels not referenced in text: " + ", ".join(unreferenced))

    return [f"labels and references are closed ({len(labels)} labels, {len(refs)} refs)"]


def check_citations(tex: str) -> list[str]:
    bib = "\n".join(_read(path) for path in _bibliography_files(tex))
    bib_keys = set(re.findall(r"@\w+\s*\{\s*([^,\s]+)", bib))
    cited: set[str] = set()
    for cite_body in re.findall(r"\\cite\w*\s*(?:\[[^\]]*\]\s*){0,2}\{([^{}]+)\}", tex):
        cited.update(key.strip() for key in cite_body.split(",") if key.strip())

    missing = sorted(key for key in cited if key not in bib_keys)
    if missing:
        raise AssertionError("Citation keys missing from references.bib: " + ", ".join(missing))
    return [f"citation keys resolve ({len(cited)} cited keys)"]


def check_submission_source(tex: str) -> list[str]:
    required = [
        r"\documentclass[letterpaper]{article}",
        r"\usepackage[submission]{aaai2027}",
        r"\usepackage{natbib}",
    ]
    missing = [needle for needle in required if needle not in tex]
    if missing:
        raise AssertionError("Missing AAAI submission source markers: " + ", ".join(missing))
    bib_names = [path.stem for path in _bibliography_files(tex)]
    if "references" not in bib_names:
        raise AssertionError("Main bibliography file 'references.bib' is not included")

    if "Anonymous Authors" not in tex or "Anonymous Affiliation" not in tex:
        raise AssertionError("Submission source is not anonymized")

    banned = [
        "TODO",
        "FIXME",
        "TBD",
        "placeholder",
        "dummy result",
        "fake result",
        "Lorem ipsum",
        r"\usepackage{hyperref}",
    ]
    found = [needle for needle in banned if needle in tex]
    if found:
        raise AssertionError("Submission source contains banned draft markers: " + ", ".join(found))
    return ["AAAI submission markers and anonymization checks passed"]


def run_checks() -> list[str]:
    tex = _strip_comments(_read(MAIN_TEX))
    messages: list[str] = []
    messages.extend(check_submission_source(tex))
    messages.extend(check_included_artifacts(tex))
    messages.extend(check_labels_and_refs(tex))
    messages.extend(check_citations(tex))
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify AAAI manuscript source integrity.")
    parser.add_argument("--quiet", action="store_true", help="Only print failures.")
    args = parser.parse_args()

    try:
        messages = run_checks()
    except AssertionError as exc:
        raise SystemExit(f"[failed] {exc}") from exc

    if not args.quiet:
        for message in messages:
            print(f"[ok] {message}")
        print("[ok] manuscript integrity verification passed")


if __name__ == "__main__":
    main()
