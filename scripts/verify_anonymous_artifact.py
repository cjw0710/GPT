from __future__ import annotations

import argparse
import re
from pathlib import Path


def _patterns() -> list[tuple[str, re.Pattern[str]]]:
    # Construct these strings from pieces so this scanner does not flag itself.
    old_project = "AAAI" + "2027" + "_GNN" + "_HARP"
    old_drive_path = "D:" + "\\\\" + old_project
    local_user_path = "C:" + "\\\\" + "Users" + "\\\\"
    clipboard_marker = "codex-" + "clipboard"
    temp_path = "AppData" + "\\\\Local" + "\\\\Temp"
    local_user_name = "\u76ae\u76ae\u4f1f"
    chinese_thread_title = (
        "\u5e2e\u6211\u4ece\u5934\u5b8c\u6210\u4e00\u4e2a"
        "\u548c\u56fe\u795e\u7ecf\u7f51\u7edc\u76f8\u5173\u7684\u5de5\u4f5c"
    )
    labels = [
        ("local user name", local_user_name),
        ("local thread title", chinese_thread_title),
        ("old project token", old_project),
        ("temporary clipboard marker", clipboard_marker),
        ("local temp path", temp_path),
        ("local user path", local_user_path),
        ("old drive path", old_drive_path),
    ]
    return [(label, re.compile(re.escape(pattern), re.IGNORECASE)) for label, pattern in labels]

TEXT_SUFFIXES = {
    "",
    ".bib",
    ".csv",
    ".json",
    ".md",
    ".ps1",
    ".py",
    ".tex",
    ".txt",
    ".yaml",
    ".yml",
}

SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "build",
    "data",
    "tools",
}


def iter_text_files(root: Path) -> list[Path]:
    files: list[Path] = []
    for path in root.rglob("*"):
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan an artifact for local deanonymizing traces.")
    parser.add_argument("--root", default=".", help="Artifact root to scan.")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    regexes = _patterns()
    findings: list[str] = []
    for path in iter_text_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="utf-8", errors="ignore")
        rel = path.relative_to(root)
        for label, regex in regexes:
            for match in regex.finditer(text):
                line_no = text.count("\n", 0, match.start()) + 1
                findings.append(f"{rel}:{line_no}: {label}")

    if findings:
        raise SystemExit("Deanonymizing/local traces found:\n" + "\n".join(findings))
    print(f"[ok] anonymous artifact scan passed ({len(iter_text_files(root))} text files)")


if __name__ == "__main__":
    main()
