from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN_TEX = PAPER / "main.tex"
SCIENTIFIC_AUDIT = PAPER / "SCIENTIFIC_AUDIT.md"
PAIRED_TEST_FILES = [
    ROOT / "results" / "harp_select_paired_tests.csv",
]


class ClaimVerificationError(AssertionError):
    pass


@dataclass(frozen=True)
class EvidenceSummary:
    significant_positive: int
    significant_negative: int
    paired_rows: int


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ClaimVerificationError(message)


def _read_text(path: Path) -> str:
    _require(path.exists(), f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def _strip_tex_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        pieces = re.split(r"(?<!\\)%", line, maxsplit=1)
        lines.append(pieces[0])
    return "\n".join(lines)


def _line_records(text: str) -> list[tuple[int, str, str]]:
    records: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        compact = " ".join(line.split())
        if compact:
            records.append((line_number, compact, compact.lower()))
    return records


def _is_negated_or_bounded(lower: str) -> bool:
    bounded_markers = [
        "not ",
        "no ",
        "does not",
        "do not",
        "rather than",
        "without",
        "avoid",
        "bounded",
        "not significant",
        "negative",
        "deficit",
        "deficits",
        "trails",
        "fail to",
    ]
    return any(marker in lower for marker in bounded_markers)


def _paired_evidence() -> EvidenceSummary:
    frames: list[pd.DataFrame] = []
    for path in PAIRED_TEST_FILES:
        _require(path.exists(), f"Missing paired-test CSV: {path}")
        frame = pd.read_csv(path)
        required = {"diff_mean", "p_value"}
        missing = required - set(frame.columns)
        _require(not missing, f"Paired-test CSV {path} is missing columns: {sorted(missing)}")
        frames.append(frame)
    paired = pd.concat(frames, ignore_index=True)
    significant = paired["p_value"].astype(float) < 0.05
    positive = paired["diff_mean"].astype(float) > 0.0
    negative = paired["diff_mean"].astype(float) < 0.0
    return EvidenceSummary(
        significant_positive=int((significant & positive).sum()),
        significant_negative=int((significant & negative).sum()),
        paired_rows=len(paired),
    )


def check_audit_contains_top_conference_gate() -> list[str]:
    audit = _read_text(SCIENTIFIC_AUDIT)
    required_sections = [
        "## Executive Readout",
        "## Top-Conference Readiness Gate",
        "## Claim Boundaries for the Current Draft",
        "Claims to avoid:",
        "## Next Scientific Moves",
    ]
    missing = [section for section in required_sections if section not in audit]
    _require(not missing, "Scientific audit is missing required claim-boundary sections: " + ", ".join(missing))
    return ["scientific audit contains explicit top-conference readiness and claim-boundary sections"]


def check_no_unbounded_overclaims() -> list[str]:
    tex = _strip_tex_comments(_read_text(MAIN_TEX))
    patterns = [
        ("state-of-the-art claim", re.compile(r"state[- ]of[- ]the[- ]art|\bSOTA\b", re.IGNORECASE)),
        ("broad superiority claim", re.compile(r"consistently outperforms|outperforms all|dominates|superior to", re.IGNORECASE)),
        ("significant-gain claim", re.compile(r"significant(?:ly)? (?:improve|outperform|better|gain)|significant (?:WebKB )?gains?", re.IGNORECASE)),
        ("inflated improvement claim", re.compile(r"substantially improves?|large improvement|clear improvement", re.IGNORECASE)),
        ("readiness claim", re.compile(r"AAAI[- ]main ready|AAAI[- ]ready|ready for submission", re.IGNORECASE)),
    ]

    problems: list[str] = []
    for line_number, line, lower in _line_records(tex):
        for label, pattern in patterns:
            if pattern.search(line) and not _is_negated_or_bounded(lower):
                problems.append(f"{MAIN_TEX}:{line_number}: unbounded {label}: {line}")
    _require(not problems, "Unbounded top-conference claims found:\n" + "\n".join(problems))
    return ["manuscript has no unbounded SOTA, superiority, significant-gain, or readiness claims"]


def check_claims_match_paired_evidence() -> list[str]:
    tex = _strip_tex_comments(_read_text(MAIN_TEX))
    lower_tex = tex.lower()
    evidence = _paired_evidence()

    if evidence.significant_positive == 0:
        positive_significance_lines: list[str] = []
        for line_number, line, lower in _line_records(tex):
            mentions_positive_gain = any(token in lower for token in ["positive", "gain", "margin", "improv"])
            if "significant" in lower and mentions_positive_gain and not _is_negated_or_bounded(lower):
                positive_significance_lines.append(f"{MAIN_TEX}:{line_number}: {line}")
        _require(
            not positive_significance_lines,
            "Paired tests have no significant positive HARP-Select margins, but manuscript suggests positive significance:\n"
            + "\n".join(positive_significance_lines),
        )
        _require(
            "not significant at $p<0.05$" in lower_tex,
            "Manuscript should explicitly state that current positive margins are not significant at p<0.05.",
        )

    if evidence.significant_negative > 0:
        required_negative_markers = ["chameleon", "squirrel", "trails", "significant"]
        missing = [marker for marker in required_negative_markers if marker not in lower_tex]
        _require(
            not missing,
            "Manuscript should explicitly acknowledge significant negative larger-heterophily results; missing markers: "
            + ", ".join(missing),
        )

    _require(
        "does not claim final state-of-the-art performance" in lower_tex,
        "Limitations section should retain an explicit no-state-of-the-art boundary.",
    )
    return [
        f"manuscript claims match paired evidence ({evidence.significant_positive} significant positive, "
        f"{evidence.significant_negative} significant negative over {evidence.paired_rows} paired rows)"
    ]


def run_checks() -> list[str]:
    messages: list[str] = []
    messages.extend(check_audit_contains_top_conference_gate())
    messages.extend(check_no_unbounded_overclaims())
    messages.extend(check_claims_match_paired_evidence())
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that manuscript claims stay within current top-conference evidence boundaries.")
    parser.add_argument("--quiet", action="store_true", help="Only print failures.")
    args = parser.parse_args()

    try:
        messages = run_checks()
    except ClaimVerificationError as exc:
        raise SystemExit(f"[failed] {exc}") from exc

    if not args.quiet:
        for message in messages:
            print(f"[ok] {message}")
        print("[ok] top-conference claim verification passed")


if __name__ == "__main__":
    main()
