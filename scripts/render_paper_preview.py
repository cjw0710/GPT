from __future__ import annotations

import html
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
MAIN = PAPER / "main.tex"
OUT = PAPER / "preview.html"


def clean_inline(text: str) -> str:
    text = text.replace("\\method", "HARP-GNN")
    text = text.replace("\\%", "%")
    text = text.replace("\\_", "_")
    text = text.replace("~", " ")
    text = re.sub(r"\\cite\{([^}]+)\}", r"[citation: \1]", text)
    text = re.sub(r"\\ref\{([^}]+)\}", r"\1", text)
    text = re.sub(r"\\textbf\{([^{}]+)\}", r"<strong>\1</strong>", text)
    text = text.replace("$\\pm$", "±")
    text = text.replace("$p<0.05$", "p<0.05")
    text = text.replace("$<0.001$", "<0.001")
    text = text.replace("$p$", "p")
    text = text.replace("$h=0.2$", "h=0.2")
    text = text.replace("\\ldots", "...")
    text = re.sub(r"\$([^$]+)\$", r"<span class=\"math\">\1</span>", text)
    return text


def latex_rows_to_html(tabular: str) -> str:
    tabular = re.sub(r"\\begin\{tabular\}\{[^}]+\}", "", tabular)
    tabular = tabular.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "")
    tabular = tabular.replace("\\end{tabular}", "")
    rows = []
    for raw_row in re.split(r"\\\\", tabular):
        raw_row = raw_row.strip()
        if not raw_row:
            continue
        cells = [clean_inline(cell.strip()) for cell in raw_row.split("&")]
        rows.append(cells)
    if not rows:
        return ""
    head = "".join(f"<th>{cell}</th>" for cell in rows[0])
    body = "\n".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        for row in rows[1:]
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def extract_caption(block: str) -> str:
    match = re.search(r"\\caption\{(.+?)\}", block, flags=re.S)
    return clean_inline(" ".join(match.group(1).split())) if match else ""


def extract_label(block: str) -> str:
    match = re.search(r"\\label\{([^}]+)\}", block)
    return match.group(1) if match else ""


def render_table(block: str, table_number: int) -> tuple[str, str | None]:
    caption = extract_caption(block)
    label = extract_label(block)
    input_match = re.search(r"\\input\{([^}]+)\}", block)
    tabular = ""
    if input_match:
        input_name = input_match.group(1)
        input_path = PAPER / input_name
        if input_path.suffix != ".tex":
            input_path = input_path.with_suffix(".tex")
        if input_path.exists():
            tabular = input_path.read_text(encoding="utf-8")
    if not tabular:
        tabular_match = re.search(r"(\\begin\{tabular\}.*?\\end\{tabular\})", block, flags=re.S)
        tabular = tabular_match.group(1) if tabular_match else ""
    body = latex_rows_to_html(tabular)
    html_block = (
        f"<figure class=\"table-block\">"
        f"<figcaption>Table {table_number}. {caption}</figcaption>"
        f"{body}"
        f"</figure>"
    )
    return html_block, label


def render_figure(block: str, figure_number: int) -> tuple[str, str | None]:
    caption = extract_caption(block)
    label = extract_label(block)
    include = re.search(r"\\includegraphics(?:\[[^\]]+\])?\{([^}]+)\}", block)
    if include:
        src = include.group(1)
        body = f"<img src=\"{html.escape(src)}\" alt=\"{html.escape(caption)}\">"
    else:
        body = "<div class=\"figure-placeholder\">Figure placeholder</div>"
    return (
        f"<figure class=\"figure-block\"><figcaption>Figure {figure_number}. {caption}</figcaption>{body}</figure>",
        label,
    )


def strip_preamble(text: str) -> str:
    text = text.split("\\begin{document}", 1)[-1]
    text = text.rsplit("\\bibliographystyle", 1)[0]
    text = text.replace("\\maketitle", "")
    return text


def render_body(text: str) -> str:
    text = strip_preamble(text)
    title = re.search(r"\\title\{(.+?)\}", MAIN.read_text(encoding="utf-8"), flags=re.S)
    title_html = clean_inline(" ".join(title.group(1).split())) if title else "Paper Preview"

    abstract_match = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", text, flags=re.S)
    abstract = clean_inline(" ".join(abstract_match.group(1).split())) if abstract_match else ""
    text = re.sub(r"\\begin\{abstract\}.+?\\end\{abstract\}", "", text, flags=re.S)

    label_to_number: dict[str, str] = {}
    rendered_blocks: list[tuple[str, str]] = []

    table_no = 0
    figure_no = 0

    def stash_table(match: re.Match[str]) -> str:
        nonlocal table_no
        table_no += 1
        rendered, label = render_table(match.group(0), table_no)
        key = f"@@BLOCK{len(rendered_blocks)}@@"
        rendered_blocks.append((key, rendered))
        if label:
            label_to_number[label] = str(table_no)
        return key

    def stash_figure(match: re.Match[str]) -> str:
        nonlocal figure_no
        figure_no += 1
        rendered, label = render_figure(match.group(0), figure_no)
        key = f"@@BLOCK{len(rendered_blocks)}@@"
        rendered_blocks.append((key, rendered))
        if label:
            label_to_number[label] = str(figure_no)
        return key

    text = re.sub(r"\\begin\{table\}.*?\\end\{table\}", stash_table, text, flags=re.S)
    text = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", stash_figure, text, flags=re.S)

    for label, number in label_to_number.items():
        text = text.replace(f"Table~\\ref{{{label}}}", f"Table {number}")
        text = text.replace(f"Figure~\\ref{{{label}}}", f"Figure {number}")
        text = text.replace(f"\\ref{{{label}}}", number)

    text = re.sub(r"\\section\{([^}]+)\}", r"\n<h2>\1</h2>\n", text)
    text = re.sub(r"\\paragraph\{([^}]+)\}", r"\n<h3>\1</h3>\n", text)
    text = re.sub(r"\\begin\{itemize\}", "<ul>", text)
    text = re.sub(r"\\end\{itemize\}", "</ul>", text)
    text = re.sub(r"\\item\s+", "<li>", text)
    text = re.sub(r"\\begin\{equation\}(.+?)\\end\{equation\}", r"<div class=\"equation\">\1</div>", text, flags=re.S)
    text = text.replace("\\centering", "")
    text = re.sub(r"\\pdfinfo\{.*?\}", "", text, flags=re.S)
    text = re.sub(r"\\setcounter\{[^}]+\}\{[^}]+\}", "", text)
    text = re.sub(r"\\newcommand\{[^}]+\}\{[^}]+\}", "", text)
    text = re.sub(r"\\author\{.*?\}", "", text, flags=re.S)
    text = re.sub(r"\\affiliations\{.*?\}", "", text, flags=re.S)
    text = re.sub(r"\\title\{.*?\}", "", text, flags=re.S)

    paragraphs = []
    for chunk in re.split(r"\n\s*\n", text):
        chunk = chunk.strip()
        if not chunk:
            continue
        for key, rendered in rendered_blocks:
            chunk = chunk.replace(key, rendered)
        if chunk.startswith(("<h2", "<h3", "<ul", "<figure", "<div")):
            paragraphs.append(clean_inline(chunk))
        else:
            paragraphs.append(f"<p>{clean_inline(chunk)}</p>")

    body = "\n".join(paragraphs)
    return title_html, abstract, body


def main() -> None:
    title, abstract, body = render_body(MAIN.read_text(encoding="utf-8"))
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
  @page {{ size: letter; margin: 0.75in; }}
  body {{ font-family: "Times New Roman", Times, serif; margin: 0 auto; max-width: 8.5in; color: #111; background: #f3f3f3; }}
  .page {{ background: white; padding: 0.75in; box-shadow: 0 2px 16px rgba(0,0,0,.15); }}
  h1 {{ font-size: 22px; text-align: center; margin: 0 0 8px; }}
  .authors {{ text-align: center; margin-bottom: 16px; }}
  .abstract {{ border-top: 1px solid #bbb; border-bottom: 1px solid #bbb; padding: 8px 0; margin: 14px 0; font-size: 10.5px; }}
  .content {{ column-count: 2; column-gap: 0.28in; font-size: 10px; line-height: 1.18; }}
  h2 {{ font-size: 12px; margin: 12px 0 5px; break-after: avoid; }}
  h3 {{ font-size: 10.5px; margin: 8px 0 3px; font-style: italic; break-after: avoid; }}
  p {{ margin: 0 0 6px; text-align: justify; }}
  figure {{ break-inside: avoid; margin: 8px 0 10px; }}
  figcaption {{ font-size: 9px; font-weight: bold; margin-bottom: 4px; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 7px; }}
  th, td {{ border-top: 1px solid #bbb; padding: 2px 3px; text-align: center; vertical-align: top; }}
  th:first-child, td:first-child {{ text-align: left; }}
  thead th {{ border-top: 1px solid #000; border-bottom: 1px solid #000; }}
  tbody tr:last-child td {{ border-bottom: 1px solid #000; }}
  img {{ max-width: 100%; display: block; margin: 0 auto; }}
  .equation {{ font-family: "Times New Roman", Times, serif; text-align: center; margin: 6px 0; font-size: 10px; }}
  .math {{ font-style: italic; }}
  @media print {{ body {{ background: white; }} .page {{ box-shadow: none; padding: 0; }} }}
</style>
</head>
<body>
<main class="page">
<h1>{title}</h1>
<div class="authors">Anonymous Authors<br>Anonymous Affiliation</div>
<section class="abstract"><strong>Abstract.</strong> {abstract}</section>
<section class="content">
{body}
</section>
</main>
</body>
</html>
"""
    OUT.write_text(html_doc, encoding="utf-8")
    print(f"[saved] {OUT}")


if __name__ == "__main__":
    main()
