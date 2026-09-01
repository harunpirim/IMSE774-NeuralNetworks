#!/usr/bin/env python3
"""Export the Python cells of a lecture note into a Colab-ready notebook.

Each note carries its code inline in ```{python} blocks. Students who want to
run that code -- rather than read it -- open the notebook this script writes,
which is linked from the top of every note as an "Open in Colab" badge.

    python scripts/qmd_to_notebook.py                  # every note that has code
    python scripts/qmd_to_notebook.py notes/02-*.qmd   # just these

Notebooks land in notebooks/weekNN-slug.ipynb. Section headings that precede a
code block become markdown cells, so the notebook keeps the shape of the note.
Existing notebooks are overwritten, so re-run this after editing a note.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "notes"
NOTEBOOK_DIR = ROOT / "notebooks"

SITE_URL = "https://harunpirim.github.io/IMSE774-NeuralNetworks/"
REPO_URL = "https://github.com/harunpirim/IMSE774-NeuralNetworks"

CODE_BLOCK = re.compile(r"^```\{python\}\n(.*?)^```$", re.M | re.S)
HEADING = re.compile(r"^(#{2,4})\s+(.*?)(?:\s*\{#.*\})?\s*$", re.M)
FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.S)


def front_matter_value(text: str, key: str) -> str:
    """Pull one scalar out of the YAML header without a YAML dependency."""
    match = FRONT_MATTER.search(text)
    if not match:
        return ""
    for line in match.group(1).splitlines():
        if line.startswith(f"{key}:"):
            return line.split(":", 1)[1].strip().strip('"')
    return ""


def strip_cell_options(source: str) -> tuple[str, str]:
    """Split a cell's `#|` option lines from its code, returning (label, code)."""
    label = ""
    body: list[str] = []
    for line in source.splitlines():
        if line.startswith("#|"):
            if line.startswith("#| label:"):
                label = line.split(":", 1)[1].strip()
            continue
        body.append(line)
    return label, "\n".join(body).strip("\n")


def top_level_headings(text: str) -> list[tuple[int, str]]:
    """Headings that are not inside a `:::` callout or a fenced code block.

    Callouts title themselves with `##`, so a naive scan would pick up "Notation"
    and "Figure credits" as if they were sections of the note.
    """
    found: list[tuple[int, str]] = []
    offset, div_depth, in_code = 0, 0, False
    for line in text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_code = not in_code
        elif stripped.startswith(":::") and not in_code:
            div_depth += -1 if set(stripped) == {":"} else 1
        elif div_depth == 0 and not in_code:
            match = HEADING.match(line.rstrip("\n"))
            if match:
                found.append((offset, match.group(2).strip()))
        offset += len(line)
    return found


def heading_before(headings: list[tuple[int, str]], position: int) -> str:
    """The nearest section heading above `position`, for a markdown cell."""
    last = ""
    for start, title in headings:
        if start >= position:
            break
        last = title
    return last


def markdown_cell(lines: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": lines}


def code_cell(code: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.splitlines(keepends=True),
    }


def build(qmd: Path) -> Path | None:
    text = qmd.read_text()
    blocks = list(CODE_BLOCK.finditer(text))
    if not blocks:
        return None

    title = front_matter_value(text, "title") or qmd.stem
    subtitle = front_matter_value(text, "subtitle")
    page = f"{SITE_URL}notes/{qmd.stem}.html"

    cells = [markdown_cell([
        f"# {title}\n",
        "\n",
        f"**IMSE 774 · Neural Networks · {subtitle}**\n" if subtitle else "**IMSE 774 · Neural Networks**\n",
        "\n",
        f"Every code cell from the [lecture note]({page}), in the order it appears\n",
        "there. The prose, figures, and exercises live on that page; this notebook is\n",
        "for running and changing the code.\n",
        "\n",
        "Run a cell with **Shift+Enter**. Run the setup cell first — the later cells\n",
        "depend on the names it defines.\n",
        "\n",
        f"Source: [{REPO_URL}]({REPO_URL})\n",
    ])]

    headings = top_level_headings(text)
    seen = ""
    for block in blocks:
        label, code = strip_cell_options(block.group(1))
        if not code:
            continue
        heading = "Setup" if label == "setup" else heading_before(headings, block.start())
        if heading and heading != seen:
            cells.append(markdown_cell([f"## {heading}\n"]))
            seen = heading
        if label:
            cells.append(markdown_cell([f"`{label}`\n"]))
        cells.append(code_cell(code))

    notebook = {
        "cells": cells,
        "metadata": {
            "colab": {"provenance": [], "toc_visible": True},
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }

    NOTEBOOK_DIR.mkdir(exist_ok=True)
    out = NOTEBOOK_DIR / f"week{qmd.stem.split('-')[0]}-{qmd.stem.split('-', 1)[1]}.ipynb"
    out.write_text(json.dumps(notebook, indent=1) + "\n")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("notes", nargs="*", type=Path,
                        help="note files to export (default: every note with code)")
    args = parser.parse_args()

    targets = args.notes or sorted(NOTES_DIR.glob("*.qmd"))
    for qmd in targets:
        out = build(qmd)
        if out is None:
            print(f"    {qmd.name}: no code cells, skipped")
        else:
            n = len(json.loads(out.read_text())["cells"])
            print(f"==> {out.relative_to(ROOT)} ({n} cells)")


if __name__ == "__main__":
    main()
