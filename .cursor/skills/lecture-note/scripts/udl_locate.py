#!/usr/bin/env python3
"""Locate sections, figures, problems, and notebooks in Prince, Understanding Deep Learning.

The Kindle/Apple Books highlight exports report ebook locations and ebook page
numbers, neither of which match the printed book that students cite. This script
reads the textbook PDF and reports *printed* page numbers.

    python .cursor/skills/lecture-note/scripts/udl_locate.py --chapter 3
    python .cursor/skills/lecture-note/scripts/udl_locate.py --chapter 3 --dump

--dump prints the full chapter text, which is the fastest way to check a
paraphrase or a page number against what the book actually says.

Requires pypdf (pip install pypdf) and the textbook PDF at the repository root.
The PDF is gitignored as copyrighted material, so it must be supplied locally.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
PDF_GLOB = "UnderstandingDeepLearning*.pdf"

# Caption labels are set on their own line by the book's LaTeX class, which is
# what separates a real caption from a sentence that happens to open with
# "Figure 1.2 depicts ...".
FIGURE_CAPTION = r"Figure\s*\n\s*({ch}\.\d+)\b"


def find_pdf(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit).expanduser()
        if not p.is_file():
            sys.exit(f"error: no PDF at {p}")
        return p
    matches = sorted(REPO_ROOT.glob(PDF_GLOB))
    if not matches:
        sys.exit(
            f"error: no file matching {PDF_GLOB} in {REPO_ROOT}.\n"
            "The textbook is gitignored as copyrighted material; put your copy at the\n"
            "repository root, or pass --pdf /path/to/UnderstandingDeepLearning.pdf"
        )
    return matches[-1]


def detect_offset(pages: list[str]) -> int:
    """Return the constant such that printed_page = pdf_index_1based - offset.

    Self-calibrating: every page carries its printed number in the running
    header, so the correct offset recurs on every page. Section numbers and
    stray digits produce offsets that differ from page to page, so the true
    value wins by a wide margin.
    """
    votes: Counter[int] = Counter()
    for i, text in enumerate(pages[10:400], start=11):
        head = " ".join(text.split("\n")[:3])
        for tok in re.findall(r"\b\d{1,3}\b", head):
            votes[i - int(tok)] += 1
    if not votes:
        sys.exit("error: could not calibrate page offset; pass --offset explicitly")
    return votes.most_common(1)[0][0]


def chapter_starts(pages: list[str]) -> dict[int, int]:
    """Map chapter number -> 1-based PDF page where the chapter opens."""
    starts: dict[int, int] = {}
    for i, text in enumerate(pages, start=1):
        m = re.match(r"\s*Chapter\s*\n?\s*(\d+)\b", text)
        if m:
            starts.setdefault(int(m.group(1)), i)
    return starts


def outline_sections(reader, chapter_start: int) -> list[tuple[int, str, int]]:
    """Numbered top-level sections from the PDF bookmarks.

    Returns (section_index, title, pdf_page). The bookmark tree stores a
    chapter's sections as a list immediately following the chapter entry.
    """
    try:
        top = list(reader.outline)
    except Exception:
        return []

    def dest(node) -> int | None:
        try:
            return reader.get_destination_page_number(node) + 1
        except Exception:
            return None

    for idx, item in enumerate(top):
        if isinstance(item, list) or dest(item) != chapter_start:
            continue
        if idx + 1 >= len(top) or not isinstance(top[idx + 1], list):
            return []
        out = []
        for n, child in enumerate(
            [c for c in top[idx + 1] if not isinstance(c, list)], start=1
        ):
            page = dest(child)
            if page is not None:
                out.append((n, str(child.title).strip(), page))
        return out
    return []


def first_pages(pages: list[str], span: tuple[int, int], pattern: str) -> dict[str, int]:
    """Map captured label -> 1-based PDF page of its first appearance in span."""
    lo, hi = span
    seen: dict[str, int] = {}
    rx = re.compile(pattern, re.M)
    for i in range(lo, hi + 1):
        for m in rx.finditer(pages[i - 1]):
            seen.setdefault(m.group(1), i)
    return seen


def sort_key(label: str) -> list[int]:
    return [int(n) for n in label.split(".")]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--chapter", type=int, required=True)
    ap.add_argument("--pdf", default=None, help="override textbook path")
    ap.add_argument("--offset", type=int, default=None, help="override page calibration")
    ap.add_argument("--dump", action="store_true", help="print the full chapter text")
    args = ap.parse_args()

    logging.getLogger("pypdf").setLevel(logging.ERROR)
    try:
        from pypdf import PdfReader
    except ImportError:
        sys.exit("error: pypdf is required.  pip install pypdf")

    reader = PdfReader(str(find_pdf(args.pdf)))
    pages = [(p.extract_text() or "") for p in reader.pages]

    offset = args.offset if args.offset is not None else detect_offset(pages)

    def printed(pdf_page: int) -> int:
        return pdf_page - offset

    starts = chapter_starts(pages)
    ch = args.chapter
    if ch not in starts:
        sys.exit(f"error: chapter {ch} not found; detected chapters {sorted(starts)}")

    lo = starts[ch]
    later = [p for c, p in starts.items() if p > lo]
    hi = (min(later) - 1) if later else len(pages)
    span = (lo, hi)

    print(
        f"Chapter {ch}  ·  printed pages {printed(lo)}-{printed(hi)}"
        f"  (PDF pages {lo}-{hi}, offset {offset})"
    )
    print(
        f"Cite as: Prince, Understanding Deep Learning (MIT Press, 2024), "
        f"pp. {printed(lo)}-{printed(hi)}."
    )

    sections = outline_sections(reader, lo)
    if sections:
        print("\nSections")
        for n, title, page in sections:
            print(f"  §{ch}.{n:<6} p. {printed(page):>3}   {title}")

    for heading, pattern in (
        ("Subsections", rf"^({ch}\.\d+\.\d+)\b"),
        ("Figures", FIGURE_CAPTION.format(ch=ch)),
        ("Problems", rf"Problem\s+({ch}\.\d+)\b"),
        ("Notebooks", rf"Notebook\s+({ch}\.\d+)\b"),
    ):
        found = first_pages(pages, span, pattern)
        if found:
            note = ""
            if heading == "Problems":
                note = "   (first mention; a page before the end of the chapter is a\n" \
                       "              margin cue marking where to attempt it)"
            print(f"\n{heading}{note}")
            for label in sorted(found, key=sort_key):
                prefix = "§" if heading == "Subsections" else " "
                print(f"  {prefix}{label:<7} p. {printed(found[label]):>3}")

    if args.dump:
        for i in range(lo, hi + 1):
            print(f"\n{'='*24} PRINTED PAGE {printed(i)} (pdf {i}) {'='*24}")
            print(pages[i - 1])


if __name__ == "__main__":
    main()
