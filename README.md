# IMSE 774 — Neural Networks

Course materials for **IMSE 774: Neural Networks** at North Dakota State
University, Fall 2026.

**Course website:** <https://harunpirim.github.io/IMSE774-NeuralNetworks/>

Lecture notes are written in [Quarto](https://quarto.org) and published as both
a website and downloadable PDFs. Every figure and numerical result in the notes
is produced by code that runs when the site is built, so nothing on the site can
silently drift out of sync with the code that generated it.

| | |
|:------------------|:-----------------------------------------------------|
| **Meeting** | Tuesday & Thursday, 12:30–1:45 p.m. |
| **Term** | August 25 – December 10, 2026 |
| **Instructor** | Harun Pirim, PhD · [harun.pirim@ndsu.edu](mailto:harun.pirim@ndsu.edu) |
| **Textbook** | Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/), MIT Press, 2024 |

---

## Repository Layout

```
.
├── index.qmd               # Course homepage
├── syllabus.qmd            # Syllabus
├── schedule.qmd            # Week-by-week schedule
├── notes/                  # One .qmd per topic
│   ├── _metadata.yml       # Settings shared by all notes
│   ├── 01-introduction.qmd
│   ├── ...
│   └── 15-diffusion-models.qmd
├── notebooks/              # Colab copies of each note's code, generated
├── assignments/            # Colab assignment notebooks (solutions/ is gitignored)
├── assets/
│   ├── css/                # Site theme (SCSS + CSS)
│   └── tex/preamble.tex    # LaTeX preamble for PDF output
├── scripts/
│   ├── new_note.py         # Generates skeleton note files
│   └── qmd_to_notebook.py  # Exports a note's code cells to notebooks/
├── .github/workflows/      # Build and deploy to GitHub Pages
├── _quarto.yml             # Project configuration
├── references.bib          # Bibliography
├── requirements.txt        # Python dependencies
└── Makefile                # Common build commands
```

## Building Locally

You need [Quarto](https://quarto.org/docs/get-started/) and Python 3.11+.

```bash
git clone https://github.com/harunpirim/IMSE774-NeuralNetworks.git
cd IMSE774-NeuralNetworks

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Then:

```bash
make site      # render the website to _site/
make preview   # live-reloading preview in your browser
make pdf       # render every page to PDF in _site/pdf/
make annotate  # wide-margin PDFs for stylus markup, into _annotate/
make notebooks # export each note's code cells to notebooks/ for Colab
make all       # website + PDFs
make clean     # remove build artifacts
```

PDF output additionally requires a LaTeX installation. The simplest option is
Quarto's own:

```bash
quarto install tinytex
```

### Annotating notes on a tablet

`make annotate` renders the notes with the text column narrowed to 5.15in and
the remaining 2.6in of the page left as a dot-ruled band, which is room enough
to write beside any paragraph. Restrict it to one note with `make annotate
NOTE=03`.

Output goes to `_annotate/`, which is gitignored but still inside iCloud Drive,
so the PDFs turn up on an iPad under **Files > iCloud Drive** with nothing to
upload. Open one in GoodNotes, Notability, PDF Expert, or Books and mark it up
with the Apple Pencil. Ink lives in the annotation app, not in the repository,
so re-running `make annotate` never overwrites handwriting held elsewhere —
but it does replace the file in `_annotate/`, so keep marked-up copies in the
annotation app rather than in that folder.

Adjust the ruling in `assets/tex/annotate-preamble.tex` and the page geometry in
`_quarto-annotate.yml`; the two have to agree about where the text column ends.
The dot grid needs `tikz`, and silently falls back to a plain empty margin if
the LaTeX installation lacks it.

### Rendering a single note

```bash
quarto render notes/03-shallow-networks.qmd --to html
quarto render notes/03-shallow-networks.qmd --to pdf
```

## Writing Notes

Every note follows the same structure: a metadata banner, numbered learning
objectives, content sections, a summary, exercises, and further reading. To
scaffold any notes that do not yet exist:

```bash
python scripts/new_note.py          # create missing notes
python scripts/new_note.py --list   # show which notes exist
```

Existing files are never overwritten, so this is safe to re-run at any time.

Each note links to a Colab notebook holding the same code, so students can run
it without installing anything. Regenerate the notebooks after editing a note's
code and commit the result — Colab loads the `.ipynb` straight from GitHub:

```bash
make notebooks                                   # every note
python scripts/qmd_to_notebook.py notes/02-*.qmd  # just one
```

A few conventions worth keeping:

- **Follow the textbook's notation.** $\theta_{d\bullet}$ for parameters into
  the hidden layer, $\phi_\bullet$ for parameters out of it, square brackets for
  function application.
- **Label equations and figures** with `{#eq-name}` and `#| label: fig-name`, and
  reference them as `@eq-name` and `@fig-name`. Cross-references then work in
  both HTML and PDF.
- **Let code produce the figures.** Prefer a generated plot over a static image
  so that the figure updates when the underlying idea does.
- **Verify claims in code** where it is cheap to do so. If the notes assert that
  a slope equals $\theta_{11}\phi_1 + \theta_{31}\phi_3$, compute it.
- **Fold setup blocks** with `#| code-fold: true` to keep the reader's attention
  on the substance.

### Execution caching

`execute: freeze: auto` means a note's code is re-run only when that note
changes. Results live in `_freeze/`, which is gitignored locally and cached in
CI. To force a full re-run:

```bash
make clean && make site
```

## Publishing

Pushing to `main` triggers `.github/workflows/publish.yml`, which renders the
site and PDFs and deploys to GitHub Pages. Enable it once under
**Settings → Pages → Build and deployment → Source: GitHub Actions**.

## What Is Not in This Repository

Deliberately excluded, and enforced by `.gitignore`:

- **The textbook PDF and instructor manual** — copyrighted. The book is free
  from the author at <https://udlbook.github.io/udlbook/>.
- **Student submissions, grades, and exam keys** — protected under FERPA.
- **Previous semesters' slide decks and lecture PDFs** — kept locally.

## License

Course notes and written materials are licensed under
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).
Code samples are licensed under the MIT License. See [LICENSE](LICENSE).

The textbook is © Simon J. D. Prince and MIT Press; these notes follow its
structure and notation but are independent work.
