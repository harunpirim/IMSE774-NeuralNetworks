---
name: lecture-note
description: Turns exported textbook highlights and figure screenshots into a publish-ready Quarto lecture note for the IMSE 774 neural networks course site, with paraphrased prose, verified printed-page citations, generated code figures, and an optional render-and-push. Use when the user supplies Understanding Deep Learning highlights or a Notebook Export, attaches textbook figure images, asks to write or fill in a note under notes/, or asks to update the course website.
---

# Lecture Note

Builds one week's note in `notes/NN-slug.qmd` from the user's reading
highlights, then renders, commits, and pushes so the live site updates.

The textbook is Prince, *Understanding Deep Learning* (MIT Press, 2024).
Site: <https://harunpirim.github.io/IMSE774-NeuralNetworks/>

## Inputs

The user supplies some or all of:

- **A highlights export** — Apple Books or Kindle "Notebook Export" HTML,
  usually attached from `~/Library/Containers/com.apple.mail/...`. That path is
  sandboxed: open it with the **Read** tool, not `cat`/`ls`, which get
  `Operation not permitted`.
- **Figure screenshots** pasted into chat, one per textbook figure.
- Sometimes just "chapter 5" with no highlights, in which case work from the
  PDF alone.

## Workflow

```
- [ ] 1  Read highlights, identify the chapter and target .qmd
- [ ] 2  Run udl_locate.py for verified printed page numbers
- [ ] 3  Identify every attached image, then file them
- [ ] 4  Write the note
- [ ] 5  Render and verify
- [ ] 6  Commit, push, confirm CI
```

### 1. Read the highlights

Note the chapter and which sections the highlights actually cover. Match the
chapter to its note via `schedule.qmd` and the sidebar in `_quarto.yml`; the
files are already scaffolded by `scripts/new_note.py`, so **edit the existing
`notes/NN-slug.qmd` rather than creating a file.**

Highlights are raw material, not the note. See "Rewriting highlights" below.

If the highlights cover only part of the note's assigned reading, write the
covered chapter thoroughly and tell the user which part is thin, rather than
silently padding it.

### 2. Get verified page numbers

Highlight exports report *ebook* locations and pages. These do not match the
printed book. Always resolve real page numbers:

```bash
python .cursor/skills/lecture-note/scripts/udl_locate.py --chapter 3
python .cursor/skills/lecture-note/scripts/udl_locate.py --chapter 3 --dump
```

It prints the chapter's printed page range, numbered sections, subsections,
figure pages, problem numbers, and notebook cues. `--dump` prints the full
chapter text — use it to check any paraphrase or claim against the source.
Takes about 20 seconds; the textbook PDF is gitignored, so it must be present
locally at the repository root.

### 3. Identify and file the figures

**Read every attached image before using it.** Do not infer which figure is
which from attachment order alone. Expect three kinds of junk in the batch:
caption-only screenshots with no graphic, duplicates of the same figure, and
screenshots holding two consecutive figures at once.

Copy the keepers to `notes/images/chNN/` named
`fig-N-MM-short-slug.png` (zero-padded, e.g. `fig-1-02-regression-classification.jpg`).
Keep the original extension. A screenshot containing two figures keeps both
numbers: `fig-1-07-1-08-inpainting-text-completion.jpg`.

### 4. Write the note

Follow [house-style.md](house-style.md) for the template, prose conventions,
and the exact figure/citation syntax. The essentials:

- Every `##` section opens with an italic reference line: `*Prince §1.1, pp. 1–7.*`
- Textbook figures are captioned `Prince (2024), figure 1.3, p. 4.` and are
  **reproduced unaltered** — the book is CC-BY-NC-ND, which permits verbatim
  non-commercial redistribution with attribution but forbids derivatives, so
  never crop or recolor them.
- The note ends with a **Textbook Reference Map** table covering every section,
  figure, and equation, plus a figure-credits callout.

### 5. Render and verify

```bash
quarto render notes/NN-slug.qmd --to html
```

Run this **outside the sandbox** (`required_permissions: ["all"]`) — torch
segfaults under the sandbox. The interpreter with the dependencies is
`/opt/homebrew/Caskroom/miniconda/base/bin/python`.

Then, before moving on:

- **Fix every crossref warning.** `Unable to resolve crossref @fig-x` usually
  means a panel letter got absorbed into the label. Write `@fig-udl-1-2**d**`,
  not `@fig-udl-1-2d`.
- **Read the generated figure PNGs** under
  `_site/notes/NN-slug_files/figure-html/` and confirm each one shows what its
  caption claims.
- **Check printed cell output against the prose.** Any number stated in a
  caption or paragraph must match what the code actually produced. Iterative
  demos in particular must converge before the text claims they agree with a
  closed form.
- Render `--to pdf` too; both formats ship.

### 6. Publish

Commit the `.qmd` and `notes/images/chNN/`. Commit style is a short
sentence-case imperative subject with no prefix, e.g. *"Write the Week 3
lecture note on shallow neural networks"*. Push to `main`, then confirm the
build:

```bash
gh run watch <run-id> --exit-status --interval 20
```

CI re-executes every code cell from a clean checkout because `_freeze/` is
gitignored, so a cell that only works locally will fail there.

## Rewriting highlights

The highlights mark what the user found worth keeping. They are **not** quotes
to be pasted in sequence. Turn them into continuous prose in the note's own
voice:

- **Paraphrase by default.** Reserve direct quotation for a handful of lines
  genuinely better in Prince's words — a crisp definition, a memorable aside —
  and attribute each with a page number.
- **Restore the connective tissue.** Highlights are disjoint fragments; the
  note needs the argument that runs between them.
- **Reorder freely** to follow the chapter's logic rather than highlight order.
- **Fill the gaps.** A highlighted subsection that skips a definition still
  needs that definition for the note to stand alone.
- **Ignore highlights that carry no content** — empty `noteText` entries, or
  fragments like "women should be paid less than men" that are half a sentence.
  Go back to the source with `--dump` and use the whole idea.

## Adding value beyond the book

Prince has almost no code. Each note should add two to five executed cells that
teach something the prose cannot. Good candidates:

- **Make an abstract number concrete.** Print the input dimensionality of each
  modality the chapter discusses; show why a dense layer is infeasible.
- **Visualize the mechanism.** Decompose an equation stage by stage.
- **Show the failure mode**, not just the success — an unconverged fit, an
  agent that stops exploring.
- **Reproduce a book figure in the course's own style**, which lets you label
  it as ours and cite the original as the inspiration.
- **Verify a claim numerically**, e.g. gradient descent against a closed form.

Keep cells fast and deterministic (`np.random.seed(774)`, `torch.manual_seed(774)`
are already in the setup cell). Prefer vectorized code over long Python loops so
CI stays quick.

Exercises follow the same principle. Where the chapter has problems, cite them
as `**(UDL 3.1)**`; where it does not (Chapter 1 has none), write your own.
Frame at least one or two around industrial and manufacturing examples, since
this is an IMSE course.

## Invoking

`/lecture-note` with the highlights and figures attached. The skill also
triggers on requests like "write the week 4 note from these highlights" or
"update the course website".

If the user says only "update the website" with no new material, do not invent
content: ask which note, or offer to re-render and push what already exists.

## Reference

- [house-style.md](house-style.md) — note template, prose conventions, figure and citation syntax, repository layout.
