.PHONY: help site preview pdf annotate all clean deep-clean notes notebooks check flyer

QUARTO ?= quarto

help:
	@echo "IMSE 774 course site"
	@echo ""
	@echo "  make site       Render the website to _site/"
	@echo "  make preview    Live-reloading preview in the browser"
	@echo "  make pdf        Render every page to PDF alongside its HTML"
	@echo "  make annotate   Stylus-markup PDFs of the notes into _annotate/"
	@echo "  make all        Website plus PDFs"
	@echo "  make notes      Scaffold any lecture notes that don't exist yet"
	@echo "  make notebooks  Export each note's code cells to notebooks/ for Colab"
	@echo "  make check      List which notes exist"
	@echo "  make flyer      Rebuild the one-page recruiting flyer"
	@echo "  make clean      Remove rendered output"
	@echo "  make deep-clean Also drop the execution cache (forces full re-run)"

site:
	$(QUARTO) render --to html

preview:
	$(QUARTO) preview

# Quarto places each PDF next to its HTML inside _site/, which is where the
# "Other Formats" download link on each page points.
pdf:
	@for f in syllabus.qmd schedule.qmd notes/*.qmd; do \
		echo "==> $$f"; \
		$(QUARTO) render "$$f" --to pdf \
			|| echo "    PDF render failed for $$f (continuing)"; \
	done

# Wide-margin, dot-ruled PDFs sized for marking up with a stylus. _annotate/ is
# gitignored but still lives inside iCloud Drive, so the files appear on the
# iPad under Files > iCloud Drive with no upload step; open one in GoodNotes,
# Notability, or Books and write on it. Limit to one note with: make annotate NOTE=03
# stdin is closed because a missing LaTeX package otherwise stalls xelatex on a
# prompt that never gets answered.
NOTE ?=
annotate:
	@for f in notes/$(NOTE)*.qmd; do \
		[ -e "$$f" ] || { echo "No note matches notes/$(NOTE)*.qmd"; exit 1; }; \
		echo "==> $$f"; \
		$(QUARTO) render "$$f" --profile annotate --to pdf < /dev/null \
			|| echo "    annotate render failed for $$f (continuing)"; \
	done
	@echo ""
	@echo "Ready in _annotate/notes/ — on the iPad open Files > iCloud Drive."

all: site pdf

flyer:
	cd flyer && ./build.sh

notes:
	python scripts/new_note.py

# The "Open in Colab" link at the top of each note points at the notebook this
# target writes, so re-run it whenever a note's code changes and commit the
# result -- Colab reads the .ipynb straight out of the GitHub repository.
notebooks:
	python scripts/qmd_to_notebook.py

check:
	python scripts/new_note.py --list

clean:
	rm -rf _site _annotate .quarto
	rm -rf notes/*_files notes/*.tex notes/*.log
	rm -rf *_files *.tex *.log

deep-clean: clean
	rm -rf _freeze
