.PHONY: help site preview pdf all clean deep-clean notes check flyer

QUARTO ?= quarto

help:
	@echo "IMSE 774 course site"
	@echo ""
	@echo "  make site       Render the website to _site/"
	@echo "  make preview    Live-reloading preview in the browser"
	@echo "  make pdf        Render every page to PDF alongside its HTML"
	@echo "  make all        Website plus PDFs"
	@echo "  make notes      Scaffold any lecture notes that don't exist yet"
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

all: site pdf

flyer:
	cd flyer && ./build.sh

notes:
	python scripts/new_note.py

check:
	python scripts/new_note.py --list

clean:
	rm -rf _site .quarto
	rm -rf notes/*_files notes/*.tex notes/*.log
	rm -rf *_files *.tex *.log

deep-clean: clean
	rm -rf _freeze
