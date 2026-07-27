#!/usr/bin/env bash
# Build the recruiting flyer. Two pdflatex passes are required: the header and
# footer bands are drawn with TikZ "remember picture / overlay", which needs the
# page anchors recorded on the first pass.
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
"$PY" banner.py

for _ in 1 2; do
  pdflatex -interaction=nonstopmode -halt-on-error flyer.tex >build.log 2>&1 \
    || { echo "pdflatex failed; see flyer/build.log"; tail -25 build.log; exit 1; }
done

rm -f flyer.aux flyer.out flyer.log
echo "wrote flyer/flyer.pdf"
