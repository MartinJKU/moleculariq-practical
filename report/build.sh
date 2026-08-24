#!/usr/bin/env bash
#
# Build the report PDF.
#
#   ./build.sh          # build main-report.pdf
#   ./build.sh clean    # remove build artefacts
#
# Requires a TeX distribution with XeLaTeX and Biber, e.g. on Debian/Ubuntu:
#   apt-get install texlive-xetex texlive-latex-extra texlive-fonts-recommended \
#                   texlive-bibtex-extra biber latexmk
#
# Figures are included directly from ../results/matrix/plots/, so run
# `./scripts/reproduce.sh report` first to generate them. The document also
# compiles without them: any missing figure renders as a visible placeholder
# box instead of failing the build.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "clean" ]]; then
    latexmk -C main-report >/dev/null 2>&1 || true
    rm -f main-report.bbl main-report.run.xml
    echo "cleaned"
    exit 0
fi

# Sync generated figures into figures/ under short, underscore-free names.
# Underscores are special in LaTeX text mode, and copying (rather than linking
# across directories) keeps report/ self-contained: the folder can be handed in
# or archived on its own and still builds.
PLOTS=../results/matrix/plots
sync_fig() {  # <source-relative-path> <target-basename>
    if [[ -f "$PLOTS/$1" ]]; then
        cp -f "$PLOTS/$1" "figures/$2.pdf"
        echo "  [fig] $2.pdf"
    fi
}
mkdir -p figures
sync_fig heldout/bars_avg_accuracy.pdf            heldout-bars
sync_fig heldout/heatmap_delta_vs_baseline.pdf    heldout-delta
sync_fig official/bars_pass_at_3.pdf              official-bars
sync_fig official/heatmap_delta_vs_baseline.pdf   official-delta
sync_fig official/heatmap_pass_at_3.pdf           official-heatmap
sync_fig aromatic_ring/bars_avg_accuracy.pdf      aromatic-bars

latexmk -xelatex -interaction=nonstopmode -halt-on-error main-report.tex

echo
echo "Built: $(pwd)/main-report.pdf"
if ! ls ../results/matrix/plots/*/*.pdf >/dev/null 2>&1; then
    echo "NOTE: no generated figures found under ../results/matrix/plots/."
    echo "      Placeholders were used. Run ./scripts/reproduce.sh report first."
fi
