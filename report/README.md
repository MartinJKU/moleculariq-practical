# Report

Practical-work report, built on the [JKU LaTeX report
template](https://github.com/michaelroland/jku-templates-report-latex) by
Michael Roland (MPL-2.0 — see `LICENSE.template`). The template's `.sty`,
fonts and logos are vendored here so the report builds from a clean checkout
with no network access.

## Build

**Linux / macOS:**

```bash
./build.sh          # syncs figures, then runs latexmk -xelatex + biber
./build.sh clean    # remove build artefacts
```

Needs XeLaTeX and Biber. On Debian/Ubuntu:

```bash
apt-get install texlive-xetex texlive-latex-extra texlive-fonts-recommended \
                texlive-fonts-extra texlive-bibtex-extra texlive-plain-generic \
                texlive-lang-german biber latexmk
```

**Windows (PowerShell):**

```powershell
.\build.ps1          # syncs figures, then xelatex -> biber -> xelatex x2
.\build.ps1 clean
```

Install [MiKTeX](https://miktex.org/download) and allow it to fetch missing
packages on first run. `build.ps1` calls `xelatex` and `biber` directly instead
of `latexmk`, because latexmk needs a Perl installation that MiKTeX does not
ship — the four-pass sequence is what latexmk would run anyway.

If PowerShell blocks the script, either allow it for the current session with
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, or run the four
commands by hand.

**The build must use XeLaTeX, not pdfLaTeX** — the JKU template loads the
vendored TTF fonts in `fonts/`, which pdfLaTeX cannot handle. On Overleaf set
this under Menu → Compiler → XeLaTeX.

> Note: `report/figures/*.pdf` is git-ignored, so a `git clone`/`git pull` gives
> you the sources but **not** the generated figures. Copy them across from the
> cluster separately (or build there), otherwise the report renders placeholder
> boxes where the plots should be.

## Figures

`build.sh` copies the generated figures out of `../results/matrix/plots/` into
`figures/` under short, underscore-free names (underscores are special in LaTeX
text mode, and copying keeps this folder self-contained for submission):

| Source | Becomes |
|---|---|
| `heldout/bars_avg_accuracy.pdf` | `figures/heldout-bars.pdf` |
| `official/bars_pass_at_3.pdf` | `figures/official-bars.pdf` |
| `official/heatmap_delta_vs_baseline.pdf` | `figures/official-delta.pdf` |
| `aromatic_ring/bars_avg_accuracy.pdf` | `figures/aromatic-bars.pdf` |

Generate them first with:

```bash
./scripts/reproduce.sh report        # writes PDFs under results/matrix/plots/
```

The document compiles **without** them too: any missing figure renders as a
visible placeholder box rather than failing the build, so you can iterate on
the text before the runs finish.

## Before submitting

The report is written around the results that exist so far. Everything still
needing a real number is marked with a loud red `\tbd{...}` in the PDF — search
the built PDF for `[TBD:` and make sure none survive. Currently outstanding:

1. **Author and supervisor.** `FIRSTNAME LASTNAME`, the e-mail address, and
   `SUPERVISOR NAME` in `main-report.tex` (near the top).
2. **Constant-control row** of Table 1 — run `./scripts/reproduce.sh eval`,
   which now includes the `constant` control model, and copy the row from
   `results/matrix/plots/*/summary.md`.
3. **Single-construct study** (Section 4.4) — the aromatic-ring model's numbers
   from `results/matrix/plots/aromatic_ring/summary.md`.

The numbers already in the text (the full matrix in Table 1, the constraint
diagnosis in Section 4.3, the degenerate-answer table) come from completed
runs. Section 4.3's confidence interval and answer-diversity figures were
computed from the actual `baseline__constraint_val` sample dump.

Re-running the pipeline after the fix in `data.py` (see "Known limitations" in
the top-level README) would change the constraint column; if you do that,
update Table 1 and Section 4.3 accordingly.
