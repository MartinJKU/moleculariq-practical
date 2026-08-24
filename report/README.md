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

**All results are reported on the official `ml-jku/moleculariq-v0.0` splits.**
The generated held-out set appears only in Section 4.4 as a methodological
caveat — it is narrower than the benchmark and its constraint questions are
largely satisfiable by a constant, so it inflated the earlier numbers. The
single-construct (aromatic-ring) model is excluded; the report covers the three
task families only.

Everything still needing a real number is marked with a loud red `\tbd{...}` —
search the built PDF for `[TBD:` and make sure none survive. Outstanding:

1. **Author and supervisor.** `FIRSTNAME LASTNAME`, the e-mail address, and
   `SUPERVISOR NAME` in `main-report.tex` (near the top).
2. **Table 1** — the official pass@3 values currently in the table were read off
   an earlier plot and carry no intervals. Replace them from
   `results/matrix/plots/official/summary.csv`, and add the 95% CIs and
   `p_value_vs_baseline` column. **The constant-control row is still empty.**

This matters more than usual here: several claims in Section 4.1 turn on
differences of a few points, and the report says so explicitly. Filling in the
intervals is what converts them from suggestive to established.

Numbers that are already final: the answer-diversity table (Section 4.3) and
the constraint-set control figures in Section 4.4, both from completed runs;
the `0.530 [0.486, 0.574]` control interval was computed from the actual
`baseline__constraint_val` sample dump.

The `aromatic_ring` group is still defined in `configs/eval_matrix.yaml`. It is
unused by the report — delete that block if you do not want it evaluated.

Re-running after the `data.py` fix (see "Known limitations" in the top-level
README) would change the constraint results; update Sections 4.1, 4.3 and 4.4
if you do that.
