# Build the report PDF on Windows (PowerShell).
#
#   .\build.ps1          # build main-report.pdf
#   .\build.ps1 clean    # remove build artefacts
#
# Requires XeLaTeX and Biber on PATH. With MiKTeX (https://miktex.org) both are
# included; on first run MiKTeX will prompt to install missing packages -- allow
# it. TeX Live works too.
#
# This deliberately calls xelatex/biber directly rather than latexmk, because
# latexmk needs a Perl install that MiKTeX does not ship. The four-pass sequence
# below is what latexmk would do anyway: first pass writes the .bcf and
# collects labels, biber resolves the bibliography, and the last two passes
# settle citations, the table of contents and cross-references.
#
# If PowerShell refuses to run this ("running scripts is disabled"), either
# unblock it for this session:
#     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# or run the commands at the bottom of this file by hand.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

$jobname = "main-report"

if ($args.Count -gt 0 -and $args[0] -eq "clean") {
    Get-ChildItem -Path . -Include `
        "$jobname.aux", "$jobname.bbl", "$jobname.bcf", "$jobname.blg", `
        "$jobname.fdb_latexmk", "$jobname.fls", "$jobname.lof", "$jobname.log", `
        "$jobname.lot", "$jobname.out", "$jobname.run.xml", "$jobname.toc", `
        "$jobname.xdv", "$jobname.synctex.gz" -File -ErrorAction SilentlyContinue |
        Remove-Item -Force
    Write-Host "cleaned"
    exit 0
}

# --- Sync generated figures ------------------------------------------------
# Copy the pipeline's figures in under short, underscore-free names (underscores
# are special in LaTeX text mode). Missing figures are fine: the document
# renders a visible placeholder box instead of failing.
$plots = Join-Path ".." "results\matrix\plots"
$figureMap = @{
    "heldout\bars_avg_accuracy.pdf"          = "heldout-bars"
    "heldout\heatmap_delta_vs_baseline.pdf"  = "heldout-delta"
    "official\bars_pass_at_3.pdf"            = "official-bars"
    "official\heatmap_delta_vs_baseline.pdf" = "official-delta"
    "official\heatmap_pass_at_3.pdf"         = "official-heatmap"
    "aromatic_ring\bars_avg_accuracy.pdf"    = "aromatic-bars"
}
New-Item -ItemType Directory -Force -Path "figures" | Out-Null
$found = 0
foreach ($src in $figureMap.Keys) {
    $path = Join-Path $plots $src
    if (Test-Path $path) {
        Copy-Item $path (Join-Path "figures" "$($figureMap[$src]).pdf") -Force
        Write-Host "  [fig] $($figureMap[$src]).pdf"
        $found++
    }
}
if ($found -eq 0) {
    Write-Host "  NOTE: no generated figures found under $plots"
    Write-Host "        Placeholders will be used. Copy report\figures\*.pdf from"
    Write-Host "        the cluster, or run ./scripts/reproduce.sh report there first."
}

# --- Compile ---------------------------------------------------------------
function Invoke-Step($exe, $stepArgs, $label) {
    Write-Host "==> $label"
    & $exe @stepArgs
    if ($LASTEXITCODE -ne 0) {
        throw "$label failed (exit $LASTEXITCODE). See $jobname.log for the first error."
    }
}

$xelatexArgs = @("-interaction=nonstopmode", "-halt-on-error", "$jobname.tex")
Invoke-Step "xelatex" $xelatexArgs "xelatex (pass 1/3)"
Invoke-Step "biber"   @($jobname)  "biber (bibliography)"
Invoke-Step "xelatex" $xelatexArgs "xelatex (pass 2/3)"
Invoke-Step "xelatex" $xelatexArgs "xelatex (pass 3/3)"

Write-Host ""
Write-Host "Built: $(Join-Path $PSScriptRoot "$jobname.pdf")"
