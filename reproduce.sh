#!/usr/bin/env bash
# Reproduce the figures and numbers of the manuscript.
#
#   ./reproduce.sh            redraw every manuscript figure from the saved result JSON
#                             (minutes, CPU only, no data needed) and assemble them under
#                             paper_figures/ with the file names the manuscript includes
#   ./reproduce.sh --check    also verify every number quoted in the manuscript against
#                             the JSON (scripts/paper_numbers.py)
#   ./reproduce.sh --full     recompute every result JSON from the processed cohort
#                             matrices first (needs data/, see README; hours), then redraw
#
# Run from the repository root after `pip install -e .` (see README, "Setup").
set -euo pipefail
cd "$(dirname "$0")"

FULL=0; CHECK=0
for a in "$@"; do
  case "$a" in
    --full) FULL=1 ;;
    --check) CHECK=1 ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done

if [ "$FULL" = 1 ]; then
  echo "### full recompute from data/ (run order matters: later scripts read earlier JSON)"
  python scripts/attainable_bound_simulation.py
  python scripts/attainable_bound_cohorts.py kirc nsclc adni
  python scripts/channel_ucl.py kirc nsclc adni
  python scripts/total_information_ucl.py kirc nsclc adni
  python scripts/shape_sensitivity.py kirc nsclc adni
  python scripts/anchor_gene_saturation.py nsclc kirc adni
  python scripts/attainable_summary.py
  python scripts/per_gene_auc_sweep.py kirc nsclc adni
  python scripts/bound_stress_test.py kirc nsclc adni
  python scripts/kirc_target_classes.py
  python scripts/build_kirc_representation_notebook.py
  jupyter nbconvert --execute --inplace --ExecutePreprocessor.timeout=-1 notebooks/kirc_representation_bound.ipynb
  python scripts/kirc_channel_ucl_by_dimension.py
  python scripts/kirc_per_gene_budget.py
  python scripts/cohort_demographics.py
  python scripts/nsclc_tranche_heterogeneity.py
  # the per-cohort stats.json (Figure 4, ADNI deconfounding) come from the notebook pipeline:
  rgit-recoverability --genomics data/adni/genomics/gene_expression.h5ad \
      --imaging data/adni/imaging/fastsurfer_gex_pts.h5ad --genomics-data-type expression \
      --demographics-csv data/adni/demographics.csv \
      --output-dir notebooks/figures/adni/gene_expression/fastsurfer
fi

echo "### redraw from saved JSON"
python scripts/attainable_bound_simulation.py --replot      # Figure S1
python scripts/anchor_gene_saturation.py --replot kirc      # Figure S3
python scripts/bound_stress_test.py --replot                # source of Figure 1
python scripts/per_gene_auc_sweep.py --replot               # source of Figure S2
python scripts/attainable_summary.py                        # Table 2 inputs, attainable_summary.{pdf,json}
python scripts/deconfound_figure.py adni                    # Figure 4
python scripts/kirc_target_classes_figure.py                # Figure 3, Table S1, source of Figure 2
python scripts/render_submission_figures.py                 # Figures 1, 2, S2 with the manuscript's labels

echo "### assemble paper_figures/"
F=notebooks/figures
mkdir -p paper_figures
cp "$F/submission/radiology_bound_stress_test.pdf"   paper_figures/   # Figure 1
cp "$F/submission/radiology_target_classes.pdf"      paper_figures/   # Figure 2
cp "$F/kirc_representations.pdf"                     paper_figures/   # Figure 3
cp "$F/adni/gene_expression/fastsurfer/deconfound_comparison.pdf" paper_figures/adni_deconfound.pdf  # Figure 4
cp "$F/synthetic/attainable_bound.pdf"               paper_figures/sim_attainable_bound.pdf          # Figure S1
cp "$F/submission/radiology_per_gene_auc_sweep.pdf"  paper_figures/   # Figure S2
cp "$F/tcga_kirc/gene_expression/organ_radiomics/anchor_saturation.pdf" paper_figures/kirc_anchor_saturation.pdf  # Figure S3
ls -1 paper_figures

if [ "$CHECK" = 1 ]; then
  echo "### numbers quoted in the manuscript"
  python scripts/paper_numbers.py --check
fi
