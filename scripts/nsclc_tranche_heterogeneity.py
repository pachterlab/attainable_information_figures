"""NSCLC release-tranche heterogeneity: is the leading image-identifiable genomic
direction the same in the two releases of the cohort?

The NSCLC radiogenomics collection was released in two tranches (98 original
patients, 31 later cases; patient numbers R01-128 and above are the later
release). This script runs the cohort pipeline of ``rgit`` (the one behind the
per-cohort ``stats.json``: CPM + log1p, 2000 highly variable genes selected on
the full cohort, Gaussian-copula marginals, PCA to floor(n / 5) components per
modality, Ledoit--Wolf regularized CCA) separately on each tranche and on the
pooled cohort, and reports

  * the leading five-fold cross-validated recoverability of each fit,
  * the absolute cosine between the two tranches' leading genomic directions in
    standardized gene space (``Preprocessed.to_gene_loadings``), and the same
    quantity in one shared pooled PCA basis as a check.

Needs data/. Writes notebooks/figures/nsclc_tranche_heterogeneity.json.

    python scripts/nsclc_tranche_heterogeneity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from rgit.config import RecoverabilityConfig  # noqa: E402
from rgit.datasets import load_modalities, preprocess  # noqa: E402
from rgit.model import cross_validated_recoverability, fit_recoverability  # noqa: E402

OUT = REPO / "notebooks/figures/nsclc_tranche_heterogeneity.json"
TRANCHE_RULE = "patient number R01-128 and above is the later release"


def is_late(pid: str) -> bool:
    return int(pid.split("-")[1]) >= 128


def analyze(gen, img, cfg):
    """Fit the pipeline on one patient set: (Preprocessed, leading R1_cv, gene-space a_1)."""
    pre = preprocess(gen, img, cfg)
    fit = fit_recoverability(pre.G_model, pre.X_model, n_components=pre.n_components)
    cv = cross_validated_recoverability(pre.G_model, pre.X_model, pre.n_components,
                                        cfg.n_folds, random_state=cfg.seed).mean(0)
    a = pre.to_gene_loadings(fit.genomic_dirs[:, :1])[:, 0]
    return pre, float(cv[0]), a / np.linalg.norm(a), fit


def main():
    cfg = RecoverabilityConfig(
        genomics_h5ad=REPO / "data/nsclc/genomics/gene_expression.h5ad",
        imaging_h5ad=REPO / "data/nsclc/imaging/organ_radiomics_subset.h5ad",
        genomics_data_type="expression", save_figures=False,
        run_sample_complexity_sweep=False, run_downsample_sweep=False,
    )
    gen, img, _ = load_modalities(cfg)          # HVG selection on the full cohort
    pids = np.asarray(gen.obs_names)
    late = np.array([is_late(p) for p in pids])

    res = {"n": int(len(pids)), "n_early": int((~late).sum()), "n_late": int(late.sum()),
           "tranche_rule": TRANCHE_RULE, "n_hvg": int(gen.shape[1]),
           "patients_per_dim": cfg.samples_per_dim, "n_folds": cfg.n_folds}
    dirs = {}
    for name, mask in (("early", ~late), ("late", late), ("pooled", np.ones_like(late))):
        pre, R1, a, fit = analyze(gen[mask].copy(), img[mask].copy(), cfg)
        res[name] = {"n": int(mask.sum()), "working_dim_genomic": int(pre.G_model.shape[1]),
                     "working_dim_imaging": int(pre.X_model.shape[1]),
                     "n_components": int(pre.n_components), "R1_cv": R1,
                     "R1_insample": float(fit.recoverability[0])}
        dirs[name] = a
    res["abs_cosine_leading_genomic"] = float(abs(dirs["early"] @ dirs["late"]))
    res["abs_cosine_early_vs_pooled"] = float(abs(dirs["early"] @ dirs["pooled"]))
    res["abs_cosine_late_vs_pooled"] = float(abs(dirs["late"] @ dirs["pooled"]))

    # check: both tranches fitted inside the shared pooled PCA basis
    pre = preprocess(gen, img, cfg)
    ids = np.asarray(pre.shared)
    late_s = np.array([is_late(p) for p in ids])
    a_e = fit_recoverability(pre.G_model[~late_s], pre.X_model[~late_s], n_components=3).genomic_dirs[:, 0]
    a_l = fit_recoverability(pre.G_model[late_s], pre.X_model[late_s], n_components=3).genomic_dirs[:, 0]
    res["abs_cosine_leading_genomic_shared_basis"] = float(
        abs(a_e @ a_l) / (np.linalg.norm(a_e) * np.linalg.norm(a_l)))

    OUT.write_text(json.dumps(res, indent=2))
    for name in ("early", "late", "pooled"):
        r = res[name]
        print(f"{name:6s} n={r['n']:3d}  dims {r['working_dim_genomic']}/{r['working_dim_imaging']}  "
              f"R1_cv = {r['R1_cv']:.3f}  (in-sample {r['R1_insample']:.2f})")
    print(f"|cos| between leading genomic directions (early vs late): "
          f"{res['abs_cosine_leading_genomic']:.3f} gene space, "
          f"{res['abs_cosine_leading_genomic_shared_basis']:.3f} shared PCA basis")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
