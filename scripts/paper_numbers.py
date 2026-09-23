"""Every number quoted in the submitted manuscript, read back from the saved JSON.

Prints each quantity with its source file and, with ``--check``, compares it at
the manuscript's rounding to the value the text reports; the exit status is
non-zero if anything disagrees. This is the audit trail from the tables, Results
section, key points and figure legends to notebooks/figures/.

    python scripts/paper_numbers.py            # print
    python scripts/paper_numbers.py --check    # print and verify
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG = REPO / "notebooks/figures"
COHORTS = ("kirc", "nsclc", "adni")
REP = {"tumor_radiomics": "Tumor radiomics", "organ_radiomics": "Kidney radiomics",
       "tumor_radimagenet": "Tumor embedding", "whole_radimagenet": "Whole-volume embedding"}


def load(rel):
    p = FIG / rel
    return json.loads(p.read_text()) if p.exists() else None


class Audit:
    def __init__(self):
        self.rows = []

    def add(self, label, value, expected, digits=None, source=""):
        """digits: decimals at which the manuscript reports the value (None = exact)."""
        self.rows.append((label, value, expected, digits, source))

    def section(self, title):
        self.rows.append((title, None, None, None, None))

    def report(self, check):
        bad = 0
        for label, value, expected, digits, source in self.rows:
            if source is None:
                print(f"\n{label}\n{'-' * len(label)}")
                continue
            if value is None:
                print(f"  {label:58s} {'(not computed)':>14s}   expected {expected}   [{source}]")
                if check:
                    bad += 1
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and digits is not None:
                shown = f"{float(value):.{digits}f}"
                ok = round(float(value), digits) == round(float(expected), digits)
            else:
                shown = str(value)
                ok = value == expected
            flag = "" if ok else "   <-- MISMATCH"
            bad += (not ok)
            print(f"  {label:58s} {shown:>14s}   expected {expected}{flag}   [{source}]")
        print()
        if check:
            print("all numbers agree with the manuscript" if bad == 0
                  else f"{bad} quantity(ies) disagree with the manuscript")
        return bad


def main(check=False):
    a = Audit()
    summ = {r["cohort"]: r for r in load("attainable_summary.json")}
    shape = load("shape_sensitivity.json")
    stress = load("bound_stress_test.json")
    sweep = load("per_gene_auc_sweep.json")
    targets = load("kirc_target_classes.json")["targets"]
    rep = load("kirc_representations/summary.json")
    anchor = load("anchor_saturation.json")
    adni_stats = load("adni/gene_expression/fastsurfer/stats.json")
    nsclc_stats = load("nsclc/gene_expression/organ_radiomics_subset/stats.json")
    demo = load("cohort_demographics.json")

    # ---------------- Table 1 / study population ----------------
    a.section("Table 1 and 'Study population' (cohort_demographics.json, kirc_representations/summary.json)")
    a.add("participants total", sum(summ[c]["n"] for c in COHORTS), 1021, source="attainable_summary.json")
    for c, n in zip(COHORTS, (190, 129, 702)):
        a.add(f"{c} participants", summ[c]["n"], n, source="attainable_summary.json")
    a.add("KIRC women / men", f"{rep['cohort']['n_female']} / {rep['cohort']['n_male']}", "65 / 125",
          source="kirc_representations/summary.json")
    a.add("KIRC with mutation calls", targets["mut_VHL"]["n"], 131, source="kirc_target_classes.json")
    a.add("KIRC with known stage", rep["cohort"]["n_stage_known"], 154, source="kirc_representations/summary.json")
    a.add("KIRC stage III-IV", rep["cohort"]["n_stage_III_IV"], 57, source="kirc_representations/summary.json")
    if demo:
        k, d = demo["kirc"], demo["adni"]
        a.add("KIRC age mean", k["age"]["mean"], 59.9, 1, "cohort_demographics.json")
        a.add("KIRC age SD", k["age"]["sd"], 12.2, 1, "cohort_demographics.json")
        a.add("KIRC age range", f"{k['age']['min']:.0f}-{k['age']['max']:.0f}", "26-88", source="cohort_demographics.json")
        a.add("KIRC women age mean / SD", f"{k['age_female']['mean']:.1f} / {k['age_female']['sd']:.1f}", "62.9 / 11.5",
              source="cohort_demographics.json")
        a.add("KIRC men age mean / SD", f"{k['age_male']['mean']:.1f} / {k['age_male']['sd']:.1f}", "58.3 / 12.3",
              source="cohort_demographics.json")
        a.add("ADNI women / men", f"{d['n_female']} / {d['n_male']}", "325 / 377", source="cohort_demographics.json")
        a.add("ADNI age mean", d["age"]["mean"], 74.5, 1, "cohort_demographics.json")
        a.add("ADNI age SD", d["age"]["sd"], 7.7, 1, "cohort_demographics.json")
        a.add("ADNI age range", f"{d['age']['min']:.0f}-{d['age']['max']:.0f}", "55-94", source="cohort_demographics.json")
        a.add("ADNI women age mean / SD", f"{d['age_female']['mean']:.1f} / {d['age_female']['sd']:.1f}", "73.6 / 8.0",
              source="cohort_demographics.json")
        a.add("ADNI men age mean / SD", f"{d['age_male']['mean']:.1f} / {d['age_male']['sd']:.1f}", "75.3 / 7.3",
              source="cohort_demographics.json")
        a.add("ADNI collection years", "-".join(map(str, d["collection_years"])), "2010-2013", source="cohort_demographics.json")
        t = demo["nsclc"]["release_tranche"]
        a.add("NSCLC release tranches (early / late)", f"{t['n_early']} / {t['n_late']}", "98 / 31",
              source="cohort_demographics.json")
    else:
        for lbl in ("KIRC age mean", "ADNI women / men", "ADNI age mean", "NSCLC release tranches"):
            a.add(lbl, None, "see Table 1", source="run scripts/cohort_demographics.py (needs data/)")

    # ---------------- Table 2 ----------------
    a.section("Table 2 (attainable_summary.json, shape_sensitivity.json)")
    # KIRC null median is 0.3947 bits, i.e. 0.39 at two decimals (a draft table read 0.40)
    exp = {"kirc": (0.152, 112, 0.370, 0.65, 0.39, 0.76, 1006),
           "nsclc": (0.151, 112, 0.497, 0.52, 0.60, 0.18, 1010),
           "adni": (0.177, 93, 0.492, 0.58, 0.10, 1.20, 835)}
    for c in COHORTS:
        s, e = summ[c], exp[c]
        a.add(f"{c} fitted leading signal", s["rho2_fit"], e[0], 3, "attainable_summary.json:rho2_fit")
        a.add(f"{c} learning cost (patients)", s["nu_fit"], e[1], 0, "attainable_summary.json:nu_fit")
        a.add(f"{c} leading signal calibration limit", s["rho2_ucl95"], e[2], 3, "attainable_summary.json:rho2_ucl95")
        a.add(f"{c} observed information statistic (bits)", shape[c]["observed_bits"], e[3], 2,
              "shape_sensitivity.json:observed_bits")
        a.add(f"{c} permutation-null median (bits)", s["I_total_null_median_bits"], e[4], 2,
              "attainable_summary.json:I_total_null_median_bits")
        a.add(f"{c} calibrated total limit (bits)", shape[c]["true_bound_bits"], e[5], 2,
              "shape_sensitivity.json:true_bound_bits")
        a.add(f"{c} training size for 90% benchmark", s["n_for_90pct_fit"], e[6], 0,
              "attainable_summary.json:n_for_90pct_fit")
        a.add(f"{c} imaging components", s["d_star"], 20, source="attainable_summary.json:d_star")

    # ---------------- stress test ----------------
    a.section("Figure 1 / stress test (bound_stress_test.json)")
    a.add("configurations", sum(stress[c]["n_cells"] for c in COHORTS), 360, source="bound_stress_test.json:n_cells")
    a.add("crossings of the limit", sum(len(stress[c]["crossings"]) for c in COHORTS), 0,
          source="bound_stress_test.json:crossings")
    for c in COHORTS:
        vals = [v for row in stress[c]["achieved_null_corrected"].values() for v in row.values() if v is not None]
        med = float(np.median(vals))
        a.add(f"{c} median null-corrected information in [0.03, 0.05]", 0.025 <= med < 0.055, True,
              source=f"bound_stress_test.json (median {med:.3f})")

    # ---------------- per-gene sweep ----------------
    a.section("Figure S2 / per-gene sweep (per_gene_auc_sweep.json)")
    for c, med, mx in zip(COHORTS, (0.52, 0.63, 0.54), (0.70, 0.77, 0.88)):
        a.add(f"{c} median per-gene AUC", sweep[c]["summary"]["median"], med, 2, "per_gene_auc_sweep.json:summary.median")
        a.add(f"{c} maximum per-gene AUC", sweep[c]["summary"]["max"], mx, 2, "per_gene_auc_sweep.json:summary.max")
    a.add("kirc best-of-2000 null reference", sweep["kirc"]["null_max_auc_q95"], 0.70, 2,
          "per_gene_auc_sweep.json:null_max_auc_q95")
    a.add("adni adjusted maximum AUC", sweep["adni"]["summary_adjusted"]["max"], 0.61, 2,
          "per_gene_auc_sweep.json:summary_adjusted.max")
    a.add("adni best-of-2000 null reference", sweep["adni"]["null_max_auc_q95"], 0.61, 2,
          "per_gene_auc_sweep.json:null_max_auc_q95")
    a.add("nsclc leading expression component variance (%)", 100 * sweep["nsclc"]["shared_axis"]["pc1_variance_ratio"],
          87, 0, "per_gene_auc_sweep.json:shared_axis.pc1_variance_ratio")
    a.add("nsclc gene labels matching the PC1 split (%)", 100 * sweep["nsclc"]["shared_axis"]["frac_genes_phi_ge_0.5"],
          96, 0, "per_gene_auc_sweep.json:shared_axis.frac_genes_phi_ge_0.5")
    a.add("nsclc tranche-adjusted median AUC", sweep["nsclc"]["summary_adjusted"]["median"], 0.58, 2,
          "per_gene_auc_sweep.json:summary_adjusted.median")

    # ---------------- Table 3, Table S1, Figure 2 ----------------
    a.section("Table 3 and Figure 2 (kirc_target_classes.json)")
    best = {"stage_III_IV": (154, 0.80, .005, "Tumor embedding"),
            "sex_female": (190, 0.86, .005, "Whole-volume embedding"),
            "mut_VHL": (131, 0.62, .080, "Kidney radiomics"),
            "mut_PBRM1": (131, 0.56, .318, "Tumor radiomics"),
            "mut_BAP1": (131, 0.61, .214, "Tumor embedding"),
            "mut_SETD2": (131, 0.58, .388, "Tumor embedding"),
            "age": (190, 0.122, .005, "Kidney radiomics"),
            "angiogenesis": (190, 0.071, .005, "Tumor radiomics"),
            "myeloid_inflammation": (190, 0.034, .010, "Whole-volume embedding"),
            "t_effector": (190, 0.026, .050, "Whole-volume embedding")}
    for key, (n, score, p, rname) in best.items():
        t = targets[key]
        row = t["by_image"][t["best_image"]]
        digits = 3 if t["kind"] in ("continuous", "signature") or key in ("age", "angiogenesis",
                                                                        "myeloid_inflammation", "t_effector") else 2
        a.add(f"{key} n", t["n"], n, source="kirc_target_classes.json:n")
        a.add(f"{key} best score", row["observed"], score, digits, "kirc_target_classes.json:by_image.observed")
        a.add(f"{key} P", row["p"], p, 3, "kirc_target_classes.json:by_image.p")
        a.add(f"{key} representation", REP[t["best_image"]], rname, source="kirc_target_classes.json:best_image")
    for key, nm in zip(("mut_VHL", "mut_PBRM1", "mut_BAP1", "mut_SETD2"), (63, 49, 16, 12)):
        a.add(f"{key} mutated", targets[key]["n_mutated"], nm, source="kirc_target_classes.json:n_mutated")
    for key, adj in zip(("angiogenesis", "myeloid_inflammation", "t_effector"), (0.089, 0.037, 0.034)):
        a.add(f"{key} covariate-adjusted R2", targets[key]["deconf_all_r2"], adj, 3, "kirc_target_classes.json:deconf_all_r2")
    a.add("mutation P values all >= .080", min(targets[k]["by_image"][targets[k]["best_image"]]["p"]
                                                for k in ("mut_VHL", "mut_PBRM1", "mut_BAP1", "mut_SETD2")) >= 0.0795,
          True, source="kirc_target_classes.json")

    a.section("Table S1 (kirc_target_classes.json, every target x representation)")
    s1 = {"stage_III_IV": (0.79, 0.73, 0.80, 0.59), "sex_female": (0.72, 0.73, 0.58, 0.86),
          "mut_VHL": (0.60, 0.62, 0.54, 0.53), "mut_PBRM1": (0.56, 0.53, 0.43, 0.43),
          "mut_BAP1": (0.61, 0.57, 0.61, 0.56), "mut_SETD2": (0.49, 0.53, 0.58, 0.44),
          "age": (0.077, 0.122, -0.021, 0.000), "angiogenesis": (0.071, 0.014, -0.010, 0.014),
          "myeloid_inflammation": (0.027, 0.029, -0.003, 0.034), "t_effector": (-0.001, -0.004, -0.002, 0.026)}
    for key, vals in s1.items():
        digits = 3 if key in ("age", "angiogenesis", "myeloid_inflammation", "t_effector") else 2
        for img, v in zip(REP, vals):
            a.add(f"{key} x {REP[img]}", targets[key]["by_image"][img]["observed"], v, digits,
                  "kirc_target_classes.json:by_image")

    # ---------------- Figure 3 and representation results ----------------
    a.section("Figure 3 and representation results (kirc_representations/summary.json)")
    a.add("working dimension (principal components)", rep["working_dim"], 38, source="summary.json:working_dim")
    a.add("genes", rep["n_hvg"], 2000, source="summary.json:n_hvg")
    a.add("tumor radiomics null-corrected mean variance (%)", 100 * rep["budget"]["tumor_radiomics"]["tau"], 1.74, 2,
          "summary.json:budget.tumor_radiomics.tau")
    a.add("tumor radiomics budget P", rep["budget"]["tumor_radiomics"]["p"], .02, 2, "summary.json:budget.tumor_radiomics.p")
    a.add("whole-volume null-corrected mean variance (%)", 100 * rep["budget"]["whole_radimagenet"]["tau"], 0.05, 2,
          "summary.json:budget.whole_radimagenet.tau")
    a.add("whole-volume budget P", rep["budget"]["whole_radimagenet"]["p"], .31, 2, "summary.json:budget.whole_radimagenet.p")
    b = rep["bounds"]
    a.add("whole-volume leading squared correlation", b["whole_radimagenet"]["R1_cv"], 0.40, 2, "summary.json:bounds.R1_cv")
    a.add("whole-volume adjusted", b["whole_radimagenet"]["deconf_all"]["R1_cv"], 0.027, 3, "summary.json:bounds.deconf_all.R1_cv")
    a.add("whole-volume adjusted P", b["whole_radimagenet"]["deconf_all"]["R1_cv_p"], .36, 2, "summary.json:bounds.deconf_all.R1_cv_p")
    a.add("kidney radiomics leading squared correlation", b["organ_radiomics"]["R1_cv"], 0.121, 3, "summary.json:bounds.R1_cv")
    a.add("kidney radiomics adjusted", b["organ_radiomics"]["deconf_all"]["R1_cv"], 0.161, 3, "summary.json:bounds.deconf_all.R1_cv")
    a.add("kidney radiomics adjusted P", b["organ_radiomics"]["deconf_all"]["R1_cv_p"], .005, 3, "summary.json:bounds.deconf_all.R1_cv_p")
    top = [g["gene"] for g in b["whole_radimagenet"]["deconf_demo"].get("top_genes", [])] if isinstance(
        b["whole_radimagenet"]["deconf_demo"], dict) else []

    # ---------------- Figure S3 / anchor panel ----------------
    a.section("Figure S3 / anchor analysis (anchor_saturation.json)")
    k = anchor["kirc"]
    a.add("best panel genes (any order)", ", ".join(sorted(k["best_panel_genes"])),
          ", ".join(sorted(["EPAS1", "HIF1A", "VEGFA", "SLC17A3"])), source="anchor_saturation.json:best_panel_genes")
    a.add("best panel bits", k["best_panel_bits"], 0.094, 3, "anchor_saturation.json:best_panel_bits")
    a.add("total bits", k["I_total_bits"], 0.118, 3, "anchor_saturation.json:I_total_bits")
    a.add("panel share (%)", 100 * k["best_panel_frac_of_total"], 80, 0, "anchor_saturation.json:best_panel_frac_of_total")
    a.add("selection-aware P", k["best_panel_p_selection_aware"], .010, 3, "anchor_saturation.json:best_panel_p_selection_aware")

    # ---------------- Figure 4 / ADNI ----------------
    a.section("Figure 4 and key points (adni/gene_expression/fastsurfer/stats.json)")
    dc = adni_stats["deconfound_comparison"]
    a.add("confounders", ", ".join(dc["confounders"]), "age, sex, education", source="stats.json:deconfound_comparison")
    a.add("leading squared correlation, raw", dc["raw"]["cv_R"][0], 0.37, 2, "stats.json:deconfound_comparison.raw")
    a.add("leading squared correlation, adjusted", dc["deconfounded"]["cv_R"][0], 0.006, 3, "stats.json:deconfound_comparison.deconfounded")
    a.add("adjusted P", dc["deconfounded"]["cv_perm_p"][0], .48, 2, "stats.json:deconfound_comparison.deconfounded")
    a.add("effective rank raw -> adjusted", f"{dc['raw']['eff_rank']} -> {dc['deconfounded']['eff_rank']}", "2 -> 0",
          source="stats.json:deconfound_comparison")
    lead = [g["gene"] for g in adni_stats["top_gene_loadings"]["a1"]]
    a.add("leading pattern contrasts XIST/TSIX with Y-linked genes",
          "XIST" in lead and "TSIX" in lead and any(g in lead for g in ("RPS4Y1", "DDX3Y", "KDM5D", "UTY")), True,
          source="stats.json:top_gene_loadings.a1")

    # ---------------- NSCLC heterogeneity ----------------
    a.section("NSCLC heterogeneity (nsclc/gene_expression/organ_radiomics_subset/stats.json, nsclc_tranche_heterogeneity.json)")
    a.add("pooled leading score held-out R2", nsclc_stats["heldout_R2_leading_direction"], -0.46, 2,
          "stats.json:heldout_R2_leading_direction")
    tr = load("nsclc_tranche_heterogeneity.json")
    if tr:
        a.add("tranche sizes (early / late)", f"{tr['n_early']} / {tr['n_late']}", "98 / 31", source="nsclc_tranche_heterogeneity.json")
        a.add("|cosine| between leading genomic directions", tr["abs_cosine_leading_genomic"], 0.07, 2,
              "nsclc_tranche_heterogeneity.json")
    else:
        a.add("|cosine| between tranche leading directions", None, 0.07,
              source="run scripts/nsclc_tranche_heterogeneity.py (needs data/)")

    return a.report(check)


if __name__ == "__main__":
    sys.exit(1 if main("--check" in sys.argv) else 0)
