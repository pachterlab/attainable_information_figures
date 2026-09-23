"""Redraw the three journal-submission figures from the saved result JSON.

No model is fitted here. The figures are the same data as
``bound_stress_test.pdf``, ``per_gene_auc_sweep.pdf`` and ``kirc_target_classes.pdf``
(scripts/bound_stress_test.py, scripts/per_gene_auc_sweep.py,
scripts/kirc_target_classes_figure.py) redrawn with the qualified labels used in
the submitted manuscript.

Reads
  notebooks/figures/bound_stress_test.json
  notebooks/figures/per_gene_auc_sweep.json
  notebooks/figures/attainable_summary.json
  notebooks/figures/kirc_target_classes.json
and writes
  notebooks/figures/submission/radiology_bound_stress_test.{pdf,png}   (main Figure 1)
  notebooks/figures/submission/radiology_target_classes.{pdf,png}      (main Figure 2)
  notebooks/figures/submission/radiology_per_gene_auc_sweep.{pdf,png}  (Figure S2)

    python scripts/render_submission_figures.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
SOURCE = REPO / "notebooks/figures"
OUT = SOURCE / "submission"
COHORT_LABEL = {"kirc": "TCGA-KIRC (CT)", "nsclc": "NSCLC (CT)", "adni": "ADNI (MRI)"}
plt.rcParams.update({"pdf.fonttype": 42})


def max_information_given_fourth_moment(S, n, d):
    u = np.sqrt(S / d)
    recovery = u * n / (n + d * (1 - u) / u)
    return -0.5 * d * np.log2(1 - recovery)


def label_panels(fig, axes):
    for ax, label in zip(axes, "abcd"):
        ax.text(-.12, 1.04, label, transform=ax.transAxes, fontweight="bold")


def _save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{stem}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT / stem}.pdf")


# ---------------------------------------------------------------------------
# Figure 1: stress test
# ---------------------------------------------------------------------------
def stress_figure(all_res):
    """Colour encodes the genomics set, marker/linestyle the algorithm."""
    cohorts = list(all_res)
    gcolors = {"HVG 2000": "tab:blue", "HVG 500": "tab:orange",
               "random 300": "tab:green", "anchor panel": "tab:red"}
    astyle = {"ridge": ("o", "-"), "kernel ridge": ("s", "--"),
              "random forest": ("^", "-."), "grad. boosting": ("D", ":"),
              "PLS": ("v", (0, (3, 1, 1, 1)))}

    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.2 * len(cohorts), 4.6),
                             squeeze=False)
    for ax, c in zip(axes[0], cohorts):
        r = all_res[c]
        ng = [int(v) for v in r["n_grid"]]
        get = lambda d, k: d.get(k, d.get(str(k)))  # noqa: E731
        nd = np.geomspace(min(ng) * 0.9, max(ng) * 1.15, 200)
        ax.semilogx(nd, [max_information_given_fourth_moment(r["S_bar"], v, r["d_star"])
                         for v in nd], "k-", lw=3.0, zorder=10,
                    label="calibrated model limit")
        for key, row in r["achieved_null_corrected"].items():
            gname, aname = [t.strip() for t in key.split("|")]
            xs = [v for v in ng if get(row, v) is not None]
            ys = [get(row, v) for v in xs]
            es = [get(r["se"].get(key, r["se"].get(str(key), {})), v) or 0.0
                  for v in xs]
            m, ls = astyle.get(aname, ("o", "-"))
            ax.errorbar(xs, ys, yerr=es, marker=m, ls=ls, ms=3.5, lw=1.0,
                        alpha=0.85, capsize=1.5, elinewidth=0.6,
                        color=gcolors.get(gname, "grey"))
        ax.axhline(0, color="grey", lw=0.8, ls=":")
        ax.set_xlim(min(ng) * 0.88, max(ng) * 1.18)
        top = max_information_given_fourth_moment(r["S_bar"], max(ng) * 1.15,
                                                  r["d_star"])
        ax.set_ylim(-0.03 * top, top * 1.06)
        ax.set_xlabel("training patients $n$")
        ax.set_ylabel("information-scale performance (bits, null-corrected)")
        ax.grid(alpha=0.25)
        ax.text(0.03, 0.955,
                f"{COHORT_LABEL.get(c, c.upper())}, $n$ = {r['n']}\n"
                f"{len(r['achieved_null_corrected'])} configurations "
                f"$\\times$ {len(ng)} sizes\n"
                f"{len(r['crossings'])} crossings "
                f"({r['expected_false_crossings']:.1f} expected by chance)",
                transform=ax.transAxes, fontsize=7.8, va="top",
                bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=3))

    handles = [plt.Line2D([], [], color="k", lw=3.0, label="calibrated model limit")]
    handles += [plt.Line2D([], [], color=v, lw=2.2, label=k)
                for k, v in gcolors.items()]
    handles += [plt.Line2D([], [], color="0.35", marker=m, ls=ls, ms=4, lw=1.0,
                           label=k) for k, (m, ls) in astyle.items()]
    fig.legend(handles=handles, fontsize=8.5, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.16), frameon=False,
               title="black: model limit   |   colour: genomics set   |   "
                     "marker: algorithm", title_fontsize=8.5)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    label_panels(fig, axes[0])
    _save(fig, "radiology_bound_stress_test")


# ---------------------------------------------------------------------------
# Figure S2: per-gene AUC sweep
# ---------------------------------------------------------------------------
def pergene_figure(all_res):
    ceilings = {r["cohort"]: r for r in json.load(open(SOURCE / "attainable_summary.json"))}
    cohorts = [c for c in ("kirc", "nsclc", "adni") if c in all_res]
    bins = np.linspace(0.3, 0.9, 61)
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.0 * len(cohorts), 3.9),
                             squeeze=False, sharey=True)
    for ax, c in zip(axes[0], cohorts):
        r = all_res[c]
        auc = np.asarray(r["auc"], float)
        auc = auc[np.isfinite(auc)]
        null = np.asarray(r["null_hist"], float) / r["n_perm"]
        ax.bar(0.5 * (bins[1:] + bins[:-1]), null, width=np.diff(bins), color="0.72",
               label="permutation null", zorder=1)
        ax.hist(auc, bins=bins, color="tab:blue", alpha=0.85,
                label="observed, 2000 HVGs", zorder=2)
        if "auc_adjusted" in r:
            adj = np.asarray(r["auc_adjusted"], float)
            ax.hist(adj[np.isfinite(adj)], bins=bins, histtype="step", lw=1.4,
                    color="tab:orange", zorder=3, label="covariate-adjusted")
        ax.axvline(0.5, color="0.3", ls=":", lw=1.0, zorder=4)
        ax.axvline(r["null_max_auc_q95"], color="0.3", ls="--", lw=1.0, zorder=4,
                   label="best of 2000 under the null (95th pct)")
        ce = ceilings[c]
        ax.axvline(ce["auc_ceiling_ucl"], color="k", lw=2.2, zorder=5,
                   label=r"Gaussian AUC benchmark at $\bar\rho_1^2$ (calibrated)")
        ax.axvline(ce["auc_ceiling_fit"], color="k", lw=1.2, ls="-.", zorder=5,
                   label=r"Gaussian AUC benchmark at $\hat\rho_1^2$ (fit)")
        s = r["summary"]
        t = s["top_genes"][0]
        ax.annotate(t["gene"], xy=(t["auc"], 0), xytext=(-6, 22), textcoords="offset points",
                    ha="right", fontsize=7, style="italic",
                    arrowprops=dict(arrowstyle="-", color="0.4", lw=0.6))
        sa = r["shared_axis"]
        ax.plot([sa["auc_pc1_label"]], [0], marker="^", color="tab:red", ms=7,
                clip_on=False, zorder=6, ls="none",
                label="leading expression axis, binarized")
        lines = [f"{COHORT_LABEL[c]}, $n$ = {r['n']}",
                 f"median AUC {s['median']:.2f}, max {s['max']:.2f}",
                 f"{s['n_above_null_q95']} of {r['n_genes']} genes above null 95th pct "
                 f"({s['expected_above_null_q95']:.0f} expected)"]
        if "summary_adjusted" in r:
            sa2 = r["summary_adjusted"]
            lines.append(f"adjusted for {', '.join(r['adjusted_for'])}: "
                         f"median {sa2['median']:.2f}, max {sa2['max']:.2f}")
        ax.text(0.03, 0.96, "\n".join(lines), transform=ax.transAxes, fontsize=7.2,
                va="top", bbox=dict(fc="white", ec="0.8", alpha=0.9, pad=3))
        ax.set_xlim(0.3, 0.9)
        ax.set_xlabel("cross-validated AUC, median-binarized expression")
        ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("genes")
    h, l = axes[0][-1].get_legend_handles_labels()
    fig.legend(h, l, fontsize=8, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), frameon=False)
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    label_panels(fig, axes[0])
    _save(fig, "radiology_per_gene_auc_sweep")


# ---------------------------------------------------------------------------
# Figure 2: KIRC target classes (best representation per target)
# ---------------------------------------------------------------------------
CLINICAL_RC = {"font.family": "DejaVu Sans", "font.size": 10,
               "axes.spines.top": False, "axes.spines.right": False,
               "pdf.fonttype": 42}


def clinical_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.7),
                             gridspec_kw={"width_ratios": [1, 1.15]})
    fig.subplots_adjust(left=.12, right=.98, bottom=.23, top=.88, wspace=.9)
    binary = ["stage_III_IV", "sex_female", "mut_VHL", "mut_PBRM1", "mut_BAP1", "mut_SETD2"]
    labels = ["Stage III–IV", "Sex", "VHL", "PBRM1", "BAP1", "SETD2"]
    ax = axes[0]
    for y, key in enumerate(binary):
        row = data[key]["by_image"][data[key]["best_image"]]
        ax.barh(y, row["null_q95"] - .5, left=.5, height=.16, color=".75",
                label="Null 95th percentile" if y == 0 else None)
        ax.scatter(row["observed"], y, s=40, zorder=3,
                   facecolor="#2563a6" if row["p"] < .05 else "white",
                   edgecolor="#2563a6" if row["p"] < .05 else ".3")
        ax.annotate(f'{row["observed"]:.2f}', (row["observed"], y),
                    xytext=(8, 0), textcoords="offset points", va="center")
    ax.set(yticks=np.arange(len(binary)), yticklabels=labels, xlim=(.49, .94),
           xlabel="Area under ROC curve", title="a  Binary targets")
    ax.invert_yaxis()
    ax.set_xticks([.5, .6, .7, .8, .9])
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(-.05, -.16), frameon=False, fontsize=9)
    ax = axes[1]
    continuous = ["age", "angiogenesis", "myeloid_inflammation", "t_effector"]
    for y, key in enumerate(continuous):
        row = data[key]["by_image"][data[key]["best_image"]]
        adjusted = data[key].get("deconf_all_r2")
        offset = -.15 if adjusted is not None else 0
        ax.barh(y + offset, row["observed"], height=.27, color="#2563a6",
                label="Unadjusted" if y == 0 else None)
        ax.annotate(f'{row["observed"]:.3f}', (row["observed"], y + offset),
                    xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
        ax.plot([row["null_q95"]] * 2, [y - .3, y + .3], color=".2", lw=1)
        if adjusted is not None:
            ax.barh(y + .15, adjusted, height=.27, color="#9dc3ee",
                    label="Covariate adjusted" if key == "angiogenesis" else None)
            ax.annotate(f"{adjusted:.3f}", (adjusted, y + .15),
                        xytext=(4, 0), textcoords="offset points", va="center", fontsize=9)
    ax.set(yticks=np.arange(4),
           yticklabels=["Age", "Angiogenesis", "Myeloid\ninflammation", "T-effector"],
           xlim=(-.008, .16), xlabel=r"Prediction $R^2$", title="b  Continuous targets")
    ax.invert_yaxis()
    ax.set_xticks([0, .05, .10, .15])
    ax.grid(axis="x", alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(-.12, -.16), frameon=False, fontsize=9)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "radiology_target_classes.pdf")
    fig.savefig(OUT / "radiology_target_classes.png", dpi=220)
    plt.close(fig)
    print(f"wrote {OUT / 'radiology_target_classes.pdf'}")


if __name__ == "__main__":
    stress_figure(json.loads((SOURCE / "bound_stress_test.json").read_text()))
    pergene_figure(json.loads((SOURCE / "per_gene_auc_sweep.json").read_text()))
    with plt.rc_context(CLINICAL_RC):
        clinical_figure(json.loads((SOURCE / "kirc_target_classes.json").read_text())["targets"])
