"""Demographics of the analyzed participants (manuscript Table 1) and the NSCLC
release-tranche split.

Uses exactly the participant sets the analyses use (patients present in both
the genomics and the imaging matrix of each cohort, see
scripts/per_gene_auc_sweep.py) and the same demographic parsing as the covariate
adjustment:

  * TCGA-KIRC: age at index and gender from data/tcga_kirc/clinical_tcga.tsv
  * NSCLC:     age at study and sex from the TCIA series digest
               data/nsclc/imaging/metadata.xlsx (first CT series per patient);
               release tranche from the patient number (R01-128 and above is
               the later release)
  * ADNI:      collection year minus birth year and sex from
               data/adni/demographics.csv (first demographic record per PTID)

Needs data/. Writes notebooks/figures/cohort_demographics.json.

    python scripts/cohort_demographics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
from per_gene_auc_sweep import patient_ids as load_ids  # noqa: E402

OUT = REPO / "notebooks/figures/cohort_demographics.json"


def summarize(age: pd.Series, female: pd.Series) -> dict:
    """Mean, SD, range of age overall and by sex; counts of women and men."""
    age = pd.to_numeric(age, errors="coerce")
    female = female.astype(bool)

    def stats(a):
        a = a.dropna()
        return {"n": int(a.size), "mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                "min": float(a.min()), "max": float(a.max())}

    return {
        "n": int(len(age)),
        "n_female": int(female.sum()),
        "n_male": int((~female).sum()),
        "n_age_missing": int(age.isna().sum()),
        "age": stats(age),
        "age_female": stats(age[female]),
        "age_male": stats(age[~female]),
    }


def kirc() -> dict:
    pids, _ = load_ids("kirc")
    cols = ["cases.submitter_id", "demographic.age_at_index", "demographic.gender"]
    clin = (pd.read_csv(REPO / "data/tcga_kirc/clinical_tcga.tsv", sep="\t",
                        usecols=cols, low_memory=False)
            .replace("'--", np.nan).drop_duplicates("cases.submitter_id")
            .set_index("cases.submitter_id").reindex(pids))
    out = summarize(clin["demographic.age_at_index"], clin["demographic.gender"] == "female")
    out["age_definition"] = "age at index (clinical_tcga.tsv)"
    return out


def nsclc() -> dict:
    pids, _ = load_ids("nsclc")
    digest = pd.read_excel(REPO / "data/nsclc/imaging/metadata.xlsx")
    digest = digest[digest["Modality"] == "CT"].drop_duplicates("Patient ID").set_index("Patient ID")
    digest = digest.reindex(pids)
    age = digest["Patient Age"].astype(str).str.extract(r"(\d+)")[0].astype(float)
    age[digest["Patient Age"].isna()] = np.nan
    out = summarize(age, digest["Patient Sex"] == "F")
    out["n_sex_missing"] = int(digest["Patient Sex"].isna().sum())
    out["age_definition"] = "patient age at CT study (TCIA digest)"
    late = np.array([int(p.split("-")[1]) >= 128 for p in pids])
    out["release_tranche"] = {"rule": "patient number R01-128 and above is the later release",
                              "n_early": int((~late).sum()), "n_late": int(late.sum())}
    return out


def adni() -> dict:
    pids, ycoll = load_ids("adni")
    demo = (pd.read_csv(REPO / "data/adni/demographics.csv", low_memory=False)
            .drop_duplicates("PTID").set_index("PTID"))
    yob = pd.to_datetime(demo["PTDOBYY"], errors="coerce").dt.year
    yob = yob.fillna(pd.to_numeric(demo["PTDOBYY"], errors="coerce"))
    yc = ycoll.fillna(2011.0).values if ycoll is not None else np.full(len(pids), 2011.0)
    age = pd.Series(yc - yob.reindex(pids).values, index=pids)
    sex = pd.to_numeric(demo["PTGENDER"], errors="coerce").reindex(pids)
    out = summarize(age, sex == 2)  # ADNI coding: 1 = male, 2 = female
    out["n_sex_missing"] = int(sex.isna().sum())
    out["age_definition"] = "expression collection year minus birth year (approximate)"
    yc_known = pd.Series(ycoll.values if ycoll is not None else [], dtype=float).dropna()
    out["collection_years"] = [int(yc_known.min()), int(yc_known.max())] if len(yc_known) else None
    return out


def main():
    res = {"kirc": kirc(), "nsclc": nsclc(), "adni": adni()}
    res["n_total"] = sum(res[c]["n"] for c in ("kirc", "nsclc", "adni"))
    OUT.write_text(json.dumps(res, indent=2))
    for c in ("kirc", "nsclc", "adni"):
        r = res[c]
        print(f"{c:6s} n={r['n']:4d}  women/men {r['n_female']}/{r['n_male']}  "
              f"age {r['age']['mean']:.1f} +/- {r['age']['sd']:.1f} "
              f"({r['age']['min']:.0f}-{r['age']['max']:.0f}), "
              f"{r['n_age_missing']} missing")
    print(f"total n={res['n_total']}\nwrote {OUT}")


if __name__ == "__main__":
    main()
