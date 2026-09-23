"""Compare two trees of saved result JSON (for example a fresh ``reproduce.sh --full``
run against the committed ``notebooks/figures/``) number by number.

Every numeric leaf is compared with a relative tolerance; strings, booleans and
structure must match exactly. Lists of dicts and nested dicts are walked
recursively. Files present in only one tree are reported.

    python scripts/compare_results.py notebooks/figures /path/to/other/notebooks/figures
    python scripts/compare_results.py A B --rtol 1e-6 --only attainable_summary.json shape_sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

DEFAULT_FILES = [
    "attainable_summary.json", "attainable_bound_cohorts.json", "channel_ucl.json",
    "total_information_ucl.json", "shape_sensitivity.json", "anchor_saturation.json",
    "bound_stress_test.json", "per_gene_auc_sweep.json", "kirc_target_classes.json",
    "kirc_representations/summary.json", "kirc_representations/budget.json",
    "kirc_representations/invert2.json", "synthetic/attainable_bound.json",
    "cohort_demographics.json", "nsclc_tranche_heterogeneity.json",
    "adni/gene_expression/fastsurfer/stats.json",
    "nsclc/gene_expression/organ_radiomics_subset/stats.json",
    "tcga_kirc/gene_expression/organ_radiomics/stats.json",
]


def walk(a, b, path, rtol, atol, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "missing in " + ("A" if k not in a else "B")))
            else:
                walk(a[k], b[k], f"{path}.{k}", rtol, atol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((path, f"length {len(a)} vs {len(b)}"))
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{path}[{i}]", rtol, atol, out)
    elif isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        if a != b:
            out.append((path, f"{a!r} vs {b!r}"))
    else:
        x, y = float(a), float(b)
        if math.isnan(x) and math.isnan(y):
            return
        if not math.isclose(x, y, rel_tol=rtol, abs_tol=atol):
            out.append((path, f"{x:.6g} vs {y:.6g}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tree_a")
    ap.add_argument("tree_b")
    ap.add_argument("--rtol", type=float, default=1e-3)
    ap.add_argument("--atol", type=float, default=1e-6)
    ap.add_argument("--only", nargs="*", default=None, help="relative JSON paths to compare")
    ap.add_argument("--max-print", type=int, default=25)
    args = ap.parse_args()
    A, B = Path(args.tree_a), Path(args.tree_b)
    total_bad = 0
    for rel in (args.only or DEFAULT_FILES):
        fa, fb = A / rel, B / rel
        if not fa.exists() or not fb.exists():
            print(f"{rel:60s} {'missing in A' if not fa.exists() else 'missing in B'}")
            continue
        diffs = []
        walk(json.loads(fa.read_text()), json.loads(fb.read_text()), "", args.rtol, args.atol, diffs)
        status = "identical" if not diffs else f"{len(diffs)} difference(s)"
        print(f"{rel:60s} {status}")
        for p, msg in diffs[:args.max_print]:
            print(f"    {p}: {msg}")
        if len(diffs) > args.max_print:
            print(f"    ... {len(diffs) - args.max_print} more")
        total_bad += len(diffs)
    print(f"\n{total_bad} differing values at rtol={args.rtol}")
    return 1 if total_bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
