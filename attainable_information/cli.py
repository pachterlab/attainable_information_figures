"""Command-line entry point: ``attainable-information``.

Runs the estimator on a synthetic dataset with known ground truth (the default)
or on two whitespace/comma-delimited numeric matrices, and prints the channel
spectrum next to the attainable ceiling at the given cohort size.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from attainable_information import (
    attainable_information,
    attainable_recoverability,
    channel_information,
    cross_validated_recoverability,
    fit_recoverability,
    learning_cost,
    make_synthetic_radiogenomics,
    true_recoverability,
)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="attainable-information",
        description="Channel recoverability spectrum and finite-sample attainable-information ceiling.",
    )
    p.add_argument("--G", help="path to an (n x p) numeric matrix (csv/tsv/npy); omit for synthetic data")
    p.add_argument("--X", help="path to an (n x d) numeric matrix (csv/tsv/npy); omit for synthetic data")
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--n", type=int, default=None, help="cohort size for the ceiling (default: rows of G)")
    p.add_argument("--d-star", type=int, default=None, help="working dimension (default: columns of X)")
    p.add_argument("--synthetic-n", type=int, default=300)
    p.add_argument("--synthetic-p", type=int, default=60)
    p.add_argument("--synthetic-d", type=int, default=20)
    p.add_argument("--synthetic-k", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    return p


def _load(path: str) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path)
    delim = "," if path.endswith(".csv") else None
    return np.loadtxt(path, delimiter=delim)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    truth = None
    if args.G and args.X:
        G, X = _load(args.G), _load(args.X)
    elif args.G or args.X:
        print("pass both --G and --X, or neither for synthetic data", file=sys.stderr)
        return 2
    else:
        G, X, truth = make_synthetic_radiogenomics(
            n=args.synthetic_n, p=args.synthetic_p, d=args.synthetic_d,
            k=args.synthetic_k, random_state=args.seed,
        )

    n, d = X.shape[0], X.shape[1]
    n_ceiling = args.n or n
    d_star = args.d_star or d
    r = min(args.n_components, G.shape[1], d)

    fit = fit_recoverability(G, X, n_components=r)
    cv = cross_validated_recoverability(G, X, r, n_folds=args.n_folds, random_state=args.seed).mean(0)
    Rn = attainable_recoverability(fit.recoverability, n_ceiling, d_star)
    nu = learning_cost(fit.recoverability, d_star)

    print(f"n = {n}, p = {G.shape[1]}, d = {d}; ceiling at n = {n_ceiling}, d* = {d_star}")
    header = f"{'dir':>3} {'R_hat':>8} {'R_cv':>8} {'R_n':>8} {'nu':>10}"
    if truth is not None:
        header += f" {'R_true':>8}"
    print(header)
    R_true = true_recoverability(truth)[1] if truth is not None else None
    for i in range(r):
        line = f"{i + 1:>3} {fit.recoverability[i]:8.3f} {cv[i]:8.3f} {Rn[i]:8.3f} {nu[i]:10.1f}"
        if R_true is not None:
            line += f" {R_true[i]:8.3f}"
        print(line)
    print(f"channel information  I(G;X) = {channel_information(fit.recoverability):.3f} bits")
    print(f"attainable ceiling   I_n    = {attainable_information(fit.recoverability, n_ceiling, d_star):.3f} bits")
    return 0


if __name__ == "__main__":
    sys.exit(main())
