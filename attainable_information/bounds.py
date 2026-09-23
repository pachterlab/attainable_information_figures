"""Attainable-information bounds: what *any* model trained on n patients can extract.

The rest of :mod:`attainable_information` estimates the population recoverability spectrum
``{rho_i^2}`` -- the infinite-data ceiling set by the biology--imaging channel.
This module adds the finite-sample ceiling: the largest out-of-sample
recoverability (and hence mutual information) achievable by *any* learning
algorithm given ``n`` training patients and a ``d_star``-dimensional imaging
representation.

The central identity (paper, section "The attainable-information ceiling") is

    R_n(rho^2) = rho^2 * n / (n + nu),      nu = d_star * (1 - rho^2) / rho^2,

which is 0 at ``n = 0``, increases monotonically, and saturates at the channel
value ``rho^2``. ``nu`` is the *learning cost* of the direction: the sample size
at which half the channel's recoverability is attainable. It is also exactly the
optimal ridge penalty for the corresponding regression.

Everything here is a function of ``(rho^2, n, d_star)`` only -- no data -- so the
bounds can be drawn as curves and trained models plotted underneath them.
"""

from __future__ import annotations

import numpy as np

from attainable_information.recoverability import to_dense

__all__ = [
    "learning_cost",
    "attainable_recoverability",
    "channel_information",
    "attainable_information",
    "sample_size_for_fraction",
    "optimal_working_dimension",
    "auc_ceiling",
    "weak_direction_bound",
    "max_total_information",
    "max_information_given_fourth_moment",
    "residualize",
    "anchor_decomposition",
]

_LOG2 = np.log(2.0)


def _as_rho2(rho2):
    r = np.atleast_1d(np.asarray(rho2, dtype=np.float64))
    return np.clip(r, 0.0, 1.0 - 1e-12)


# ---------------------------------------------------------------------------
# the finite-sample ceiling
# ---------------------------------------------------------------------------
def learning_cost(rho2, d_star: int) -> np.ndarray:
    """Learning cost ``nu = d_star (1 - rho^2) / rho^2`` of a canonical direction.

    ``nu`` is the number of patients at which half of the direction's channel
    recoverability is attainable, and simultaneously the Bayes-optimal ridge
    penalty (in whitened imaging coordinates) for predicting that direction.
    Directions with weak population correlation are expensive: ``nu`` diverges
    like ``d_star / rho^2``.
    """
    r = _as_rho2(rho2)
    with np.errstate(divide="ignore"):
        nu = d_star * (1.0 - r) / np.where(r > 0, r, np.nan)
    return nu


def attainable_recoverability(rho2, n: int, d_star: int) -> np.ndarray:
    """Upper bound on out-of-sample ``R`` for any algorithm trained on ``n`` patients.

    Parameters
    ----------
    rho2 : float or array
        Population (channel) recoverability of the direction(s), ``rho_i^2``.
    n : int
        Number of training patients.
    d_star : int
        Working dimension of the imaging representation.

    Returns
    -------
    array of the same shape as ``rho2``, values in ``[0, rho2]``.

    Notes
    -----
    Equals ``rho^2 n / (n + nu)``. Derived as the Bayes risk of the
    SNR-matched g-prior ``beta ~ N(0, (rho^2/d_star) Sigma_X^{-1})``; the
    substitution ``X'X -> n Sigma_X`` is conservative by operator convexity of
    the matrix inverse, so the expression is a valid upper bound for a random
    design as well as an exactly attained value for a balanced one.
    """
    r = _as_rho2(rho2)
    n = float(n)
    out = np.zeros_like(r)
    pos = r > 0
    # written as n r^2 / (d_star + r (n - d_star)) to stay finite as r -> 0
    denom = d_star + r[pos] * (n - d_star)
    out[pos] = n * r[pos] ** 2 / denom
    return np.clip(out, 0.0, 1.0 - 1e-12)


def channel_information(rho2, bits: bool = True) -> float:
    """Infinite-data channel information ``I(G;X) = -1/2 sum log(1 - rho_i^2)``."""
    r = _as_rho2(rho2)
    val = -0.5 * np.sum(np.log1p(-r))
    return float(val / _LOG2) if bits else float(val)


def attainable_information(rho2, n: int, d_star: int, bits: bool = True) -> float:
    """Ceiling on genomic information extractable from one image at cohort size ``n``.

    ``I_n = -1/2 sum_i log(1 - R_n(rho_i^2))``. This is the headline number: no
    predictor trained on ``n`` patients conveys more than ``I_n`` about the
    genomic state of a held-out patient. ``I_n -> I(G;X)`` as ``n -> inf`` and
    ``I_0 = 0``.

    The canonical directions are independent under the model (the residuals
    ``S_i - rho_i T_i`` have covariance ``diag(1 - rho_i^2)``), so the
    per-direction ceilings add exactly.
    """
    Rn = attainable_recoverability(rho2, n, d_star)
    val = -0.5 * np.sum(np.log1p(-Rn))
    return float(val / _LOG2) if bits else float(val)


def sample_size_for_fraction(rho2, d_star: int, fraction: float = 0.9) -> np.ndarray:
    """Patients needed to attain ``fraction`` of the channel recoverability.

    Inverts ``n / (n + nu) = fraction``, giving ``n = nu * f / (1 - f)``.
    """
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must lie in (0, 1)")
    return learning_cost(rho2, d_star) * fraction / (1.0 - fraction)


def optimal_working_dimension(rho2_by_dim: dict[int, np.ndarray], n: int):
    """Best imaging feature budget at cohort size ``n``.

    Enlarging the imaging representation can only raise the population spectrum
    (data processing), but raises the learning cost ``nu`` proportionally. The
    attainable information is therefore maximized at a finite ``d_star``.

    Parameters
    ----------
    rho2_by_dim : mapping ``d_star -> array of rho_i^2``
        Population spectra estimated at each candidate working dimension.
    n : int

    Returns
    -------
    (best_d_star, best_information_bits, {d_star: information_bits})
    """
    curve = {
        d: attainable_information(r, n, d) for d, r in sorted(rho2_by_dim.items())
    }
    best_d = max(curve, key=curve.get)
    return best_d, curve[best_d], curve


def weak_direction_bound(rho2, n: int, d_star: int) -> np.ndarray:
    """The quartic small-signal bound ``R_n <= n rho^4 / (d_star (1 - rho^2))``.

    Follows from ``n / (n + nu) <= n / nu``. The point is the exponent: for weak
    channels the *attainable* signal decays like ``rho^4``, not ``rho^2``, so
    halving the canonical correlation quarters what a finite cohort can recover.
    """
    r = _as_rho2(rho2)
    out = np.full_like(r, np.inf)
    pos = r > 0
    # r is rho^2, so rho^4 = r ** 2
    out[pos] = n * r[pos] ** 2 / (d_star * (1.0 - r[pos]))
    return np.minimum(out, r)


def max_total_information(total_signal, n: int, d_star: int,
                          rho2_max: float = 1.0, bits: bool = True) -> float:
    """Largest attainable information consistent with a total-signal budget.

    ``attainable_information`` needs the whole spectrum. When only the *total*
    signal ``T = sum_i rho_i^2`` and a per-direction cap ``rho2_max`` are known,
    this returns the maximum of ``I_n`` over every spectrum satisfying both --
    i.e. a genuine upper bound on total information retention rather than on the
    leading axis alone.

    The maximiser is concentration: ``u -> -1/2 log(1 - R_n(u))`` is convex on
    ``[0, 1]``, so at fixed total the information is largest when the signal is
    packed into as few directions as possible. The optimum therefore fills
    ``floor(T / rho2_max)`` directions at the cap and puts the remainder in one
    more -- a continuous knapsack.

    Note that ``T`` alone cannot be bounded from data (signal spread thinly
    enough over many directions is undetectable at any ``n``); it is precisely
    that regime in which the convexity makes ``I_n`` small, which is why the
    bound is stated in terms of information rather than of ``T``.
    """
    T = float(total_signal)
    cap = float(min(max(rho2_max, 1e-12), 1.0 - 1e-12))
    if T <= 0:
        return 0.0
    full = int(np.floor(T / cap + 1e-12))
    rem = max(T - full * cap, 0.0)
    val = full * -0.5 * np.log1p(-attainable_recoverability(cap, n, d_star)[0])
    if rem > 0:
        val += -0.5 * np.log1p(-attainable_recoverability(rem, n, d_star)[0])
    return float(val / _LOG2) if bits else float(val)


def max_information_given_fourth_moment(fourth_moment, n: int, d_star: int,
                                        bits: bool = True) -> float:
    """Spectrum-free ceiling from a fourth-moment budget ``S = sum_i rho_i^4``.

    ``max_total_information`` needs a budget on ``sum_i rho_i^2``, which cannot
    be estimated (signal spread thinly enough is undetectable at any ``n``). The
    fourth moment does not have that defect: thin spread drives ``S`` to zero
    at the same rate it drives the attainable information to zero, so ``S`` is
    both estimable and sufficient.

    Maximising ``sum_i g(rho_i^2)``, ``g(u) = -1/2 log(1 - R_n(u))``, subject to
    ``sum_i rho_i^4 <= S`` and rank ``<= d_star`` has its optimum at the *even*
    spread (the opposite of :func:`max_total_information`, because a
    fourth-moment budget penalises concentration quadratically). Since
    ``k g(sqrt(S/k))`` increases in ``k``, the maximum is at ``k = d_star``:

        I_n <= d_star * g(sqrt(S / d_star)).

    This holds for *every* spectrum satisfying the budget -- no family, no shape
    assumption -- so it converts an estimate of the single scalar ``S`` into a
    genuine upper bound on total information retention.
    """
    S = float(fourth_moment)
    if S <= 0:
        return 0.0
    u = np.sqrt(S / d_star)
    if u >= 1.0:
        return float("inf")
    val = d_star * -0.5 * np.log1p(-attainable_recoverability(u, n, d_star)[0])
    return float(val / _LOG2) if bits else float(val)


# ---------------------------------------------------------------------------
# anchor-gene (chain-rule) decomposition
# ---------------------------------------------------------------------------
def residualize(Y, C, ridge: float = 1e-8) -> np.ndarray:
    """Regress ``C`` out of ``Y`` columnwise (both centred first).

    This is Gram--Schmidt in the covariance metric: the returned matrix is the
    component of ``Y`` orthogonal to the span of ``C``, which is what the
    conditional term of the chain rule requires.
    """
    Y = to_dense(Y).astype(np.float64)
    C = to_dense(C).astype(np.float64)
    Yc = Y - Y.mean(0)
    Cc = C - C.mean(0)
    gram = Cc.T @ Cc
    gram.flat[:: gram.shape[0] + 1] += ridge * np.trace(gram) / max(gram.shape[0], 1)
    coef = np.linalg.solve(gram, Cc.T @ Yc)
    return Yc - Cc @ coef


def anchor_decomposition(G_anchor, G_rest, X, n_components: int | None = None,
                         reg_g="lw", reg_x="lw"):
    """Chain-rule split of the genomic information into anchor and residual parts.

    ``I(G; X) = I(G_A; X) + I(G_rest; X | G_A)``, exactly, under the joint
    Gaussian law. The conditional term is computed as the plain mutual
    information of the pair after residualizing both ``G_rest`` and ``X`` on the
    anchor block -- i.e. after whitening the genomics matrix against
    hand-specified anchor genes.

    Returns a dict with the two spectra, the two information terms (bits), and
    the *anchor sufficiency* ``eta = I(G_A;X) / I(G;X)``.
    """
    from attainable_information.recoverability import fit_recoverability

    G_anchor = to_dense(G_anchor)
    G_rest = to_dense(G_rest)
    X = to_dense(X)

    k_a = min(n_components or G_anchor.shape[1], G_anchor.shape[1], X.shape[1])
    fit_a = fit_recoverability(G_anchor, X, reg_g, reg_x, k_a)

    G_perp = residualize(G_rest, G_anchor)
    X_perp = residualize(X, G_anchor)
    k_r = min(n_components or G_perp.shape[1], G_perp.shape[1], X_perp.shape[1])
    fit_r = fit_recoverability(G_perp, X_perp, reg_g, reg_x, k_r)

    I_a = channel_information(fit_a.recoverability)
    I_r = channel_information(fit_r.recoverability)
    total = I_a + I_r
    return {
        "rho2_anchor": fit_a.recoverability,
        "rho2_residual": fit_r.recoverability,
        "I_anchor_bits": I_a,
        "I_residual_bits": I_r,
        "I_total_bits": total,
        "eta": I_a / total if total > 0 else np.nan,
        "fit_anchor": fit_a,
        "fit_residual": fit_r,
    }


# ---------------------------------------------------------------------------
# binary / probit genomics
# ---------------------------------------------------------------------------
def auc_ceiling(recoverability) -> np.ndarray:
    """Bayes AUC ceiling for a median-binarized latent-Gaussian genomic feature.

    ``AUC <= 1/2 + (2/pi) arcsin( sqrt(R / 2) )``, the exact AUC of the Bayes
    score ``E[Z | X]`` for ``G = 1{Z > 0}`` when ``(Z, X)`` are jointly Gaussian
    and ``R = Var(E[Z | X])`` (a trivariate orthant probability). The median
    threshold is the least favourable prevalence; a feature binarized away from
    its median admits a higher Bayes AUC at the same ``R``. Pass
    :func:`attainable_recoverability` output to get the *finite-sample* AUC
    ceiling at cohort size ``n``.

    The earlier binormal expression ``Phi(sqrt(R / (2 (1 - R))))`` is not an
    upper bound and has been replaced.
    """
    R = _as_rho2(recoverability)
    return 0.5 + (2.0 / np.pi) * np.arcsin(np.sqrt(R / 2.0))
