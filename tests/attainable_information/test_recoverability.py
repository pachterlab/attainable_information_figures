import numpy as np
import pytest

from attainable_information import (
    anchor_decomposition,
    channel_information,
    cross_validated_recoverability,
    cv_permutation_test,
    direction_recoverability,
    distance_correlation,
    fit_recoverability,
    gaussian_rank_transform,
    make_synthetic_radiogenomics,
    mutual_information,
    permutation_test,
    posterior,
    residualize,
    subspace_alignment,
    true_recoverability,
)


@pytest.fixture(scope="module")
def synthetic():
    G, X, truth = make_synthetic_radiogenomics(n=2000, p=40, d=15, k=3, random_state=1)
    return G, X, truth


def test_spectrum_recovers_ground_truth(synthetic):
    G, X, truth = synthetic
    fit = fit_recoverability(G, X, reg_g=0.0, reg_x=0.0, n_components=6)
    rho, R = true_recoverability(truth)
    assert fit.recoverability[:3] == pytest.approx(R[:3], abs=0.03)
    assert np.all(fit.recoverability[3:] < 0.05)
    assert fit.mutual_information() == pytest.approx(mutual_information(rho), abs=0.1)


def test_directions_align_with_truth(synthetic):
    G, X, truth = synthetic
    fit = fit_recoverability(G, X, reg_g=0.0, reg_x=0.0, n_components=3)
    # the identifiable genomic subspace is span(Sg^{-1} Wg); compare via alignment
    target = np.linalg.solve(truth["Sg"], truth["Wg"])
    cos = subspace_alignment(fit.genomic_dirs, target)
    assert cos.min() > 0.9


def test_posterior_is_consistent(synthetic):
    G, X, _ = synthetic
    fit = fit_recoverability(G, X, reg_g=0.0, reg_x=0.0)
    B, Sgcx = posterior(fit)
    assert B.shape == (G.shape[1], X.shape[1])
    w = np.linalg.eigvalsh(Sgcx)
    assert w.min() > -1e-8
    # named-direction recoverability equals rho^2 on a canonical direction
    a0 = fit.genomic_dirs[:, 0]
    assert direction_recoverability(a0, fit.Sg, Sgcx) == pytest.approx(fit.recoverability[0], abs=1e-6)


def test_cross_validation_below_channel(synthetic):
    G, X, truth = synthetic
    cv = cross_validated_recoverability(G, X, n_components=3, n_folds=5)
    _, R = true_recoverability(truth)
    assert cv.shape == (5, 3)
    assert cv.mean(0)[0] == pytest.approx(R[0], abs=0.05)


def test_permutation_tests_detect_signal():
    G, X, _ = make_synthetic_radiogenomics(n=200, p=20, d=10, k=2, random_state=3)
    _, _, p = permutation_test(G, X, n_components=2, n_perm=50)
    assert p[0] < 0.05
    _, _, p_cv = cv_permutation_test(G, X, n_components=2, n_perm=20, n_folds=3)
    assert p_cv[0] < 0.1


def test_null_data_has_no_cv_signal():
    rng = np.random.default_rng(0)
    G, X = rng.standard_normal((300, 20)), rng.standard_normal((300, 10))
    cv = cross_validated_recoverability(G, X, n_components=2, n_folds=5).mean(0)
    assert cv.max() < 0.05


def test_gaussian_rank_transform_marginals():
    rng = np.random.default_rng(0)
    M = rng.exponential(size=(500, 3))
    Z = gaussian_rank_transform(M)
    assert Z.mean(0) == pytest.approx(np.zeros(3), abs=1e-8)
    assert Z.std(0) == pytest.approx(np.ones(3), abs=0.05)


def test_distance_correlation_bounds():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((200, 2))
    assert distance_correlation(x, x) == pytest.approx(1.0)
    assert 0.0 <= distance_correlation(x, rng.standard_normal((200, 2))) < 0.3


def test_residualize_is_orthogonal():
    rng = np.random.default_rng(0)
    C = rng.standard_normal((100, 3))
    Y = C @ rng.standard_normal((3, 5)) + rng.standard_normal((100, 5))
    Yp = residualize(Y, C, ridge=0.0)
    assert np.abs((C - C.mean(0)).T @ Yp).max() < 1e-6


def test_anchor_decomposition_sums(synthetic):
    G, X, _ = synthetic
    out = anchor_decomposition(G[:, :5], G[:, 5:], X, n_components=3)
    assert out["I_total_bits"] == pytest.approx(out["I_anchor_bits"] + out["I_residual_bits"])
    assert 0.0 <= out["eta"] <= 1.0
    assert out["I_anchor_bits"] == pytest.approx(channel_information(out["rho2_anchor"]))


def test_bernoulli_synthetic_is_binary():
    G, X, truth = make_synthetic_radiogenomics(n=300, p=10, d=5, k=2, genomics_type="bernoulli", mutation_freq=0.3)
    assert set(np.unique(G)) <= {0.0, 1.0}
    assert G.mean() == pytest.approx(0.3, abs=0.05)
    assert truth["G_latent"].shape == G.shape
