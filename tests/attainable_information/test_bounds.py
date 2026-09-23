import numpy as np
import pytest

from attainable_information import (
    attainable_information,
    attainable_recoverability,
    auc_ceiling,
    channel_information,
    learning_cost,
    max_information_given_fourth_moment,
    max_total_information,
    optimal_working_dimension,
    sample_size_for_fraction,
    weak_direction_bound,
)


def test_closed_form_matches_definition():
    rho2, n, d = 0.3, 150, 20
    nu = d * (1 - rho2) / rho2
    assert learning_cost(rho2, d)[0] == pytest.approx(nu)
    assert attainable_recoverability(rho2, n, d)[0] == pytest.approx(rho2 * n / (n + nu))


def test_limits():
    rho2 = np.array([0.6, 0.3, 0.05])
    assert np.all(attainable_recoverability(rho2, 0, 20) == 0.0)
    assert attainable_recoverability(rho2, 10**9, 20) == pytest.approx(rho2, rel=1e-6)
    assert attainable_recoverability(0.0, 100, 20)[0] == 0.0
    assert attainable_information(rho2, 0, 20) == 0.0
    assert attainable_information(rho2, 10**9, 20) == pytest.approx(channel_information(rho2), rel=1e-6)


def test_monotone_in_n_and_bounded_by_channel():
    rho2 = np.array([0.5, 0.2, 0.02])
    prev = np.zeros(3)
    for n in [1, 5, 20, 100, 1000, 10000]:
        cur = attainable_recoverability(rho2, n, 30)
        assert np.all(cur >= prev)
        assert np.all(cur <= rho2 + 1e-12)
        prev = cur


def test_half_channel_at_learning_cost():
    rho2, d = 0.25, 12
    nu = learning_cost(rho2, d)[0]
    assert attainable_recoverability(rho2, nu, d)[0] == pytest.approx(rho2 / 2)


def test_sample_size_for_fraction_inverts_ceiling():
    rho2, d, f = 0.15, 40, 0.9
    n = sample_size_for_fraction(rho2, d, f)[0]
    assert attainable_recoverability(rho2, n, d)[0] == pytest.approx(f * rho2)


def test_weak_direction_bound_dominates():
    rho2 = np.linspace(0.005, 0.5, 30)
    n, d = 100, 20
    assert np.all(weak_direction_bound(rho2, n, d) >= attainable_recoverability(rho2, n, d) - 1e-12)
    # min(rho^2, n rho^4 / (d (1 - rho^2))) with rho2 = rho^2
    expected = np.minimum(rho2, n * rho2**2 / (d * (1 - rho2)))
    assert weak_direction_bound(rho2, n, d) == pytest.approx(expected)


def test_information_in_bits_vs_nats():
    rho2 = [0.4, 0.1]
    assert channel_information(rho2, bits=True) == pytest.approx(channel_information(rho2, bits=False) / np.log(2))


def test_optimal_working_dimension_is_finite():
    # same channel at every d_star: the smallest representation wins
    spectra = {d: np.array([0.3, 0.1]) for d in (5, 20, 80)}
    best, _, curve = optimal_working_dimension(spectra, n=100)
    assert best == 5
    assert curve[5] > curve[20] > curve[80]


def test_spectrum_free_bounds_dominate_spectrum():
    rho2 = np.array([0.35, 0.2, 0.1, 0.05])
    n, d = 120, 25
    exact = attainable_information(rho2, n, d)
    assert max_total_information(rho2.sum(), n, d) >= exact - 1e-9
    assert max_information_given_fourth_moment(np.sum(rho2**2), n, d) >= exact - 1e-9
    assert max_total_information(0.0, n, d) == 0.0
    assert max_information_given_fourth_moment(0.0, n, d) == 0.0


def test_auc_ceiling_range():
    a = auc_ceiling(np.array([0.0, 0.5, 0.99]))
    assert a[0] == pytest.approx(0.5)
    assert np.all(np.diff(a) > 0)
    assert a[-1] < 1.0
    assert auc_ceiling(1.0) == pytest.approx(1.0)


def test_auc_ceiling_matches_orthant_probability():
    # AUC of the Bayes score mu = E[Z|X] for G = 1{Z > 0}, with Var(mu) = R and
    # Cov(mu, Z) = R, is 1/2 + (2/pi) arcsin(sqrt(R/2)); check by simulation.
    rng = np.random.default_rng(0)
    R = 0.3
    z = rng.standard_normal(400_000)
    mu = R * z + np.sqrt(R * (1 - R)) * rng.standard_normal(z.size)
    pos, neg = mu[z > 0], mu[z <= 0]
    m = 200_000
    sim = np.mean(rng.choice(pos, m) > rng.choice(neg, m))
    assert auc_ceiling(R) == pytest.approx(0.7532, abs=1e-3)
    assert sim == pytest.approx(auc_ceiling(R), abs=5e-3)
