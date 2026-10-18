"""Checks for statistical validity and the user's data-access/retention rules."""
import numpy as np
import pytest
from threadpoolctl import threadpool_limits

from gaussian_core import (batch_from_array, fit_gaussian, gaussian_batch,
                           merge_batches, tail_probabilities)
from run_experiments import ALL_METRICS, Experiment, simulate


@pytest.fixture(autouse=True)
def single_thread():
    with threadpool_limits(limits=1):
        yield


def test_pooling_matches_actual_concatenated_observations_and_ddof_one():
    rng = np.random.default_rng(112)
    a, b = rng.normal(size=(20, 4)), rng.normal(3, 2, size=(7, 4))
    pooled = merge_batches(batch_from_array(a), batch_from_array(b))
    model = fit_gaussian(pooled)
    actual = np.vstack([a, b])
    np.testing.assert_allclose(model.mean, actual.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(model.covariance, np.cov(actual, rowvar=False, ddof=1), atol=1e-13)


@pytest.mark.parametrize("batch_size", [2, 17, 10000])
def test_gaussian_statistics_have_finite_sample_moments(batch_size):
    rng = np.random.default_rng(873 + batch_size)
    model = fit_gaussian(batch_from_array(rng.normal(size=(80, 3))))
    means, scatters = [], []
    for _ in range(4000):
        batch = gaussian_batch(model, batch_size, rng)
        means.append(batch.mean)
        scatters.append(batch.scatter / (batch_size - 1))
    means, scatters = np.asarray(means), np.asarray(scatters)
    np.testing.assert_allclose(scatters.mean(axis=0), model.covariance, atol=.075, rtol=.035)
    np.testing.assert_allclose(np.cov(means, rowvar=False)*batch_size,
                               model.covariance, atol=.075, rtol=.04)
    np.testing.assert_allclose(means.mean(axis=0), model.mean,
                               atol=5*np.sqrt(np.max(np.diag(model.covariance))/batch_size/4000))


def test_singular_gaussian_uses_original_support_without_jitter():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 2))
    x = np.column_stack([x, np.zeros(30), np.zeros(30)])
    model = fit_gaussian(batch_from_array(x))
    batch = gaussian_batch(model, 1000, rng)
    np.testing.assert_array_equal(batch.mean[2:], np.zeros(2))
    np.testing.assert_array_equal(batch.scatter[2:, :], np.zeros((2, 4)))


def test_cumulative_real_k0_matches_all_actual_real_observations():
    config = Experiment("cumulative_real", d=4, n0=20, r=5, k=0, iterations=7)
    saved = simulate(config, 8, snapshots=True)
    initial_rng = np.random.default_rng(np.random.SeedSequence([8, 1, 4]))
    fresh_rng = np.random.default_rng(np.random.SeedSequence([8, 2, 4]))
    observations = [initial_rng.normal(size=(20, 4))]
    observations += [fresh_rng.normal(size=(5, 4)) for _ in range(6)]
    expected = np.vstack(observations)
    np.testing.assert_allclose(saved["snapshot_means"][-1], expected.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(saved["snapshot_covariances"][-1], np.cov(expected, rowvar=False), atol=1e-13)
    assert saved["metrics"][-1, ALL_METRICS.index("n_real_fit")] == 50


def test_current_real_k0_uses_only_last_batch_and_discards_initial_data():
    config = Experiment("current_fresh", d=4, n0=20, r=5, k=0, iterations=7)
    saved = simulate(config, 8, snapshots=True)
    fresh_rng = np.random.default_rng(np.random.SeedSequence([8, 2, 4]))
    for _ in range(6):
        last = fresh_rng.normal(size=(5, 4))
    np.testing.assert_allclose(saved["snapshot_means"][-1], last.mean(axis=0), atol=1e-14)
    np.testing.assert_allclose(saved["snapshot_covariances"][-1], np.cov(last, rowvar=False), atol=1e-13)
    assert saved["metrics"][-1, ALL_METRICS.index("n_real_fit")] == 5


def test_mixture_p1_exactly_reduces_to_current_fresh_mechanism():
    a = simulate(Experiment("current_fresh", d=5, n0=30, r=10, k=.5, iterations=12), 5)
    b = simulate(Experiment("unlabelled_mixture", d=5, n0=30, r=10, k=.5,
                            real_fraction=1, iterations=12), 5)
    np.testing.assert_array_equal(a["metrics"], b["metrics"])


def test_mixture_policy_uses_visible_size_not_hidden_true_count():
    config = Experiment("unlabelled_mixture", d=4, n0=20, r=10, k=2,
                        real_fraction=0, iterations=10)
    result = simulate(config, 12)["metrics"]
    np.testing.assert_array_equal(result[1:, ALL_METRICS.index("n_real_fit")], 0)
    np.testing.assert_array_equal(result[1:, ALL_METRICS.index("n_synthetic_extra")], 20)
    np.testing.assert_array_equal(result[1:, ALL_METRICS.index("n_train")], 30)
    assert result[-1, ALL_METRICS.index("cumulative_real_acquired")] == 20


def test_fixed_initial_k0_is_constant_and_fixed_count_policy_stays_fixed():
    constant = simulate(Experiment("fixed_initial", d=4, n0=20, r=0, k=0, iterations=8), 12)["metrics"]
    np.testing.assert_allclose(constant[:, :13], np.repeat(constant[:1, :13], 8, axis=0), atol=1e-14)
    fixed = simulate(Experiment("cumulative_real", d=4, n0=20, r=5, k=2,
                                policy="fixed_count", iterations=8), 12)["metrics"]
    np.testing.assert_array_equal(fixed[1:, ALL_METRICS.index("n_synthetic_extra")], 50)


def test_zero_variance_tail_uses_point_mass_and_boundary_is_strict():
    original, centered = tail_probabilities(np.array([0, 2, 3.0]), np.zeros(3), 2)
    np.testing.assert_array_equal(original, [0, 0, 1])
    np.testing.assert_array_equal(centered, [0, 0, 0])


def test_seed_reproducibility():
    config = Experiment("current_fresh", d=4, n0=20, r=5, k=1, iterations=8)
    np.testing.assert_array_equal(simulate(config, 19)["metrics"], simulate(config, 19)["metrics"])
