"""Independent checks of data compression, mixture moments and tail protection."""
import numpy as np
import pytest
from scipy.special import ndtr

from gaussian_core import batch_from_array, fit_gaussian, gaussian_batch, merge_batches
from memory.parameter_memory import (anchor_parameters, initial_sufficient_statistics,
                              model_from_parameters, parameter_update)


def test_initial_parameters_reproduce_raw_data_replay_path():
    rng = np.random.default_rng(303)
    x = rng.normal(size=(40, 6))
    initial = fit_gaussian(batch_from_array(x))
    compressed = initial_sufficient_statistics(initial, len(x))
    previous_raw = previous_compressed = initial
    for t in range(35):
        replay_a = gaussian_batch(previous_raw, 75, np.random.default_rng(1000 + t))
        replay_b = gaussian_batch(previous_compressed, 75, np.random.default_rng(1000 + t))
        previous_raw = fit_gaussian(merge_batches(batch_from_array(x), replay_a))
        previous_compressed = fit_gaussian(merge_batches(compressed, replay_b))
        np.testing.assert_allclose(previous_raw.mean, previous_compressed.mean, atol=1e-11)
        np.testing.assert_allclose(previous_raw.covariance, previous_compressed.covariance, atol=1e-11)


def test_anchor_matches_raw_second_moment_mixture_and_has_covariance_floor():
    rng = np.random.default_rng(892)
    for d in (1, 5, 15):
        initial = fit_gaussian(batch_from_array(rng.normal(size=(d + 20, d))))
        candidate = fit_gaussian(batch_from_array(rng.normal(1, 2, size=(d + 10, d))))
        for weight in (0, .1, .5, 1):
            mu, cov = anchor_parameters(candidate.mean, candidate.covariance, initial, weight)
            raw_second = ((1-weight)*(candidate.covariance + np.outer(candidate.mean, candidate.mean))
                          + weight*(initial.covariance + np.outer(initial.mean, initial.mean)))
            np.testing.assert_allclose(cov + np.outer(mu, mu), raw_second, atol=1e-13)
            assert np.linalg.eigvalsh(cov-weight*initial.covariance).min() >= -1e-12


def test_pure_anchoring_converges_to_initial_model_with_exact_rate():
    initial = model_from_parameters(np.array([.2, -.1]), np.array([[1, .3], [.3, 2]]))
    mu_start = np.array([2., -3.])
    cov_start = np.diag([.01, .5])
    mu, cov = mu_start.copy(), cov_start.copy()
    raw0 = initial.covariance + np.outer(initial.mean, initial.mean)
    raw_start = cov_start + np.outer(mu_start, mu_start)
    for t in range(1, 301):
        mu, cov = anchor_parameters(mu, cov, initial, .1)
        np.testing.assert_allclose(mu-initial.mean, .9**t*(mu_start-initial.mean), atol=1e-14)
        np.testing.assert_allclose(cov+np.outer(mu, mu)-raw0, .9**t*(raw_start-raw0), atol=2e-14)
    np.testing.assert_allclose(cov, initial.covariance, atol=1e-12)


def test_gaussian_projection_does_not_preserve_mixture_tail_lower_bound():
    initial = model_from_parameters(np.zeros(1), np.ones((1, 1)))
    _, cov = anchor_parameters(np.zeros(1), np.zeros((1, 1)), initial, .1)
    gaussian_tail = 2*ndtr(-3/np.sqrt(cov[0, 0]))
    actual_mixture_tail_lower_bound = .1*2*ndtr(-3)
    assert gaussian_tail < actual_mixture_tail_lower_bound / 1000


def test_full_anchor_is_exactly_initial_despite_new_noisy_data():
    rng = np.random.default_rng(492)
    initial = fit_gaussian(batch_from_array(rng.normal(size=(40, 4))))
    arrivals = batch_from_array(rng.normal(3, 2, size=(10, 4)))
    result = parameter_update(initial, arrivals, 50, initial, 1)
    np.testing.assert_allclose(result.mean, initial.mean, atol=1e-14)
    np.testing.assert_allclose(result.covariance, initial.covariance, atol=1e-14)


def test_invalid_anchor_and_indefinite_covariance_are_rejected():
    initial = model_from_parameters(np.zeros(1), np.ones((1, 1)))
    with pytest.raises(ValueError):
        anchor_parameters(initial.mean, initial.covariance, initial, -0.1)
    with pytest.raises(ValueError):
        model_from_parameters(np.zeros(2), np.diag([1., -1.]))
