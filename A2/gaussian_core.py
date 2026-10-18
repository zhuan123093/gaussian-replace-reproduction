"""Exact finite-sample Gaussian recursion; sample covariance always uses N - 1.

BatchStats contains no source labels. A Gaussian learner can use these sufficient
statistics without observing individual observations or their origin.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr


@dataclass
class BatchStats:
    n: int
    mean: np.ndarray
    scatter: np.ndarray


@dataclass
class GaussianModel:
    mean: np.ndarray
    covariance: np.ndarray
    eigenvalues: np.ndarray
    factor: np.ndarray
    clipped_negative: int = 0
    largest_negative_relative: float = 0.0


def empty_batch(d: int) -> BatchStats:
    return BatchStats(0, np.zeros(d), np.zeros((d, d)))


def batch_from_array(x: np.ndarray) -> BatchStats:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("Observations must have shape (n, d)")
    if len(x) == 0:
        return empty_batch(x.shape[1])
    mean = x.mean(axis=0)
    centered = x - mean
    return BatchStats(len(x), mean, centered.T @ centered)


def merge_batches(a: BatchStats, b: BatchStats) -> BatchStats:
    """Stable exact pooling, including the between-batch mean correction."""
    if a.n == 0:
        return BatchStats(b.n, b.mean.copy(), b.scatter.copy())
    if b.n == 0:
        return BatchStats(a.n, a.mean.copy(), a.scatter.copy())
    n = a.n + b.n
    delta = b.mean - a.mean
    mean = a.mean + (b.n / n) * delta
    scatter = a.scatter + b.scatter + (a.n * b.n / n) * np.outer(delta, delta)
    return BatchStats(n, mean, scatter)


def fit_gaussian(batch: BatchStats) -> GaussianModel:
    if batch.n < 2:
        raise ValueError("N >= 2 is required for covariance denominator N - 1")
    cov = batch.scatter / (batch.n - 1)
    cov = (cov + cov.T) * 0.5
    values, vectors = np.linalg.eigh(cov)
    scale = float(np.max(np.abs(values)))
    relative = max(0.0, -float(values[0])) / scale if scale else 0.0
    tolerance = 128 * len(values) * np.finfo(np.float64).eps
    if relative > tolerance:
        raise FloatingPointError(f"Covariance is not PSD: relative eigenvalue {relative}")
    negative_count = int(np.count_nonzero(values < 0))
    values = np.maximum(values, 0.0)
    # Known algebraic rank bound, not a variance floor or ridge regularizer.
    if batch.n - 1 < len(values):
        values[: len(values) - (batch.n - 1)] = 0
    positive = values > 0
    factor = vectors[:, positive] * np.sqrt(values[positive])
    if negative_count or batch.n - 1 < len(values):
        cov = factor @ factor.T
    return GaussianModel(batch.mean.copy(), cov, values, factor,
                         negative_count, relative)


def gaussian_batch(model: GaussianModel, n: int, rng: np.random.Generator) -> BatchStats:
    """Draw actual random sample sufficient statistics, never population moments.

For large batches use a Bartlett Wishart draw; for small/singular batches draw
the centered Gaussian contrast matrix. The sample mean and scatter are
independent, with scatter ~ Wishart(n-1, covariance), including singular cases.
"""
    if n < 0:
        raise ValueError("Batch size cannot be negative")
    d, rank = model.factor.shape
    if n == 0:
        return empty_batch(d)
    mean = model.mean + model.factor @ rng.standard_normal(rank) / np.sqrt(n)
    if n == 1 or rank == 0:
        return BatchStats(n, mean, np.zeros((d, d)))
    df = n - 1
    if df < rank:
        transformed = model.factor @ rng.standard_normal((rank, df))
    else:
        bartlett = np.tril(rng.standard_normal((rank, rank)), k=-1)
        diagonal = np.sqrt(rng.chisquare(df - np.arange(rank)))
        bartlett[np.diag_indices(rank)] = diagonal
        transformed = model.factor @ bartlett
    return BatchStats(n, mean, transformed @ transformed.T)


def choose_real_batch(mode: str, initial: BatchStats, cached: BatchStats,
                      fresh: BatchStats) -> BatchStats:
    if mode == "fixed_initial":
        return initial
    if mode == "cumulative_real":
        return merge_batches(cached, fresh)
    if mode in {"current_fresh", "unlabelled_mixture"}:
        return fresh
    raise ValueError(mode)


def mixture_arrival(model: GaussianModel, fresh_array: np.ndarray, real_fraction: float,
                    mask_rng: np.random.Generator, fake_rng: np.random.Generator
                    ) -> tuple[BatchStats, int]:
    """Environment only: learner receives the pooled batch, not mask or labels.

The integer returned separately is an evaluator-only acquisition counter.
The observable size stays len(fresh_array), including random source counts.
"""
    if not 0 <= real_fraction <= 1:
        raise ValueError("real_fraction must be between 0 and 1")
    mask = mask_rng.random(len(fresh_array)) < real_fraction
    n_real = int(mask.sum())
    real = batch_from_array(fresh_array[mask])
    fake = gaussian_batch(model, len(fresh_array) - n_real, fake_rng)
    return merge_batches(real, fake), n_real


def tail_probabilities(means: np.ndarray, variances: np.ndarray, c: float
                       ) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray(means)
    variances = np.asarray(variances)
    original = np.empty_like(means, dtype=float)
    centered = np.zeros_like(means, dtype=float)
    positive = variances > 0
    std = np.sqrt(variances[positive])
    original[positive] = (ndtr((-c - means[positive]) / std)
                          + ndtr((means[positive] - c) / std))
    centered[positive] = 2 * ndtr(-c / std)
    original[~positive] = (np.abs(means[~positive]) > c).astype(float)
    return original, centered


METRIC_NAMES = [
    "mean_mse", "trace_ratio", "w2_per_dim",
    "tail2_original", "tail3_original", "tail2_centered", "tail3_centered",
    "coordinate_mean", "coordinate_variance",
    "coordinate_tail2_original", "coordinate_tail3_original",
    "coordinate_tail2_centered", "coordinate_tail3_centered",
    "n_real_fit", "n_arrivals", "n_real_arrivals", "n_synthetic_arrivals",
    "n_synthetic_extra", "n_train", "cumulative_real_acquired",
    "cumulative_synthetic", "cumulative_training_rows",
    "negative_eigenvalues_clipped", "largest_negative_relative",
]


def model_metrics(model: GaussianModel, directions: np.ndarray, coordinate: int) -> list[float]:
    d = len(model.mean)
    mean_mse = float(model.mean @ model.mean)
    trace = float(model.eigenvalues.sum())
    w2 = (mean_mse + trace + d - 2 * np.sqrt(model.eigenvalues).sum()) / d
    if w2 < -1e-10:
        raise FloatingPointError(f"Negative Wasserstein distance: {w2}")
    projected_means = directions @ model.mean
    projected_variances = np.sum((directions @ model.factor) ** 2, axis=1)
    means = np.append(projected_means, model.mean[coordinate])
    variances = np.append(projected_variances, model.covariance[coordinate, coordinate])
    o2, c2 = tail_probabilities(means, variances, 2)
    o3, c3 = tail_probabilities(means, variances, 3)
    q2, q3 = 2 * ndtr(-2.0), 2 * ndtr(-3.0)
    return [mean_mse, trace / d, max(0.0, float(w2)),
            float(o2[:-1].mean() / q2), float(o3[:-1].mean() / q3),
            float(c2[:-1].mean() / q2), float(c3[:-1].mean() / q3),
            float(means[-1]), float(variances[-1]),
            float(o2[-1] / q2), float(o3[-1] / q3),
            float(c2[-1] / q2), float(c3[-1] / q3)]


def fixed_directions(d: int, count: int = 20) -> np.ndarray:
    directions = np.random.default_rng(20261009).standard_normal((count, d))
    return directions / np.linalg.norm(directions, axis=1, keepdims=True)
