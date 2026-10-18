"""Parameter memory for Gaussian learners; no access to the true distribution.

The initial anchor is a fitted model, not the true Gaussian. Moment anchoring
projects a two-Gaussian mixture back to a Gaussian; it preserves its first two
moments, not all its event probabilities.
"""
from __future__ import annotations

import numpy as np

from gaussian_core import BatchStats, GaussianModel
from theory.gaussian_theory import integrated_update


def model_from_parameters(mean: np.ndarray, covariance: np.ndarray) -> GaussianModel:
    mean = np.asarray(mean, dtype=float)
    covariance = np.asarray(covariance, dtype=float)
    if mean.ndim != 1 or covariance.shape != (len(mean), len(mean)):
        raise ValueError("Incompatible parameter dimensions")
    covariance = (covariance + covariance.T) / 2
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("Parameters must be finite")
    values, vectors = np.linalg.eigh(covariance)
    scale = max(float(np.max(np.abs(values))), np.finfo(float).tiny)
    negative = max(0.0, -float(values[0])) / scale
    if negative > 128 * len(mean) * np.finfo(float).eps:
        raise ValueError("Covariance must be positive semidefinite")
    clipped = int(np.count_nonzero(values < 0))
    values = np.maximum(values, 0)
    positive = values > 0
    factor = vectors[:, positive] * np.sqrt(values[positive])
    if clipped:
        covariance = factor @ factor.T
    return GaussianModel(mean.copy(), covariance, values, factor, clipped, negative)


def initial_sufficient_statistics(initial: GaussianModel, n0: int) -> BatchStats:
    """Exact original fitting statistics only with known n0 and N-1 fitting.

    This is not valid for an arbitrary pretrained/regularized model with an
    invented effective sample count.
    """
    if n0 < 2:
        raise ValueError("Need the actual original sample count >= 2")
    return BatchStats(n0, initial.mean.copy(), (n0 - 1) * initial.covariance)


def anchor_parameters(mean: np.ndarray, covariance: np.ndarray,
                      initial: GaussianModel, weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Moment projection of weight*Q_initial + (1-weight)*Q_candidate.

    covariance_out >= weight*covariance_initial in PSD order. Repeated pure
    anchoring converges to the initial model, since both mean and raw second
    moment contract by 1-weight. A fresh noisy candidate at every step does not
    have that convergence guarantee.
    """
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("Anchor weight must lie in [0, 1]")
    delta = mean - initial.mean
    updated_mean = (1 - weight) * mean + weight * initial.mean
    updated_covariance = ((1 - weight) * covariance + weight * initial.covariance
                          + weight * (1 - weight) * np.outer(delta, delta))
    return updated_mean, (updated_covariance + updated_covariance.T) / 2


def parameter_update(previous: GaussianModel, arrivals: BatchStats, m: int,
                     initial: GaussianModel | None = None,
                     anchor_weight: float = 0) -> GaussianModel:
    """Integrate extra replay randomness, optionally then anchor initial moments.

    No true mean/covariance, source labels, or mixture fraction are arguments.
    m is a memory weight here; no extra samples are actually generated.
    """
    mean, covariance = integrated_update(previous, arrivals, m)
    if anchor_weight != 0:
        if initial is None:
            raise ValueError("An initial model is required for anchoring")
        mean, covariance = anchor_parameters(mean, covariance, initial, anchor_weight)
    return model_from_parameters(mean, covariance)
