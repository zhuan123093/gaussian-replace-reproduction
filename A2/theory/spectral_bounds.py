"""Rigorous spectral bounds for E3/E4, with target N(0, I).

All covariance losses are squared W2 divided by dimension. Covariance
estimation retains the original N-1 convention. No ridge is introduced.
"""
from __future__ import annotations

import math
import numpy as np

from gaussian_theory import integrated_update
from gaussian_core import BatchStats, GaussianModel


def h(x):
    """Scalar covariance contribution, including x=0."""
    return (np.sqrt(x) - 1)**2


def psd_eigenvalues(covariance: np.ndarray) -> np.ndarray:
    covariance = np.asarray(covariance, dtype=float)
    if (covariance.ndim != 2 or covariance.shape[0] != covariance.shape[1]
            or not covariance.shape[0] or not np.isfinite(covariance).all()):
        raise ValueError("Require a finite, nonempty square covariance")
    values = np.linalg.eigvalsh((covariance + covariance.T)/2)
    tolerance = 256*len(values)*np.finfo(float).eps*max(1., np.abs(values).max())
    if values.min() < -tolerance:
        raise ValueError("Covariance is not positive semidefinite")
    return np.maximum(values, 0.)


def trace_extrema_bounds(d: int, lower: float, upper: float,
                        trace: float) -> tuple[float, float]:
    """Sharp covariance-loss bounds with attained min/max and fixed trace.

    The lower optimizer has equal interior eigenvalues. The upper optimizer
    has endpoint eigenvalues and at most one interior eigenvalue.
    """
    if d < 1 or lower < 0 or upper < lower:
        raise ValueError("Require d >= 1 and 0 <= lower <= upper")
    tol = 1e-10*max(1., abs(trace), d*upper)
    if d == 1 or upper == lower:
        if abs(trace - d*lower) > tol or (d == 1 and upper != lower):
            raise ValueError("Inconsistent extrema/trace")
        return float(h(lower)), float(h(lower))
    remaining = d - 2
    interior_sum = trace - lower - upper
    if interior_sum < remaining*lower-tol or interior_sum > remaining*upper+tol:
        raise ValueError("Trace is incompatible with attained extrema")
    ends = float(h(lower) + h(upper))
    if not remaining:
        return ends/d, ends/d
    center = float(np.clip(interior_sum/remaining, lower, upper))
    lo = (ends + remaining*h(center))/d
    # Allocate excess trace to upper endpoints, then one fractional endpoint.
    z = float(np.clip((interior_sum-remaining*lower)/(upper-lower), 0, remaining))
    count = min(math.floor(z), remaining)
    fraction = z-count
    if count == remaining:
        hi = (ends + remaining*h(upper))/d
    else:
        hi = (ends + count*h(upper) + (remaining-count-1)*h(lower)
              + h(lower+fraction*(upper-lower)))/d
    return float(lo), float(hi)


def covariance_bounds(values: np.ndarray) -> dict[str, float]:
    """Compute independently usable bounds and the exact validation target.

    `extrema_*` use only min/max/d; `trace_*` also use trace; `frobenius_*`
    instead use tr((S-I)^2)/d. `combined_*` intersect the latter two bounds.
    """
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or values.min() < 0:
        raise ValueError("Require a nonempty finite nonnegative spectrum")
    d = len(values)
    lower, upper = float(values.min()), float(values.max())
    trace = float(values.sum())
    frobenius = float(np.mean((values-1)**2))
    frob_lower = frobenius/(1+math.sqrt(upper))**2
    frob_upper = frobenius/(1+math.sqrt(lower))**2
    if d == 1 or lower == upper:
        extrema_lower = extrema_upper = float(h(lower))
    else:
        ends = h(lower)+h(upper)
        extrema_lower = float((ends+(d-2)*h(np.clip(1., lower, upper)))/d)
        extrema_upper = float((ends+(d-2)*max(h(lower), h(upper)))/d)
    trace_lower, trace_upper = trace_extrema_bounds(d, lower, upper, trace)
    chord_upper = (trace/d+1-2*(trace/d+math.sqrt(lower*upper))
                   /(math.sqrt(lower)+math.sqrt(upper))) if upper else 1.
    return {"lambda_min": lower, "lambda_max": upper, "trace_per_dim": trace/d,
            "frobenius_per_dim": frobenius, "covariance_loss": float(h(values).mean()),
            "extrema_lower": extrema_lower, "extrema_upper": extrema_upper,
            "trace_lower": trace_lower, "trace_upper": trace_upper,
            "trace_chord_upper": float(chord_upper),
            "frobenius_lower": frob_lower, "frobenius_upper": frob_upper,
            "combined_lower": max(frob_lower, trace_lower),
            "combined_upper": min(frob_upper, trace_upper)}


def integrated_spectral_bounds(previous: GaussianModel, arrivals: BatchStats,
                               m: int) -> tuple[float, float]:
    """Weyl bounds for the conditional-expectation (RB) covariance update."""
    r, n = arrivals.n, arrivals.n+m
    if r < 2 or m < 0:
        raise ValueError("Require r >= 2, m >= 0")
    old = psd_eigenvalues(previous.covariance)
    new = psd_eigenvalues(arrivals.scatter/(r-1))
    gap = arrivals.mean-previous.mean
    lo = (r-1)/(n-1)*new[0]+m/n*old[0]
    hi = (r-1)/(n-1)*new[-1]+m/n*old[-1]+r*m/(n*(n-1))*(gap@gap)
    return float(lo), float(hi)


def conditional_sample_bounds(previous: GaussianModel, arrivals: BatchStats,
                              m: int, delta: float = .05) -> dict[str, float]:
    """Conditional probability >=1-delta: next spectrum and W2 lie in bounds.

    Condition on the previous model and the WHOLE observed arrival batch.
    Only the extra m samples must be Gaussian. No p or source labels are read.
    W2 bounds evaluate against the known theoretical target N(0,I); selecting
    m by that objective is an oracle operation if the target is unknown.
    """
    r, n = arrivals.n, arrivals.n+m
    if r < 2 or m < 0 or not 0 < delta < 1:
        raise ValueError("Require r >= 2, m >= 0, 0 < delta < 1")
    d = len(previous.mean)
    old = psd_eigenvalues(previous.covariance)
    new = psd_eigenvalues(arrivals.scatter/(r-1))
    rb_mean, _ = integrated_update(previous, arrivals, m)
    if m == 0:
        lo, hi, mean_radius = float(new[0]), float(new[-1]), 0.
    else:
        x = math.sqrt(2*math.log(3/delta))
        nu = m-1
        # With one observation, the synthetic within-batch scatter is zero.
        a = max(math.sqrt(nu)-math.sqrt(d)-x, 0.)**2 if nu else 0.
        b = (math.sqrt(nu)+math.sqrt(d)+x)**2 if nu else 0.
        noise = (math.sqrt(float(old.sum()))+x*math.sqrt(float(old[-1])))/math.sqrt(m)
        gap = float(np.linalg.norm(arrivals.mean-previous.mean))
        lo = ((r-1)*new[0]+old[0]*a)/(n-1)
        hi = ((r-1)*new[-1]+old[-1]*b+r*m/n*(gap+noise)**2)/(n-1)
        mean_radius = m/n*noise
    mean_center = float(np.linalg.norm(rb_mean))
    mean_lo = max(mean_center-mean_radius, 0.)**2/d
    mean_hi = (mean_center+mean_radius)**2/d
    # These are spectral envelopes, not necessarily attained extrema.
    cov_lo = float(h(np.clip(1., lo, hi)))
    cov_hi = float(max(h(lo), h(hi)))
    return {"lambda_lower": float(lo), "lambda_upper": float(hi),
            "mean_lower": mean_lo, "mean_upper": mean_hi,
            "w2_lower": mean_lo+cov_lo, "w2_upper": mean_hi+cov_hi,
            "delta": delta}
