"""Exact moment theory for A2 E3/E4; no change to the original simulator.

q = E[||mu||^2]/d, s = E[tr(Sigma)]/d, with true distribution N(0, I).
`integrated` replaces ONLY the extra synthetic batch by conditional expectations.
The r unlabelled arrivals in E4 remain random, including their synthetic part.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gaussian_core import BatchStats, GaussianModel


def moment_transition(r: int, m: int, p: float = 1.0,
                      integrated: bool = False) -> tuple[np.ndarray, np.ndarray]:
    if r < 2 or m < 0 or not 0 <= p <= 1:
        raise ValueError("Require r >= 2, m >= 0, 0 <= p <= 1")
    n, real = r + m, r * p
    variance = r * p * (1 - p)
    a = 1 - real / n
    extra_noise_count = r - real if integrated else n - real
    matrix = np.array([[a*a + variance/(n*n), extra_noise_count/(n*n)],
                       [(real*(n-real)-variance)/(n*(n-1)), a]])
    offset = np.array([real/(n*n), real/n])
    return matrix, offset


def moment_curve(r: int, m: int, p: float, n0: int, generations: int,
                 integrated: bool = False, initial: tuple[float, float] | None = None) -> np.ndarray:
    matrix, offset = moment_transition(r, m, p, integrated)
    curve = np.empty((generations, 2))
    curve[0] = initial if initial is not None else (1/n0, 1.0)
    for t in range(1, generations):
        curve[t] = matrix @ curve[t-1] + offset
    return curve


def stationary_moments(r: int, m: int, p: float = 1.0,
                       integrated: bool = False) -> dict:
    matrix, offset = moment_transition(r, m, p, integrated)
    rho = float(np.max(np.abs(np.linalg.eigvals(matrix))))
    if p == 0 or rho >= 1:
        return {"stable": False, "spectral_radius": rho, "q": math.inf, "s": math.inf}
    n, real = r+m, r*p
    if integrated:
        variance = real*(1-p)
        coefficient = (n-real-1+p)/(n-1)
        denominator = n*n - (n-real)**2 - variance - (r-real)*coefficient
        q = r/denominator
        s = 1 + coefficient*q
    else:
        denominator = (2*real-1)*n - real*real + (1-p)*(1-real)
        q = (n-1)/denominator
        s = 1 + (n-real-1+p)/denominator
    direct = np.linalg.solve(np.eye(2)-matrix, offset)
    if not np.allclose([q, s], direct, rtol=2e-9, atol=1e-12):
        raise ArithmeticError("Closed form does not solve the exact recurrence")
    return {"stable": True, "spectral_radius": rho, "q": q, "s": s}


def cost_optimal_m(r: int, p: float, cost_per_sample: float) -> float:
    """Continuous optimum of stationary MEAN risk q(m) + cost*m, rp > 1."""
    real = r*p
    if real <= 1 or cost_per_sample <= 0:
        raise ValueError("This formula requires rp > 1 and positive cost")
    a = 2*real - 1
    b = -real*real + (1-p)*(1-real)
    c = (real-1)*(real-p)
    return max(0.0, (math.sqrt(c/cost_per_sample)-b)/a-r)


def mean_tolerance_m(r: int, p: float, tolerance: float = .05) -> int:
    """Smallest integer m with q(m)-q(infinity) <= tolerance*q(0), rp > 1."""
    real = r*p
    if real <= 1 or tolerance <= 0:
        raise ValueError("Require rp > 1 and positive tolerance")
    a = 2*real-1
    b = -real*real+(1-p)*(1-real)
    c = (real-1)*(real-p)
    baseline = stationary_moments(r, 0, p)["q"]
    boundary = (c/(a*tolerance*baseline)-b)/a-r
    answer = max(0, math.ceil(boundary-1e-10))
    return answer


def conditional_mean_risk(q: float, s: float, r: int, m: int, p: float = 1.0,
                          integrated: bool = False) -> float:
    matrix, offset = moment_transition(r, m, p, integrated)
    return float((matrix @ np.array([q, s])+offset)[0])


def oracle_arrival_weight(q: float, s: float, r: int, p: float = 1.0) -> float:
    """Best convex weight on the arrival mean, conditional MEAN risk only.

    This is an oracle expression: q involves the unknown true mean; p may be
    unknown in E4. It must not be used as a source-blind deployed selector.
    """
    noise = (p+(1-p)*s+p*(1-p)*q)/r
    denominator = p*p*q+noise
    return float(np.clip(p*q/denominator, 0, 1)) if denominator > 0 else 0.0


def integrated_update(previous: GaussianModel, arrivals: BatchStats,
                      m: int) -> tuple[np.ndarray, np.ndarray]:
    """E[(mu_next, Sigma_next) | previous model, observed arrival batch].

    Reads no p, no source labels, no initial training data and no old arrivals.
    Requires access to model parameters, rather than a samples-only black box.
    """
    r, n = arrivals.n, arrivals.n+m
    if r < 2 or m < 0:
        raise ValueError("Require r >= 2 and m >= 0")
    difference = arrivals.mean-previous.mean
    mean = (r*arrivals.mean+m*previous.mean)/n
    covariance = (arrivals.scatter/(n-1) + (m/n)*previous.covariance
                  + (r*m/(n*(n-1)))*np.outer(difference, difference))
    return mean, (covariance+covariance.T)/2


def w2_standard(mean: np.ndarray, covariance: np.ndarray) -> float:
    eigenvalues = np.linalg.eigvalsh((covariance+covariance.T)/2)
    if eigenvalues.min() < -1e-9:
        raise ValueError("Covariance must be positive semidefinite")
    return float((mean @ mean + np.sum((np.sqrt(np.maximum(eigenvalues, 0))-1)**2))/len(mean))
