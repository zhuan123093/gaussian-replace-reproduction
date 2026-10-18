"""Paired pilot: sampled replay, parameter integration and fixed initial anchor.

Standalone outputs; does not change the original A2 experiments or reports.
E4 shares exogenous innovations, not realized batches, between policies: the
synthetic arrival component depends on each policy's own previous model.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_name] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy.special import ndtr
from threadpoolctl import threadpool_limits

from gaussian_core import (batch_from_array, fit_gaussian, fixed_directions,
                           gaussian_batch, merge_batches, mixture_arrival,
                           tail_probabilities)
from memory.parameter_memory import parameter_update

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["w2_true", "w2_initial", "mean_error_per_dim", "trace_per_dim",
           "tail3_true", "tail3_initial", "min_variance_relative_initial"]
METHODS = {"frozen": 1., "random_replay": 0., "rb": 0.,
           "rb_anchor_0p1": .1, "rb_anchor_0p5": .5}


def evaluate(model, initial, directions, sqrt_initial, invsqrt_initial):
    d = len(model.mean)
    w2_true = (model.mean@model.mean + np.sum((np.sqrt(model.eigenvalues)-1)**2))/d
    middle = sqrt_initial @ model.covariance @ sqrt_initial
    values = np.maximum(np.linalg.eigvalsh((middle+middle.T)/2), 0)
    displacement = model.mean-initial.mean
    w2_initial = (displacement@displacement + np.trace(model.covariance)
                  + np.trace(initial.covariance) - 2*np.sqrt(values).sum())/d
    if w2_initial < -1e-10:
        raise ArithmeticError("Negative Wasserstein distance")
    proj_var = np.sum((directions @ model.factor)**2, axis=1)
    reference_var = np.sum((directions @ initial.factor)**2, axis=1)
    tail_true, _ = tail_probabilities(directions @ model.mean, proj_var, 3)
    # Tail events are fixed at mu_initial +/- 3*sd_initial in each direction.
    tail_initial, _ = tail_probabilities((directions @ displacement)/np.sqrt(reference_var),
                                        proj_var/reference_var, 3)
    relative = invsqrt_initial @ model.covariance @ invsqrt_initial
    minimum = np.linalg.eigvalsh((relative+relative.T)/2)[0]
    return [w2_true, max(0., w2_initial), model.mean@model.mean/d,
            model.eigenvalues.sum()/d, tail_true.mean()/(2*ndtr(-3)),
            tail_initial.mean()/(2*ndtr(-3)), minimum]


def run_one(job):
    config, seed, directory, fingerprint = job
    path = Path(directory) / f"seed_{seed:06d}.npz"
    d, n0, r, m, p, total, method = [config[k] for k in
        ("d", "n0", "r", "m", "p", "generations", "method")]
    saved_at = np.unique(np.r_[1, np.arange(25, total+1, 25), total])
    if path.exists():
        with np.load(path) as old:
            if (str(old["fingerprint"]) == fingerprint
                    and old["metrics"].shape == (len(saved_at), len(METRICS))
                    and np.isfinite(old["metrics"]).all()):
                return "reused"
    with threadpool_limits(limits=1):
        initial_rng, fresh_rng, extra_rng, fake_rng, mask_rng = [
            np.random.default_rng(np.random.SeedSequence([seed, tag, d])) for tag in (1, 2, 3, 4, 5)]
        initial_batch = batch_from_array(initial_rng.standard_normal((n0, d)))
        initial = fit_gaussian(initial_batch)
        del initial_batch
        # Learners keep only model parameters; the original samples are gone.
        model = initial
        eigenvalues, eigenvectors = np.linalg.eigh(initial.covariance)
        if eigenvalues.min() <= 0:
            raise ValueError("This evaluator requires a full-rank initial model")
        sqrt_initial = (eigenvectors*np.sqrt(eigenvalues)) @ eigenvectors.T
        invsqrt_initial = (eigenvectors/np.sqrt(eigenvalues)) @ eigenvectors.T
        directions = fixed_directions(d, 30)
        records = []
        for generation in range(1, total+1):
            if generation > 1 and method != "frozen":
                fresh = fresh_rng.standard_normal((r, d))
                # p and hidden labels live only in this simulated environment.
                arrivals, _ = mixture_arrival(model, fresh, p, mask_rng, fake_rng)
                if method == "random_replay":
                    model = fit_gaussian(merge_batches(arrivals, gaussian_batch(model, m, extra_rng)))
                else:
                    model = parameter_update(model, arrivals, m, initial, METHODS[method])
            if generation in saved_at:
                row = evaluate(model, initial, directions, sqrt_initial, invsqrt_initial)
                if METHODS[method] > 0 and row[-1] < METHODS[method]-1e-9:
                    raise ArithmeticError("Initial covariance floor violated")
                records.append(row)
        metrics = np.asarray(records)
        if not np.isfinite(metrics).all():
            raise ArithmeticError("Nonfinite trajectory")
        with path.with_suffix(".tmp").open("wb") as stream:
            np.savez_compressed(stream, metrics=metrics, generations=saved_at,
                                final_mean=model.mean, final_covariance=model.covariance,
                                initial_mean=initial.mean, initial_covariance=initial.covariance,
                                seed=seed, fingerprint=fingerprint)
        path.with_suffix(".tmp").replace(path)
    return "computed"


def summarize(output, configs, seeds, fingerprint):
    scores, curves = [], []
    for config, directory in configs:
        samples = []
        for seed in seeds:
            with np.load(directory/f"seed_{seed:06d}.npz") as saved:
                if str(saved["fingerprint"]) != fingerprint:
                    raise ArithmeticError("Stale result")
                metrics, generations = saved["metrics"], saved["generations"]
            late = metrics[generations > .8*config["generations"]].mean(axis=0)
            scores.append({**config, "seed": seed,
                           **{f"late_{name}": float(value) for name, value in zip(METRICS, late)}})
            samples.append(metrics)
        samples = np.asarray(samples)
        for i, generation in enumerate(generations):
            curves.append({**config, "generation": int(generation),
                           **{name: float(value) for name, value in zip(METRICS, samples[:, i].mean(axis=0))}})
    scores = pd.DataFrame(scores)
    scores.to_csv(output/"seed_scores.csv", index=False)
    pd.DataFrame(curves).to_csv(output/"curves.csv", index=False)
    rng = np.random.default_rng(20261011)
    bootstrap_indices = rng.integers(0, len(seeds), size=(4000, len(seeds)))
    summary, paired = [], []
    for (n0, p, method), group in scores.groupby(["n0", "p", "method"]):
        group = group.sort_values("seed")
        row = {"n0": n0, "p": p, "method": method, "seeds": len(group)}
        for metric in METRICS:
            values = group[f"late_{metric}"].to_numpy()
            low, high = np.quantile(values[bootstrap_indices].mean(axis=1), [.025, .975])
            row.update({metric: values.mean(), f"{metric}_low": low, f"{metric}_high": high})
        summary.append(row)
        for baseline_method in ("random_replay", "frozen", "rb"):
            if method == baseline_method:
                continue
            baseline = scores[(scores.n0 == n0) & (scores.p == p)
                              & (scores.method == baseline_method)].sort_values("seed")
            differences = group.late_w2_true.to_numpy()-baseline.late_w2_true.to_numpy()
            low, high = np.quantile(differences[bootstrap_indices].mean(axis=1), [.025, .975])
            paired.append({"n0": n0, "p": p, "method": method, "baseline": baseline_method,
                           "w2_difference": differences.mean(), "low": low, "high": high})
    frame = pd.DataFrame(summary)
    frame.to_csv(output/"summary.csv", index=False)
    pd.DataFrame(paired).to_csv(output/"paired_comparisons.csv", index=False)
    print(frame[["n0", "p", "method", "w2_true", "w2_initial", "tail3_initial",
                 "min_variance_relative_initial"]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=ROOT/"results"/"parameter_memory")
    args = parser.parse_args()
    if args.seeds < 2 or args.generations < 25 or args.workers < 1:
        parser.error("Need >=2 seeds, >=25 generations, >=1 worker")
    args.output.mkdir(parents=True, exist_ok=True)
    sources = [Path(__file__), ROOT/"memory"/"parameter_memory.py", ROOT/"gaussian_core.py",
               ROOT/"theory"/"gaussian_theory.py"]
    fingerprint = hashlib.sha256(b"".join(p.read_bytes() for p in sources)).hexdigest()
    seeds = list(range(30000, 30000+args.seeds))
    configs, jobs = [], []
    for n0 in (200, 500):
        for p in (1., .5):
            for method in METHODS:
                config = {"d": 100, "n0": n0, "r": 100, "m": 500, "p": p,
                          "generations": args.generations, "method": method}
                directory = args.output/f"n0_{n0}_p_{p:g}_{method}"
                directory.mkdir(exist_ok=True)
                configs.append((config, directory))
                jobs.extend((config, seed, str(directory), fingerprint) for seed in seeds)
    metadata = {"study": "paired pilot, not an optimal-weight search", "metrics": METRICS,
                "configs": [c for c, _ in configs], "seeds": seeds, "fingerprint": fingerprint,
                "actual_extra_samples_per_update": {m: 500 if m == "random_replay" else 0 for m in METHODS},
                "late_score": "within-seed average of 25-generation checkpoints in final 20%",
                "tail3_initial": "mean over 30 fixed directions of P_Q(|v'(X-mu0)|>3 sqrt(v'S0v))/P_N(|Z|>3)",
                "intervals": "4000 bootstrap resamples of whole independent seeds",
                "constraints": "Learners receive no truth, p, labels or historic observations"}
    (args.output/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    started = time.perf_counter()
    print(f"Running {len(jobs)} trajectories with {args.workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for completed, future in enumerate(as_completed(futures), 1):
            future.result()
            if completed % 30 == 0 or completed == len(jobs):
                print(f"Completed {completed}/{len(jobs)} in {time.perf_counter()-started:.1f}s", flush=True)
    summarize(args.output, configs, seeds, fingerprint)
    status = {"state": "complete", "trajectories": len(jobs), "all_saved_metrics_finite": True,
              "anchor_covariance_floor_checks": "passed at every saved checkpoint",
              "elapsed_seconds": time.perf_counter()-started, "fingerprint": fingerprint}
    (args.output/"status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
