"""Check spectral theorems, saved A2 snapshots, and one-step probability bounds."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from spectral_bounds import (h, psd_eigenvalues, covariance_bounds,
    trace_extrema_bounds, integrated_spectral_bounds, conditional_sample_bounds)
from gaussian_theory import integrated_update, w2_standard
from gaussian_core import (GaussianModel, gaussian_batch, mixture_arrival,
                           merge_batches, fit_gaussian)
from run_experiments import simulation_hash

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "spectral"


def assert_bounds(row: dict) -> None:
    actual = row["covariance_loss"]
    tol = 2e-9*max(1., actual, row["frobenius_per_dim"])
    for name in ("extrema", "trace", "frobenius", "combined"):
        assert row[name+"_lower"] <= actual+tol, (name, row)
        assert actual <= row[name+"_upper"]+tol, (name, row)
    assert row["trace_upper"] <= row["trace_chord_upper"]+tol, row


def algebra_checks() -> dict:
    rng = np.random.default_rng(7102026)
    spectra = [np.array([0.]), np.array([1.]), np.array([4.]),
               np.zeros(100), np.ones(100), np.full(10, 2.), np.array([0., 2.]),
               np.array([.25, 2., .75, 1.]), np.array([.25, 2., .25, 1.5])]
    for d in (1, 2, 3, 10, 100):
        for _ in range(300):
            values = np.exp(rng.normal(0, 3, d))
            values[rng.random(d) < .2] = 0
            spectra.append(values)
    sharp_checks = 0
    for values in spectra:
        row = covariance_bounds(values)
        assert_bounds(row)
        d, lo, hi, total = len(values), values.min(), values.max(), values.sum()
        if d <= 2 or lo == hi:
            continue
        remaining = d-2
        center = (total-lo-hi)/remaining
        lower_optimizer = np.r_[lo, hi, np.repeat(center, remaining)]
        excess = np.clip((total-lo-hi-remaining*lo)/(hi-lo), 0, remaining)
        count = min(int(np.floor(excess)), remaining)
        if count == remaining:
            interior = np.repeat(hi, remaining)
        else:
            interior = np.r_[np.repeat(hi, count),
                             lo+(excess-count)*(hi-lo),
                             np.repeat(lo, remaining-count-1)]
        upper_optimizer = np.r_[lo, hi, interior]
        predicted = trace_extrema_bounds(d, float(lo), float(hi), float(total))
        assert np.allclose(predicted, [h(lower_optimizer).mean(), h(upper_optimizer).mean()],
                           rtol=1e-9, atol=1e-9)
        assert np.isclose(upper_optimizer.sum(), total, rtol=1e-10, atol=1e-10)
        sharp_checks += 1
    # Same min, max and trace do not fix the loss when interior spectra differ.
    counterexample = [covariance_bounds(spectra[i]) for i in (7, 8)]
    assert np.isclose(counterexample[0]["trace_per_dim"], counterexample[1]["trace_per_dim"])
    assert not np.isclose(counterexample[0]["covariance_loss"], counterexample[1]["covariance_loss"])
    # General SPD target: transport through target^(1/2) bounds the Bures term.
    for _ in range(100):
        d = 7
        basis, _ = np.linalg.qr(rng.normal(size=(d, d)))
        target_eig = np.exp(rng.normal(size=d))
        root = (basis*np.sqrt(target_eig))@basis.T
        target = root@root
        a = rng.normal(size=(d, d))
        white = a@a.T/d
        covariance = root@white@root
        bures = (np.trace(target)+np.trace(covariance)
                 -2*np.sqrt(psd_eigenvalues(root@covariance@root)).sum())/d
        white_loss = h(psd_eigenvalues(white)).mean()
        assert target_eig.min()*white_loss <= bures+1e-8
        assert bures <= target_eig.max()*white_loss+1e-8
    return {"spectra_checked": len(spectra), "sharp_endpoint_constructions": sharp_checks,
            "general_SPD_target_checks": 100,
            "same_extrema_and_trace_counterexample_losses":
                [row["covariance_loss"] for row in counterexample]}


def saved_snapshot_checks() -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    original_hash = simulation_hash()
    for metadata in sorted((ROOT.parent/"results"/"02_validation").glob("*/*/config.json")):
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        config = meta["config"]
        if config["mode"] not in {"current_fresh", "unlabelled_mixture"}:
            continue
        assert meta["code_sha256"] == original_hash
        seed = meta["seeds"][0]
        with np.load(metadata.parent/"seeds"/f"seed_{seed:06d}.npz") as saved:
            assert str(saved["fingerprint"]) == original_hash
            for generation, mean, covariance in zip(saved["snapshot_generations"],
                    saved["snapshot_means"], saved["snapshot_covariances"]):
                result = covariance_bounds(psd_eigenvalues(covariance))
                assert_bounds(result)
                mean_loss = float(mean@mean/len(mean))
                result.update({"scenario": "E3" if config["mode"] == "current_fresh" else "E4",
                    "p": config["real_fraction"], "k": config["k"], "m": round(config["r"]*config["k"]),
                    "r": config["r"], "d": config["d"], "seed": seed, "generation": int(generation),
                    "mean_loss": mean_loss, "w2_loss": mean_loss+result["covariance_loss"],
                    "w2_lower": mean_loss+result["combined_lower"],
                    "w2_upper": mean_loss+result["combined_upper"],
                    "source_case": metadata.parent.relative_to(ROOT.parent).as_posix()})
                records.append(result)
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/"snapshot_bounds.csv", index=False)
    final = frame[(frame.generation == 1000) & (frame.p > 0)].copy()
    final.to_csv(OUT/"generation_1000_bounds.csv", index=False)
    pairs = []
    for p, group in final.groupby("p"):
        for a in group.to_dict("records"):
            for b in group.to_dict("records"):
                if a["k"] == b["k"]:
                    continue
                pairs.append({"p": p, "k_a": a["k"], "k_b": b["k"],
                    "upper_a": a["w2_upper"], "lower_b": b["w2_lower"],
                    "certified_a_better": a["w2_upper"] < b["w2_lower"],
                    "scope": "same saved seed, generation 1000; not expected or stationary risk"})
    pair_frame = pd.DataFrame(pairs)
    pair_frame.to_csv(OUT/"snapshot_pair_certificates.csv", index=False)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5), sharex=True)
    for ax, p in zip(axes.flat, (1., .75, .5, .25)):
        data = final[final.p == p].sort_values("k")
        ax.errorbar(data.k, data.covariance_loss,
            yerr=np.maximum(np.vstack([data.covariance_loss-data.combined_lower,
                                       data.combined_upper-data.covariance_loss]), 0),
            fmt="o-", capsize=5, label="Actual loss + rigorous interval")
        ax.set(title="E3" if p == 1 else f"E4: p={p:g}",
               xlabel="Extra ratio k=m/r", ylabel="Covariance W2 squared / d")
        ax.set_xticks([0, 5, 10, 20])
        ax.grid(alpha=.25)
        ax.legend(fontsize=8)
    fig.suptitle("Spectral bounds: preselected seed 10000, generation 1000 (not an average)")
    fig.tight_layout()
    fig.savefig(OUT/"01_covariance_intervals.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    for p in (1., .75, .5, .25):
        data = final[final.p == p].sort_values("k")
        label = "E3" if p == 1 else f"E4: p={p:g}"
        for ax, field in zip(axes, ("lambda_min", "lambda_max")):
            ax.plot(data.k, data[field], "o-", label=label)
            ax.set(xlabel="Extra ratio k=m/r", ylabel=field)
            ax.set_xticks([0, 5, 10, 20])
            ax.grid(alpha=.25)
            ax.legend(fontsize=8)
    fig.suptitle("Extreme eigenvalues: preselected seed 10000, generation 1000")
    fig.tight_layout()
    fig.savefig(OUT/"02_extreme_eigenvalues.png", dpi=160)
    plt.close(fig)
    return frame, pair_frame


def probability_checks(repetitions: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(80261010)
    d, r, delta = 8, 12, .05
    rows = []
    for singular in (False, True):
        basis, _ = np.linalg.qr(rng.normal(size=(d, d)))
        values = np.linspace(.3, 2., d)
        if singular:
            values[:3] = 0
        factor = basis*np.sqrt(values)
        covariance = factor@factor.T
        previous = GaussianModel(np.linspace(-.3, .4, d), covariance, values, factor)
        for p in (1., .4):
            arrivals, _ = mixture_arrival(previous, rng.normal(size=(r, d)), p, rng, rng)
            for m in (0, 1, 5, 40):
                bounds = conditional_sample_bounds(previous, arrivals, m, delta)
                rb_lo, rb_hi = integrated_spectral_bounds(previous, arrivals, m)
                _, rb_covariance = integrated_update(previous, arrivals, m)
                rb_values = psd_eigenvalues(rb_covariance)
                assert rb_lo <= rb_values.min()+1e-9
                assert rb_values.max() <= rb_hi+1e-9
                failed = 0
                risks = []
                for _ in range(repetitions):
                    model = fit_gaussian(merge_batches(arrivals, gaussian_batch(previous, m, rng)))
                    loss = w2_standard(model.mean, model.covariance)
                    mean_loss = model.mean@model.mean/d
                    failures = (model.eigenvalues.min() < bounds["lambda_lower"]-1e-9
                        or model.eigenvalues.max() > bounds["lambda_upper"]+1e-9
                        or loss < bounds["w2_lower"]-1e-9 or loss > bounds["w2_upper"]+1e-9
                        or mean_loss < bounds["mean_lower"]-1e-9
                        or mean_loss > bounds["mean_upper"]+1e-9)
                    failed += failures
                    risks.append(loss)
                # A numerical diagnostic, not the proof of the probability theorem.
                assert failed <= repetitions*delta+6*np.sqrt(repetitions*delta*(1-delta))
                rows.append({"singular_previous": singular, "environment_p": p,
                    "d": d, "r": r, "m": m, "repetitions": repetitions,
                    "violations": int(failed), "observed_coverage": 1-failed/repetitions,
                    "guaranteed_coverage": 1-delta, "mean_observed_w2": np.mean(risks),
                    "rb_lambda_lower": rb_lo, "rb_lambda_upper": rb_hi, **bounds})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"one_step_probability_checks.csv", index=False)
    return frame


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        algebra = algebra_checks()
        snapshots, pairs = saved_snapshot_checks()
        print("Deterministic bounds and saved snapshots: passed", flush=True)
        probabilities = probability_checks()
    report = {**algebra, "saved_snapshots_checked": len(snapshots),
        "saved_configurations": int(snapshots.source_case.nunique()),
        "saved_spectral_seeds": snapshots.seed.unique().tolist(),
        "saved_snapshot_generations": sorted(snapshots.generation.unique().tolist()),
        "positive_p_final_ordered_pairs": len(pairs),
        "positive_p_final_certified_pairs": int(pairs.certified_a_better.sum()),
        "single_step_conditions": len(probabilities),
        "single_step_draws": int(probabilities.repetitions.sum()),
        "single_step_violations": int(probabilities.violations.sum()),
        "original_simulation_sha256": simulation_hash(),
        "all_checks_passed": True,
        "interpretation": "Saved spectral checks use only the preselected first seed. Probability checks corroborate, not prove, the theorem; broad intervals can have overcoverage."}
    (OUT/"checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
