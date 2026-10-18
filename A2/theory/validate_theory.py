"""Check exact algebra, existing A2 simulations and independent one-step draws."""
from __future__ import annotations

import json
from pathlib import Path
import platform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sympy as sp
from threadpoolctl import threadpool_limits

from gaussian_theory import (moment_transition, moment_curve, stationary_moments,
    mean_tolerance_m, integrated_update, w2_standard, cost_optimal_m)
from gaussian_core import (GaussianModel, batch_from_array, fit_gaussian,
                           gaussian_batch, merge_batches, mixture_arrival)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
EXPERIMENT_RESULTS = ROOT.parent / "results"


def symbolic_checks() -> dict:
    n, r, p = sp.symbols("N r p", positive=True)
    real, variance = r*p, r*p*(1-p)
    a = 1-real/n
    h = (real*(n-real)-variance)/(n*(n-1))
    denominator = (2*real-1)*n-real**2+(1-p)*(1-real)
    q = (n-1)/denominator
    s = 1+(n-real-1+p)/denominator
    mean_residual = (a*a+variance/n**2)*q+(real+(n-real)*s)/n**2-q
    trace_residual = real/n+a*s+h*q-s
    derivative_residual = sp.diff(q, n)+(real-1)*(real-p)/denominator**2
    mean_e3_residual = q.subs(p, 1)-(n-1)/((2*r-1)*n-r*r)
    determinant_residual = ((1-a*a-variance/n**2)*(1-a)
                            - (n-real)*h/n**2 - real*denominator/(n*n*(n-1)))
    named = {"stationary_mean": mean_residual, "stationary_trace": trace_residual,
             "monotonic_derivative": derivative_residual, "E3_special_case": mean_e3_residual,
             "stability_determinant": determinant_residual}
    for name, residual in named.items():
        assert sp.factor(residual) == 0, (name, sp.factor(residual))
    for real_probability in (.25, .5, .75, 1):
        count = mean_tolerance_m(100, real_probability)
        baseline = stationary_moments(100, 0, real_probability)["q"]
        limit = 1/(200*real_probability-1)
        assert stationary_moments(100, count, real_probability)["q"]-limit <= .05*baseline+1e-14
        if count:
            assert stationary_moments(100, count-1, real_probability)["q"]-limit > .05*baseline-1e-14
        for cost in (1e-7, 1e-6, 1e-5):
            optimum = cost_optimal_m(100, real_probability, cost)
            objective = lambda m: stationary_moments(100, m, real_probability)["q"]+cost*m
            for alternative in (0, max(0, optimum-1), optimum+1, optimum+100):
                assert objective(optimum) <= objective(alternative)+1e-12
    for m in (0, 1, 100, 2000):
        assert np.isclose(stationary_moments(100, m, 1, integrated=True)["q"], 1/(100+2*m))
    return {name: "exact symbolic zero" for name in named}


def compare_existing() -> pd.DataFrame:
    records = []
    for metadata in sorted((EXPERIMENT_RESULTS/"02_validation").glob("*/*/config.json")):
        meta = json.loads(metadata.read_text(encoding="utf-8"))
        config = meta["config"]
        if config["mode"] not in {"current_fresh", "unlabelled_mixture"}:
            continue
        r, p, m = config["r"], config["real_fraction"], round(config["r"]*config["k"])
        curve = moment_curve(r, m, p, config["n0"], config["iterations"])
        prediction = curve[int(.8*len(curve)):].mean(axis=0)
        scores = pd.read_csv(metadata.parent/"seed_scores.csv")
        for metric, predicted, observed in (
                ("mean_error_per_dim", prediction[0], scores.late_mean_mse.to_numpy()/config["d"]),
                ("covariance_trace_per_dim", prediction[1], scores.late_trace_ratio.to_numpy())):
            standard_error = float(observed.std(ddof=1)/np.sqrt(len(observed)))
            records.append({"scenario": "E3" if p == 1 else "E4", "p": p,
                            "r": r, "m": m, "k": config["k"], "metric": metric,
                            "theory_late": predicted, "simulation_late": observed.mean(),
                            "simulation_standard_error": standard_error,
                            "z_discrepancy": (observed.mean()-predicted)/standard_error if standard_error else 0,
                            "n_seeds": len(observed), "source_case": str(metadata.parent)})
    frame = pd.DataFrame(records)
    frame.to_csv(OUT/"moment_comparison.csv", index=False)
    for metric, ylabel, filename in (
            ("mean_error_per_dim", "E[||mu|| squared] / d", "01_exact_mean_vs_simulation.png"),
            ("covariance_trace_per_dim", "E[trace(Sigma)] / d", "02_exact_trace_vs_simulation.png")):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
        for ax, p in zip(axes.flat, (1, .75, .5, .25)):
            group = frame[(frame.p == p) & (frame.metric == metric)].sort_values("m")
            grid = np.unique(np.rint(np.linspace(0, 2000, 201)).astype(int))
            index = 0 if metric == "mean_error_per_dim" else 1
            predicted = [moment_curve(100, int(m), p, 500, 1000)[800:, index].mean() for m in grid]
            ax.plot(grid/100, predicted, label="Exact finite-generation expectation")
            ax.errorbar(group.k, group.simulation_late,
                        yerr=1.96*group.simulation_standard_error, marker="o", ls="none", capsize=3,
                        label="Saved 100-seed simulation, mean +/- 1.96 SE")
            ax.set(title="E3" if p == 1 else f"E4: hidden p={p:g}", xlabel="Extra ratio k=m/r", ylabel=ylabel)
            ax.grid(alpha=.2)
            ax.legend(fontsize=7)
        fig.suptitle("Exact recurrences, N-1 covariance; generations 801-1000, r=100")
        fig.tight_layout()
        fig.savefig(OUT/filename, dpi=160, bbox_inches="tight")
        plt.close(fig)
    return frame


def monte_carlo_checks(repetitions: int = 12000) -> pd.DataFrame:
    rng = np.random.default_rng(20261010)
    covariance = np.array([[1.4, .25, .1], [.25, .7, -.08], [.1, -.08, .9]])
    mean = np.array([.4, -.2, .1])
    values, vectors = np.linalg.eigh(covariance)
    previous = GaussianModel(mean, covariance, values, vectors*np.sqrt(values))
    state = np.array([mean@mean/3, np.trace(covariance)/3])
    r = 7
    rows = []
    for p in (1.0, .4, 0.0):
        for m in (0, 15):
            outcomes = np.empty((repetitions, 2, 3))
            for rep in range(repetitions):
                fresh = rng.standard_normal((r, 3))
                arrivals, _ = mixture_arrival(previous, fresh, p, rng, rng)
                random_model = fit_gaussian(merge_batches(arrivals, gaussian_batch(previous, m, rng)))
                integrated_mean, integrated_covariance = integrated_update(previous, arrivals, m)
                for algorithm, (mu, cov) in enumerate(((random_model.mean, random_model.covariance),
                                                       (integrated_mean, integrated_covariance))):
                    outcomes[rep, algorithm] = [mu@mu/3, np.trace(cov)/3, w2_standard(mu, cov)]
            for algorithm, integrated in enumerate((False, True)):
                matrix, offset = moment_transition(r, m, p, integrated)
                prediction = matrix @ state+offset
                for i, metric in enumerate(("q", "s")):
                    observed = outcomes[:, algorithm, i]
                    error = float(observed.std(ddof=1)/np.sqrt(repetitions))
                    z = (observed.mean()-prediction[i])/error
                    assert abs(z) < 6, (p, m, integrated, metric, z)
                    rows.append({"p": p, "r": r, "m": m, "integrated": integrated,
                                 "metric": metric, "prediction": prediction[i], "observed": observed.mean(),
                                 "standard_error": error, "z_discrepancy": z})
            # Numerical corroboration of the proved one-step Jensen guarantee.
            difference = outcomes[:, 1, 2]-outcomes[:, 0, 2]
            se = difference.std(ddof=1)/np.sqrt(repetitions)
            assert difference.mean() <= 6*se+1e-12
            rows.append({"p": p, "r": r, "m": m, "integrated": True,
                         "metric": "paired_W2_change_vs_random", "prediction": np.nan,
                         "observed": difference.mean(), "standard_error": se, "z_discrepancy": np.nan})
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT/"one_step_checks.csv", index=False)
    return frame


def guidance_table() -> pd.DataFrame:
    records = []
    for p in (1, .75, .5, .25):
        for k in (0, 1, 5, 10, 20):
            random = stationary_moments(100, round(100*k), p)
            integrated = stationary_moments(100, round(100*k), p, integrated=True)
            records.append({"p": p, "r": 100, "k": k, "m": round(100*k),
                            "random_q_limit": random["q"], "random_s_limit": random["s"],
                            "integrated_q_limit": integrated["q"], "integrated_s_limit": integrated["s"],
                            "random_q_limit_m_infinity": 1/(200*p-1),
                            "mean_only_5_percent_baseline_count": mean_tolerance_m(100, p)})
    result = pd.DataFrame(records)
    result.to_csv(OUT/"theoretical_guidance.csv", index=False)
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        symbolic = symbolic_checks()
        print("Exact symbolic identities: passed", flush=True)
        comparison = compare_existing()
        print("Existing simulation comparison saved", flush=True)
        mc = monte_carlo_checks()
        guidance_table()
    main_comparison = comparison[comparison.p > 0]
    report = {"symbolic_checks": symbolic, "one_step_repetitions_per_condition": 12000,
              "one_step_conditions": 6,
              "one_step_max_abs_z": float(mc.z_discrepancy.abs().max()),
              "existing_positive_p_max_abs_z": float(main_comparison.z_discrepancy.abs().max()),
              "existing_p0_note": "p=0 is not stationary. Rare trajectories can dominate moments; see CSV.",
              "all_engineering_checks_passed": True, "python": platform.python_version(),
              "numpy": np.__version__, "sympy": sp.__version__}
    (OUT/"checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
