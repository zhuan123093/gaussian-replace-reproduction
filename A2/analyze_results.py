"""Aggregate independent trajectories, select candidates, and produce readable results."""
from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from run_experiments import (ALL_METRICS, ROOT, RESULTS, SCENARIOS, Experiment,
                             case_path, json_write, simulation_hash)


LABELS = {
    "replace": "E0: Replace", "fixed_initial": "E1: Keep initial data",
    "cumulative_real": "E2: Keep all real data", "current_fresh": "E3: Current real batch only",
    "unlabelled_mixture": "E4: Unlabelled arrivals", "frozen": "Frozen initial model",
}
CHINESE = {
    "replace": "E0 完全替换", "fixed_initial": "E1 保留初始样本",
    "cumulative_real": "E2 累计真实样本", "current_fresh": "E3 仅当轮真实样本",
    "unlabelled_mixture": "E4 来源不可区分", "frozen": "冻结初始模型",
}
CURVE_METRICS = ["mean_mse", "trace_ratio", "w2_per_dim", "tail2_original", "tail3_original",
                 "tail2_centered", "tail3_centered", "coordinate_mean", "coordinate_variance"]
plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 160, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})


def records_for_stage(stage: str) -> list[dict]:
    records = []
    for meta_path in sorted((RESULTS / stage).glob("*/*/config.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        directory = meta_path.parent
        score_path = directory / "seed_scores.csv"
        if not score_path.exists():
            continue
        scores = pd.read_csv(score_path)
        row = dict(meta["config"])
        row.update({"stage": stage, "case_id": meta["case_id"], "directory": str(directory),
                    "n_seeds": len(scores), "late_w2_mean": scores.late_w2.mean(),
                    "late_w2_median": scores.late_w2.median(),
                    "late_w2_ci_low": bootstrap_mean_ci(scores.late_w2.to_numpy())[0],
                    "late_w2_ci_high": bootstrap_mean_ci(scores.late_w2.to_numpy())[1],
                    "terminal_w2_mean": scores.terminal_w2.mean(),
                    "initial_w2_mean": scores["initial_w2"].mean() if "initial_w2" in scores else float("nan"),
                    "late_tail3_centered": scores.late_tail3_centered.mean(),
                    "late_tail3_original": scores.late_tail3_original.mean(),
                    "synthetic_total": scores.synthetic_total.mean(),
                    "training_total": scores.training_total.mean(),
                    "real_acquired": scores.real_acquired.mean()})
        records.append(row)
    return records


def bootstrap_mean_ci(values: np.ndarray, repetitions: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(20261009)
    samples = rng.choice(values, (repetitions, len(values)), replace=True).mean(axis=1)
    return tuple(float(x) for x in np.quantile(samples, [0.025, 0.975]))


def group_key(row: dict) -> tuple:
    return row["mode"], int(row["n0"]), int(row["r"]), float(row["real_fraction"]), row["policy"]


def grouped(records: list[dict]) -> dict[tuple, list[dict]]:
    groups = {}
    for row in records:
        groups.setdefault(group_key(row), []).append(row)
    return {key: sorted(rows, key=lambda r: r["k"]) for key, rows in groups.items()}


def choose_best_economic(rows: list[dict]) -> tuple[dict, dict, dict]:
    baseline = next(row for row in rows if row["k"] == 0)
    best = min(rows, key=lambda r: r["late_w2_mean"])
    tolerance = 0.05 * baseline["late_w2_mean"]
    eligible = [row for row in rows if row["late_w2_mean"] <= best["late_w2_mean"] + tolerance]
    economic = min(eligible, key=lambda r: (r["synthetic_total"], r["k"]))
    return best, economic, baseline


def config_from_row(row: dict) -> Experiment:
    return Experiment(**{key: row[key] for key in Experiment.__dataclass_fields__})


def analyze_case(meta_path: Path) -> dict:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    directory = meta_path.parent
    trajectories = []
    for seed in meta["seeds"]:
        with np.load(directory / "seeds" / f"seed_{seed:06d}.npz", allow_pickle=False) as saved:
            if str(saved["fingerprint"]) != meta["code_sha256"]:
                raise ValueError("Stale simulation result: " + str(directory))
            trajectories.append(saved["metrics"])
    array = np.stack(trajectories)
    if array.shape[1] != meta["config"]["iterations"] or not np.isfinite(array).all():
        raise ValueError("Incomplete/nonfinite trajectory: " + str(directory))
    index = {name: i for i, name in enumerate(meta["metrics"])}
    n, generations, _ = array.shape
    late = slice(int(0.8 * generations), generations)
    score_frame = pd.DataFrame({
        "seed": meta["seeds"],
        "initial_w2": array[:, 0, index["w2_per_dim"]],
        "late_w2": array[:, late, index["w2_per_dim"]].mean(axis=1),
        "terminal_w2": array[:, -1, index["w2_per_dim"]],
        "late_mean_mse": array[:, late, index["mean_mse"]].mean(axis=1),
        "late_trace_ratio": array[:, late, index["trace_ratio"]].mean(axis=1),
        "late_tail3_centered": array[:, late, index["tail3_centered"]].mean(axis=1),
        "late_tail3_original": array[:, late, index["tail3_original"]].mean(axis=1),
        "synthetic_total": array[:, -1, index["cumulative_synthetic"]],
        "training_total": array[:, -1, index["cumulative_training_rows"]],
        "real_acquired": array[:, -1, index["cumulative_real_acquired"]],
        "terminal_synthetic_extra": array[:, -1, index["n_synthetic_extra"]],
        "terminal_true_training_fraction": array[:, -1, index["actual_training_real_fraction"]],
        "negative_eigenvalues_clipped": array[:, :, index["negative_eigenvalues_clipped"]].sum(axis=1),
        "max_negative_relative": array[:, :, index["largest_negative_relative"]].max(axis=1),
    })
    score_frame.to_csv(directory / "seed_scores.csv", index=False)
    rng = np.random.default_rng(20261009)
    weights = rng.multinomial(n, np.repeat(1 / n, n), size=400) / n
    summary = {"generation": np.arange(1, generations + 1)}
    for metric in CURVE_METRICS:
        values = array[:, :, index[metric]]
        bootstrap = weights @ values
        summary[metric + "_mean"] = values.mean(axis=0)
        summary[metric + "_ci_low"], summary[metric + "_ci_high"] = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        summary[metric + "_median"] = np.median(values, axis=0)
        summary[metric + "_p10"], summary[metric + "_p90"] = np.quantile(values, [0.1, 0.9], axis=0)
    for metric in ALL_METRICS[13:]:
        summary[metric + "_mean"] = array[:, :, index[metric]].mean(axis=0)
    pd.DataFrame(summary).to_csv(directory / "curve_summary.csv", index=False, float_format="%.10g")
    return {"stage": meta["stage"], "case_id": meta["case_id"], "n_seeds": n,
            "generations": generations, "directory": str(directory)}


def main_groups(records: list[dict]) -> list[tuple[tuple, list[dict]]]:
    output = []
    for key, rows in grouped(records).items():
        mode, n0, r, p, policy = key
        if n0 != 500 or policy != "proportional":
            continue
        if (mode == "fixed_initial" or
                (mode in {"cumulative_real", "current_fresh"} and r == 100) or
                (mode == "unlabelled_mixture" and r == 100 and p in (0.25, 0.5, 0.75))):
            output.append((key, rows))
    return sorted(output, key=lambda item: (SCENARIOS[item[0][0]], item[0][3]))


def group_title(key: tuple) -> str:
    mode, n0, r, p, policy = key
    title = LABELS[mode]
    if mode == "unlabelled_mixture":
        title += f" (hidden p={p:g})"
    return title


def save_figure(figure: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def plot_stage(stage: str, records: list[dict]) -> None:
    out = RESULTS / stage / "figures"
    if stage == "00_baseline":
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        for row in sorted(records, key=lambda r: r["n0"]):
            curves = pd.read_csv(Path(row["directory"]) / "curve_summary.csv")
            for ax, metric, label in zip(axes,
                    ("mean_mse", "trace_ratio", "tail3_centered"),
                    ("Mean squared error", "Covariance trace / d", "Centered 3-sigma tail retention")):
                line, = ax.plot(curves.generation, np.maximum(curves[metric+"_median"], 1e-12),
                                label=f"n={row['n0']}")
                ax.plot(curves.generation, np.maximum(curves[metric+"_mean"], 1e-12),
                        color=line.get_color(), ls="--", alpha=0.65)
                ax.set(xlabel="Fitting generation", ylabel=label, yscale="log")
                ax.grid(alpha=0.2)
        axes[0].legend(fontsize=8)
        fig.suptitle("E0 Replace: 100 independent trajectories; solid=median, dashed=mean\n"
                     "Log display clips values below 1e-12; simulation uses no variance floor")
        fig.tight_layout()
        save_figure(fig, out / "01_long_replace.png")
        return
    groups = main_groups(records)
    if not groups:
        return
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, rows) in zip(axes.flat, groups):
        ks = np.asarray([r["k"] for r in rows])
        means = np.asarray([r["late_w2_mean"] for r in rows])
        lower = np.asarray([r["late_w2_ci_low"] for r in rows])
        upper = np.asarray([r["late_w2_ci_high"] for r in rows])
        ax.errorbar(ks, means, yerr=np.maximum(0, [means-lower, upper-means]), marker="o", capsize=3)
        ax.set(xscale="symlog", yscale="log", title=group_title(key),
               xlabel="k = extra synthetic / observable reference batch",
               ylabel="Mean late W2 squared / d")
        ax.set_xticks(ks, [f"{k:g}" for k in ks], fontsize=9, rotation=35, ha="right")
        ax.grid(alpha=0.2)
    n = groups[0][1][0]["n_seeds"]
    t = groups[0][1][0]["iterations"]
    fig.suptitle(f"{stage}: last 20% of {t} generations, {n} independent seeds\n"
                 "E4 denominator is all arrivals; their source labels are unavailable")
    fig.tight_layout()
    save_figure(fig, out / "01_ratio_quality.png")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    frozen = next((r for r in records if r["mode"] == "frozen" and r["n0"] == 500), None)
    accumulated_real = next((r for r in records if r["mode"] == "cumulative_real" and
                             r["n0"] == 500 and r["r"] == 100 and r["k"] == 0 and r["policy"] == "proportional"), None)
    for ax, (key, rows) in zip(axes.flat, groups):
        for row in rows:
            curves = pd.read_csv(Path(row["directory"]) / "curve_summary.csv")
            line, = ax.plot(curves.generation, np.maximum(curves.w2_per_dim_mean, 1e-12),
                            label=f"k={row['k']:g}")
            ax.fill_between(curves.generation, np.maximum(curves.w2_per_dim_ci_low, 1e-12),
                            np.maximum(curves.w2_per_dim_ci_high, 1e-12), color=line.get_color(), alpha=0.10)
        if key[0] in {"current_fresh", "unlabelled_mixture"} and frozen:
            reference = pd.read_csv(Path(frozen["directory"]) / "curve_summary.csv")
            ax.plot(reference.generation, reference.w2_per_dim_mean, "--", color="dimgray", label="Frozen initial model")
        if key[0] == "current_fresh" and accumulated_real:
            reference = pd.read_csv(Path(accumulated_real["directory"]) / "curve_summary.csv")
            ax.plot(reference.generation, reference.w2_per_dim_mean, ":", color="black", label="All real data (oracle)")
        ax.set(title=group_title(key), xlabel="Fitting generation", ylabel="Mean W2 squared / d", yscale="log")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(alpha=0.2)
    fig.suptitle(f"{stage}: trajectories and pointwise 95% seed-bootstrap intervals")
    fig.tight_layout()
    save_figure(fig, out / "02_learning_curves.png")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for ax, (key, rows) in zip(axes.flat, groups):
        for row in rows:
            ax.scatter(row["synthetic_total"], row["late_w2_mean"])
            ax.annotate(f" k={row['k']:g}", (row["synthetic_total"], row["late_w2_mean"]), fontsize=8)
        ax.set(title=group_title(key), xlabel="Cumulative generated synthetic observations",
               ylabel="Mean late W2 squared / d", xscale="symlog", yscale="log")
        ax.grid(alpha=0.2)
    fig.tight_layout()
    save_figure(fig, out / "03_quality_cost.png")
    for metric, ylabel, filename in (
            ("mean_mse", "Mean squared error", "05_mean_drift.png"),
            ("trace_ratio", "Covariance trace / d", "06_covariance_trace.png"),
            ("tail3_centered", "3-sigma tail retention", "07_tail_coverage.png")):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for ax, (key, rows) in zip(axes.flat, groups):
            for row in rows:
                curves = pd.read_csv(Path(row["directory"]) / "curve_summary.csv")
                line, = ax.plot(curves.generation, np.maximum(curves[metric+"_mean"], 1e-12),
                                label=f"k={row['k']:g}")
                ax.fill_between(curves.generation, np.maximum(curves[metric+"_ci_low"], 1e-12),
                                np.maximum(curves[metric+"_ci_high"], 1e-12),
                                color=line.get_color(), alpha=.10)
                if metric == "tail3_centered":
                    ax.plot(curves.generation, np.maximum(curves.tail3_original_mean, 1e-12),
                            ls="--", color=line.get_color(), alpha=.8)
            if metric != "mean_mse":
                ax.axhline(1, color="black", ls=":", lw=1)
            ax.set(title=group_title(key), xlabel="Fitting generation", ylabel=ylabel)
            if metric != "trace_ratio":
                ax.set_yscale("log")
            ax.grid(alpha=.2)
            ax.legend(fontsize=8, ncol=2)
        detail = "solid=after removing fitted mean, dashed=threshold at true origin" if metric == "tail3_centered" else "pointwise 95% seed-bootstrap intervals"
        fig.suptitle(f"{stage}: {ylabel}; {detail}")
        fig.tight_layout()
        save_figure(fig, out / filename)
    if stage == "01_exploration":
        fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
        for ax, (key, rows) in zip(axes.flat, groups):
            # Keep the common six-ratio grid for a visually comparable heatmap.
            rows = [row for row in rows if row["k"] in (0, .25, .5, 1, 2, 5)]
            curves = [pd.read_csv(Path(row["directory"]) / "curve_summary.csv").w2_per_dim_mean.to_numpy()
                      for row in rows]
            im = ax.imshow(np.log10(np.maximum(curves, 1e-12)), aspect="auto", origin="lower",
                           extent=[1, len(curves[0]), -.5, len(rows)-.5], cmap="viridis")
            ax.set_yticks(range(len(rows)), [f"{row['k']:g}" for row in rows])
            ax.set(title=group_title(key), xlabel="Fitting generation", ylabel="k")
            fig.colorbar(im, ax=ax, label="log10(mean W2 squared / d)")
        fig.suptitle("Exploration only: 20 seeds, 300 generations")
        fig.tight_layout()
        save_figure(fig, out / "04_ratio_generation_heatmap.png")


def analyze_stage(stage: str, make_figures: bool = True) -> list[dict]:
    print(f"ANALYZE {stage}", flush=True)
    with threadpool_limits(limits=1):
        for path in sorted((RESULTS / stage).glob("*/*/config.json")):
            analyze_case(path)
        records = records_for_stage(stage)
        pd.DataFrame(records).to_csv(RESULTS / stage / "stage_summary.csv", index=False)
        if make_figures:
            plot_stage(stage, records)
    return records


def boundary_extensions(stage: str) -> list[Experiment]:
    output = []
    for key, rows in grouped(records_for_stage(stage)).items():
        mode, n0, r, p, policy = key
        if mode in {"replace", "frozen"} or policy != "proportional" or p == 0:
            continue
        best, economic, baseline = choose_best_economic(rows)
        if best["k"] == max(row["k"] for row in rows) == 5 and best["late_w2_mean"] < .95 * baseline["late_w2_mean"]:
            output.extend(replace(config_from_row(best), k=k) for k in (10, 20))
    return output


def select_formal_configs(stage: str) -> tuple[list[Experiment], list[dict]]:
    configs, selections = [], []
    for key, rows in main_groups(records_for_stage(stage)):
        best, economic, baseline = choose_best_economic(rows)
        ks = sorted(row["k"] for row in rows)
        best_index = ks.index(best["k"])
        chosen = {0, best["k"], economic["k"], max(ks)}
        if best_index > 0:
            chosen.add(ks[best_index-1])
        selections.append({"group": list(key), "exploration_quality_k": best["k"],
                           "exploration_economic_k": economic["k"], "formal_k": sorted(chosen),
                           "criterion": "mean W2/d over final 20%; tolerance=0.05*real-only baseline"})
        configs.extend(replace(config_from_row(row), iterations=1000)
                       for row in rows if row["k"] in chosen)
    configs.append(Experiment("frozen", r=0))
    configs.extend(Experiment("unlabelled_mixture", real_fraction=0, k=k) for k in (0, 1))
    cumulative = next(x for x in selections if x["group"][0] == "cumulative_real")
    fixed_ks = {1.0, float(cumulative["exploration_quality_k"])} - {0}
    for k in fixed_ks:
        configs.append(Experiment("cumulative_real", k=k, policy="fixed_count"))
        if not any(c.mode == "cumulative_real" and c.policy == "proportional" and c.k == k for c in configs):
            configs.append(Experiment("cumulative_real", k=k))
    return list(dict.fromkeys(configs)), selections


def select_robustness_configs(exploration_stage: str, formal_stage: str) -> list[Experiment]:
    configs = []
    for key, rows in main_groups(records_for_stage(formal_stage)):
        mode, n0, r, p, policy = key
        if mode == "unlabelled_mixture" and p != .5:
            continue
        best, economic, baseline = choose_best_economic(rows)
        for new_n0 in (200, 1000):
            configs.extend(replace(config_from_row(row), n0=new_n0, iterations=1000)
                           for row in (baseline, best, economic))
    for key, rows in grouped(records_for_stage(exploration_stage)).items():
        if key[0] in {"cumulative_real", "current_fresh"} and key[2] in {20, 50} and key[4] == "proportional":
            best, economic, baseline = choose_best_economic(rows)
            configs.extend(replace(config_from_row(row), iterations=1000)
                           for row in (baseline, best, economic))
    return list(dict.fromkeys(configs))


def paired_comparisons(records: list[dict]) -> pd.DataFrame:
    rows_out = []
    for key, rows in grouped(records).items():
        if not any(row["k"] == 0 for row in rows):
            continue
        baseline = next(row for row in rows if row["k"] == 0)
        reference = pd.read_csv(Path(baseline["directory"]) / "seed_scores.csv").set_index("seed").late_w2
        for row in rows:
            scores = pd.read_csv(Path(row["directory"]) / "seed_scores.csv").set_index("seed").late_w2
            difference = (scores - reference).dropna().to_numpy()
            low, high = bootstrap_mean_ci(difference)
            full_scores = pd.read_csv(Path(row["directory"]) / "seed_scores.csv")
            frozen_difference = (full_scores.late_w2-full_scores.initial_w2).to_numpy() if "initial_w2" in full_scores else np.full(len(full_scores), np.nan)
            frozen_low, frozen_high = bootstrap_mean_ci(frozen_difference)
            rows_out.append({**{k: row[k] for k in Experiment.__dataclass_fields__},
                             "difference_from_k0": float(difference.mean()),
                             "paired_ci_low": low, "paired_ci_high": high,
                             "relative_change_percent": 100 * difference.mean() / reference.mean()})
            rows_out[-1].update({"difference_from_frozen_initial": float(frozen_difference.mean()),
                                 "frozen_paired_ci_low": frozen_low, "frozen_paired_ci_high": frozen_high})
    return pd.DataFrame(rows_out)


def plot_marginals(records: list[dict], baseline: list[dict]) -> None:
    selected = []
    if baseline:
        selected.append(next(row for row in baseline if row["n0"] == 316))
    for key, rows in main_groups(records):
        if key[0] == "unlabelled_mixture" and key[3] != .5:
            continue
        selected.append(choose_best_economic(rows)[0])
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, row in zip(axes.flat, selected):
        directory = Path(row["directory"])
        meta = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        with np.load(directory / "seeds" / f"seed_{meta['example_seed']:06d}.npz") as saved:
            j = min(85, row["d"]-1)
            means = saved["snapshot_means"][:, j]
            stds = np.sqrt(np.maximum(0, saved["snapshot_covariances"][:, j, j]))
            x = np.linspace(min(-5, float(np.min(means-4*stds))),
                            max(5, float(np.max(means+4*stds))), 3000)
            for t, mean, cov in zip(saved["snapshot_generations"], saved["snapshot_means"], saved["snapshot_covariances"]):
                std = np.sqrt(max(0, cov[j, j]))
                if std < 1e-4:
                    ax.axvline(mean[j], alpha=.5, label=f"t={t}: near point mass")
                else:
                    ax.plot(x, norm.pdf(x, mean[j], std), label=f"t={t}")
        ax.plot(x, norm.pdf(x), "k--", lw=1.3, label="True N(0,1)")
        detail = f"n={row['n0']}" if row["mode"] == "replace" else f"k={row['k']:g}"
        if row["mode"] == "unlabelled_mixture":
            detail += f", hidden p={row['real_fraction']:g}"
        ax.set(title=f"{LABELS[row['mode']]}: {detail}", xlabel="Coordinate 85", ylabel="Density")
        ax.legend(fontsize=7, ncol=2)
    for ax in axes.flat[len(selected):]:
        ax.set_visible(False)
    fig.suptitle("Preselected seed=10000: fitted marginal densities (actual means retained)")
    fig.tight_layout()
    save_figure(fig, RESULTS / "figures" / "01_marginal_snapshots.png")


def plot_fixed_policy(records: list[dict]) -> None:
    fixed = [r for r in records if r["mode"] == "cumulative_real" and r["policy"] == "fixed_count"]
    if not fixed:
        return
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for row in fixed:
        comparable = next(r for r in records if r["mode"] == "cumulative_real" and
                          r["policy"] == "proportional" and r["k"] == row["k"])
        for record in (row, comparable):
            curves = pd.read_csv(Path(record["directory"]) / "curve_summary.csv")
            label = f"k_start={record['k']:g}, {record['policy']}"
            axes[0].plot(curves.generation, curves.w2_per_dim_mean, label=label)
            axes[1].plot(curves.generation, curves.n_synthetic_extra_mean, label=label)
    axes[0].set(xlabel="Fitting generation", ylabel="Mean W2 squared / d", yscale="log")
    axes[1].set(xlabel="Fitting generation", ylabel="Synthetic count per generation")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("E2: same synthetic count at generation 2; fixed count versus fixed proportion")
    fig.tight_layout()
    save_figure(fig, RESULTS / "figures" / "02_fixed_vs_proportional.png")


def plot_robustness(records: list[dict]) -> None:
    selected = [r for r in records if r["stage"] in {"02_validation", "03_robustness"}
                and r["policy"] == "proportional"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, mode in zip(axes.flat, ("fixed_initial", "cumulative_real", "current_fresh", "unlabelled_mixture")):
        rows = [r for r in selected if r["mode"] == mode
                and r["r"] == (0 if mode == "fixed_initial" else 100)
                and (mode != "unlabelled_mixture" or r["real_fraction"] == .5)]
        for k in (0, 5, 20):
            points = sorted((r for r in rows if r["k"] == k), key=lambda r: r["n0"])
            if len(points) < 2:
                continue
            mean = np.array([r["late_w2_mean"] for r in points])
            ax.errorbar([r["n0"] for r in points], mean,
                        yerr=np.maximum(0, [mean-np.array([r["late_w2_ci_low"] for r in points]),
                                           np.array([r["late_w2_ci_high"] for r in points])-mean]),
                        marker="o", capsize=3, label=f"k={k:g}")
        ax.set(title=LABELS[mode] + (" (hidden p=0.5)" if mode == "unlabelled_mixture" else ""),
               xlabel="Initial sample size n0", ylabel="Mean late W2 squared / d", yscale="log")
        ax.set_xticks([200, 500, 1000])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Initial-data sensitivity: T=1000; n0=500 has 100 seeds, others 30\n"
                 "Only selected candidates were rechecked; omitted curves were not tested")
    fig.tight_layout()
    save_figure(fig, RESULTS / "figures" / "03_initial_size_sensitivity.png")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, mode in zip(axes, ("cumulative_real", "current_fresh")):
        rows = [r for r in selected if r["mode"] == mode and r["n0"] == 500]
        for k in (0, 5, 10, 20):
            points = sorted((r for r in rows if r["k"] == k), key=lambda r: r["r"])
            if len(points) < 2:
                continue
            ax.plot([r["r"] for r in points], [r["late_w2_mean"] for r in points],
                    marker="o", label=f"k={k:g}; extra m=k*observable count")
        ax.set(title=LABELS[mode], xlabel="New real observations per round r",
               ylabel="Mean late W2 squared / d", yscale="log")
        ax.set_xticks([20, 50, 100])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.suptitle("Arrival-size sensitivity: T=1000; r=100 has 100 seeds, r=20/50 have 30")
    fig.tight_layout()
    save_figure(fig, RESULTS / "figures" / "04_arrival_size_sensitivity.png")


def final_report() -> None:
    records = []
    for stage in ("00_baseline", "01_exploration", "02_validation", "03_robustness"):
        records.extend(records_for_stage(stage))
    if not records:
        raise RuntimeError("No analyzed results available")
    pd.DataFrame(records).to_csv(RESULTS / "all_results_index.csv", index=False)
    formal = [row for row in records if row["stage"] == "02_validation"]
    baseline = [row for row in records if row["stage"] == "00_baseline"]
    comparisons = paired_comparisons(formal)
    comparisons.to_csv(RESULTS / "paired_comparisons.csv", index=False)
    mixed_groups = [rows for key, rows in main_groups(formal) if key[0] == "unlabelled_mixture"]
    blind_rows, blind_text = [], []
    if mixed_groups:
        common_ks = set.intersection(*[{row["k"] for row in rows} for rows in mixed_groups])
        for k in sorted(common_ks):
            regrets, losses = [], []
            for rows in mixed_groups:
                best, _, reference = choose_best_economic(rows)
                row = next(row for row in rows if row["k"] == k)
                regrets.append((row["late_w2_mean"]-best["late_w2_mean"])/reference["late_w2_mean"])
                losses.append(row["late_w2_mean"])
            blind_rows.append({"k": k, "extra_per_100_arrivals": round(100*k),
                               "worst_baseline_normalized_regret": max(regrets),
                               "worst_late_w2": max(losses),
                               "within_5_percent_baseline_all_tested_p": max(regrets) <= .05})
        pd.DataFrame(blind_rows).to_csv(RESULTS / "unlabelled_blind_candidates.csv", index=False)
        economical = [row for row in blind_rows if row["within_5_percent_baseline_all_tested_p"]]
        if economical:
            chosen = min(economical, key=lambda row: row["k"])
            blind_text = ["## 第四种：不知道 p 时怎样看结果", "",
                          f"在共同验证的比例中，固定 k={chosen['k']:g}（每 100 个混合到达样本额外生成 {chosen['extra_per_100_arrivals']} 个）在所有已测试 p=0.25/0.5/0.75 条件下，均满足预设的 5% 基线损失容忍度。",
                          "这条固定策略在执行时不读取 p 或标签；它是在给定模拟条件集合内的经济候选，不保证对其他混合比例或分布仍最优。",
                          "[来源未知时的共同候选比较](unlabelled_blind_candidates.csv)。", ""]
    recommendations, table_lines = [], []
    for key, rows in main_groups(formal):
        best, economic, reference = choose_best_economic(rows)
        low, high = best["late_w2_ci_low"], best["late_w2_ci_high"]
        improvement = 100 * (reference["late_w2_mean"]-best["late_w2_mean"]) / reference["late_w2_mean"]
        max_k = max(row["k"] for row in rows)
        edge = best["k"] == max_k and max_k > 0
        formula = {"fixed_initial": f"m=500×{economic['k']:g}",
                   "cumulative_real": f"m_t≈{economic['k']:g}×[500+(t−1)×100]",
                   "current_fresh": f"m=100×{economic['k']:g}",
                   "unlabelled_mixture": f"m=100×{economic['k']:g}（分母为混合到达总数）"}[key[0]]
        label = CHINESE[key[0]] + (f"，隐藏 p={key[3]:g}" if key[0] == "unlabelled_mixture" else "")
        recommendations.append({"scenario": label, "mode": key[0], "hidden_real_fraction": key[3],
                                "quality_k": best["k"], "economic_k": economic["k"],
                                "economic_count_rule": formula,
                                "late_w2_mean": best["late_w2_mean"], "ci_low": low, "ci_high": high,
                                "improvement_over_k0_percent": improvement,
                                "initial_model_w2": best["initial_w2_mean"],
                                "change_from_initial_percent": 100*(best["late_w2_mean"]/best["initial_w2_mean"]-1),
                                "best_at_search_boundary": edge})
        table_lines.append(f"| {label} | {best['k']:g} | {economic['k']:g} | {formula} | "
                           f"{best['late_w2_mean']:.5g} [{low:.5g}, {high:.5g}] | {improvement:.1f}% | "
                           f"{'上界，尚不能确认有限最优' if edge else '已测试候选范围内'} |")
    pd.DataFrame(recommendations).to_csv(RESULTS / "recommendations.csv", index=False, encoding="utf-8-sig")
    plot_marginals(formal, baseline)
    plot_fixed_policy(formal)
    plot_robustness(records)
    freeze_notes = []
    current = next((row for row in recommendations if row["mode"] == "current_fresh"), None)
    if current:
        freeze_notes = ["## 是否真的优于已有模型", "",
                        f"主实验初始模型平均 W₂²/d 为 {current['initial_model_w2']:.5g}；第三种质量候选的后期损失为 {current['late_w2_mean']:.5g}，相对冻结初始模型改变 {current['change_from_initial_percent']:+.1f}%。",
                        "因此必须同时区分：优于只用当轮真实数据，和优于不更新已有模型。主验证轨迹图已加入冻结模型对照；第三种还加入累计全部真实数据的评价参照。",
                        "初始模型好坏会影响这个比较，补充的 n₀=200/1000 结果也保留每条轨迹的初始误差。", ""]
        for n0 in (200, 1000):
            candidates = [r for r in records if r["stage"] == "03_robustness" and
                          r["mode"] == "current_fresh" and r["n0"] == n0 and r["r"] == 100]
            if candidates:
                best = min(candidates, key=lambda r: r["late_w2_mean"])
                change = 100*(best["late_w2_mean"]/best["initial_w2_mean"]-1)
                freeze_notes.append(f"n₀={n0} 补充验证：初始损失 {best['initial_w2_mean']:.5g}，所测最佳更新候选 k={best['k']:g} 的后期损失 {best['late_w2_mean']:.5g}，相对冻结改变 {change:+.1f}%。")
        freeze_notes.append("")
    total_runs = sum(row["n_seeds"] for row in records)
    total_steps = sum(row["n_seeds"]*row["iterations"] for row in records)
    complete_cases = len(records)
    environment = json.loads((RESULTS / "environment.json").read_text(encoding="utf-8"))
    lines = ["# A2 实验结果解读", "", "以下内容由实际保存的实验结果自动生成。", "",
             f"完成 {complete_cases} 个配置、{total_runs} 条模拟轨迹、{total_steps:,} 条逐代评估记录。每个配置内使用独立种子，不同条件共享种子用于配对比较。",
             "主验证：d=100、n₀=500、每轮到达数 r=100、100 个独立种子、1000 代；补充稳健性为 30 个独立种子。",
             "协方差统一使用分母 N−1。所有结果均为真实随机有限样本递归，未用总体矩替代采样。", "",
             "## 先看哪几张图", "",
             "- [主验证：合成比例与效果](02_validation/figures/01_ratio_quality.png)",
             "- [主验证：1000 代轨迹](02_validation/figures/02_learning_curves.png)",
             "- [主验证：质量与抽样成本](02_validation/figures/03_quality_cost.png)",
             "- [主验证：均值误差](02_validation/figures/05_mean_drift.png)",
             "- [主验证：协方差迹比](02_validation/figures/06_covariance_trace.png)",
             "- [主验证：尾部覆盖](02_validation/figures/07_tail_coverage.png)",
             "- [完全替换：均值与中位数的长期变化](00_baseline/figures/01_long_replace.png)",
             "- [初末边际密度快照](figures/01_marginal_snapshots.png)",
             "- [第二种：固定数量与固定比例](figures/02_fixed_vs_proportional.png)",
             "- [初始样本量敏感性](figures/03_initial_size_sensitivity.png)",
             "- [每轮新增真实数量敏感性](figures/04_arrival_size_sensitivity.png)",
             "- [探索阶段：比例随代数的效果热图](01_exploration/figures/04_ratio_generation_heatmap.png)", "",
             "## 每种条件下抽多少", "",
             "k 是合成数／可用真实数；E4 中算法只能看到混合批次，因此 k 是额外合成数／混合到达总数。",
             "第二种累计新真实样本，第三、四种均不保留历轮到达样本。p 只由模拟环境控制，算法看不到来源标签。", "",
             "| 条件 | 验证集合内质量候选 k | 经济候选 k | 经济候选换算 | 质量候选后期 W₂²/d [95%区间] | 相比 k=0 降低 | 搜索说明 |",
             "|---|---:|---:|---|---|---:|---|", *table_lines, "",
             "后期指标为第 801—1000 代损失的轨迹内均值，再跨 seeds 汇总。经济候选容忍度为最好均值加纯真实/原样到达基线均值的 5%。",
             "候选来自独立探索种子；此表描述正式验证集合内的观察结果，不证明全局最优。隐藏 p 分组推荐是模拟中的条件分析，未知 p 时不能直接据此部署自动选参。", "",
             *blind_text,
             *freeze_notes,
             "## 统计证据和边界", "",
             "- [每个候选相对同条件 k=0 的配对差异与区间](paired_comparisons.csv)，负值表示损失更低。",
             "- [推荐表 CSV](recommendations.csv)；[全部配置与结果索引](all_results_index.csv)。",
             "- 曲线阴影为按完整 seed 轨迹重抽样得到的点态 95% 区间，不是整条曲线的同时区间。",
             "- E0 无偏协方差的跨轨迹均值可能由少数极端轨迹支撑；100 次重复不能保证可靠估计稀有事件的期望。",
             "- 第四种的 m=0 仍包含外部流里的合成数据，因此其基线不是纯真实数据。",
             "- 第四种额外生成的合成数据没有被冒充为新真实信息；真实获取量与合成量分别记录。",
             "- 实验没有根据来源标签筛选或加权第四种样本，也没有从第三、四种的旧真实数据重建训练集。",
             "- 有限 1000 代的平台不能直接当作渐近收敛证明；未制造采样偏置以保证出现内部最优比例。", "",
             "## 稳健性", "",
             "[补充结果总表](03_robustness/stage_summary.csv)检查 n₀=200/1000，以及每轮新真实数 r=20/50。每项均记录实际重复数。",
             "## 可复现环境", "", f"Python {environment['python']}；NumPy {environment['numpy']}；{environment['workers']} 个计算进程，每进程 BLAS 单线程。",
             f"模拟代码 SHA256：`{environment['code_sha256']}`。", "",
             "工程验证见 `../tests/`；完整配置与每条轨迹保存在各情形目录的 config.json 和 seeds/*.npz。",
             "各配置 curve_summary.csv 可直接查图中数值，seed_scores.csv 是配对分析和推荐表的数据来源。", ""]
    (RESULTS / "结果解读.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT: {RESULTS / '结果解读.md'}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Regenerate A2 summaries and figures from saved trajectories")
    parser.add_argument("--all", action="store_true", help="Recompute stage summaries/figures before the report")
    args = parser.parse_args()
    if args.all:
        for stage in ("00_baseline", "01_exploration", "02_validation", "03_robustness"):
            if (RESULTS / stage).exists():
                analyze_stage(stage)
    final_report()
