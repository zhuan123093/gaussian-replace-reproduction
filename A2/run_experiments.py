"""Run the A2 staged study. All paths are relative to this script, not the cwd."""
from __future__ import annotations

import os
for _blas_variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_blas_variable] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits

from gaussian_core import (METRIC_NAMES, batch_from_array, choose_real_batch,
                           empty_batch, fit_gaussian, fixed_directions,
                           gaussian_batch, merge_batches, mixture_arrival,
                           model_metrics)


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SCENARIOS = {
    "replace": "00_replace",
    "fixed_initial": "01_fixed_initial",
    "cumulative_real": "02_initial_plus_cumulative",
    "current_fresh": "03_current_fresh_only",
    "unlabelled_mixture": "04_unlabelled_mixture",
    "frozen": "05_frozen_reference",
}
ALL_METRICS = METRIC_NAMES + ["policy_basis_count", "actual_training_real_fraction"]
K_GRID = (0, 0.25, 0.5, 1, 2, 5)


def numeric_tag(x: float) -> str:
    return f"{x:g}".replace(".", "p")


@dataclass(frozen=True)
class Experiment:
    mode: str
    n0: int = 500
    r: int = 100
    k: float = 0
    real_fraction: float = 1
    policy: str = "proportional"
    d: int = 100
    iterations: int = 1000

    @property
    def case_id(self) -> str:
        return (f"n0_{self.n0}_r_{self.r}_k_{numeric_tag(self.k)}"
                f"_p_{numeric_tag(self.real_fraction)}_{self.policy}_T{self.iterations}")

    @property
    def group(self) -> tuple:
        return self.mode, self.n0, self.r, self.real_fraction, self.policy


def json_write(path: Path, data: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def simulation_hash() -> str:
    return hashlib.sha256((ROOT / "gaussian_core.py").read_bytes()
                          + Path(__file__).read_bytes()).hexdigest()


def case_path(stage: str, config: Experiment) -> Path:
    return RESULTS / stage / SCENARIOS[config.mode] / config.case_id


def seed_is_complete(path: Path, fingerprint: str, iterations: int) -> bool:
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as saved:
            return (str(saved["fingerprint"]) == fingerprint
                    and saved["metrics"].shape == (iterations, len(ALL_METRICS))
                    and np.isfinite(saved["metrics"]).all())
    except (ValueError, OSError, KeyError):
        return False


def simulate(config: Experiment, seed: int, snapshots: bool = False) -> dict:
    """Training policies receive aggregate batches; origins remain in environment.

    In E4 real_fraction is consumed only by mixture_arrival, never by the policy
    determining extra m, and origin counts are only used by the evaluator.
    """
    d = config.d
    if d < 1 or config.n0 < 2 or config.iterations < 1:
        raise ValueError("Invalid dimension, initial sample count, or generations")
    if config.mode not in SCENARIOS or config.k < 0:
        raise ValueError("Invalid mode or ratio")
    if config.mode in {"current_fresh", "cumulative_real", "unlabelled_mixture"} and config.r < 2:
        raise ValueError("At least two arriving observations are required")
    initial_rng, fresh_rng, extra_rng, arrival_fake_rng, mask_rng = [
        np.random.default_rng(np.random.SeedSequence([seed, tag, d]))
        for tag in (1, 2, 3, 4, 5)
    ]
    initial = batch_from_array(initial_rng.standard_normal((config.n0, d)))
    model = fit_gaussian(initial)
    if config.mode not in {"fixed_initial", "cumulative_real"}:
        initial = empty_batch(d)  # No initial sufficient statistics exposed later.
    cached = initial if config.mode == "cumulative_real" else empty_batch(d)
    directions = fixed_directions(d)
    coordinate = min(85, d - 1)
    metrics = np.empty((config.iterations, len(ALL_METRICS)), dtype=np.float64)
    snapshot_at = sorted({g for g in (1, 10, 100, 300, 1000, config.iterations)
                          if g <= config.iterations})
    snapshot_means, snapshot_covariances, snapshot_generations = [], [], []
    cumulative_real, cumulative_synthetic, cumulative_train = config.n0, 0, 0
    started = time.perf_counter()
    for generation in range(1, config.iterations + 1):
        n_real_fit = config.n0 if generation == 1 else 0
        n_arrivals = n_real_arrivals = n_fake_arrivals = n_extra = 0
        n_train = config.n0 if generation == 1 else 0
        basis = config.n0 if generation == 1 else 0
        if generation > 1 and config.mode == "replace":
            n_extra, n_train = config.n0, config.n0
            model = fit_gaussian(gaussian_batch(model, n_extra, extra_rng))
        elif generation > 1 and config.mode != "frozen":
            if config.mode == "fixed_initial":
                observable = initial
                n_real_fit = initial.n
            else:
                fresh_array = fresh_rng.standard_normal((config.r, d))
                n_arrivals = config.r
                if config.mode == "unlabelled_mixture":
                    # The source mask is discarded before the learner is called.
                    observable, n_real_arrivals = mixture_arrival(
                        model, fresh_array, config.real_fraction, mask_rng, arrival_fake_rng)
                    n_fake_arrivals = config.r - n_real_arrivals
                    n_real_fit = n_real_arrivals  # Evaluation-only true count.
                else:
                    fresh = batch_from_array(fresh_array)
                    observable = choose_real_batch(config.mode, initial, cached, fresh)
                    n_real_arrivals = config.r
                    n_real_fit = observable.n
                    if config.mode == "cumulative_real":
                        cached = observable
            basis = observable.n  # E4 uses total arrivals; hidden p is unavailable.
            if config.mode == "cumulative_real" and config.policy == "fixed_count":
                n_extra = int(round(config.k * (config.n0 + config.r)))
            else:
                n_extra = int(round(config.k * basis))
            extra = gaussian_batch(model, n_extra, extra_rng)
            training_batch = merge_batches(observable, extra)
            n_train = training_batch.n
            model = fit_gaussian(training_batch)
        cumulative_real += n_real_arrivals
        cumulative_synthetic += n_extra + n_fake_arrivals
        cumulative_train += n_train
        metrics[generation - 1] = model_metrics(model, directions, coordinate) + [
            n_real_fit, n_arrivals, n_real_arrivals, n_fake_arrivals, n_extra, n_train,
            cumulative_real, cumulative_synthetic, cumulative_train,
            model.clipped_negative, model.largest_negative_relative, basis,
            n_real_fit / n_train if n_train else 0.0,
        ]
        if snapshots and generation in snapshot_at:
            snapshot_generations.append(generation)
            snapshot_means.append(model.mean.copy())
            snapshot_covariances.append(model.covariance.copy())
    if not np.isfinite(metrics).all():
        raise FloatingPointError("Nonfinite metric encountered")
    return {"metrics": metrics,
            "snapshot_generations": np.asarray(snapshot_generations, dtype=int),
            "snapshot_means": np.asarray(snapshot_means),
            "snapshot_covariances": np.asarray(snapshot_covariances),
            "elapsed_seconds": np.asarray(time.perf_counter() - started)}


def run_seed(job: tuple) -> tuple:
    config_dict, seed, path_string, fingerprint, keep_snapshots = job
    config = Experiment(**config_dict)
    with threadpool_limits(limits=1):
        result = simulate(config, seed, keep_snapshots)
    path = Path(path_string)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **result, seed=np.asarray(seed),
                            fingerprint=np.asarray(fingerprint))
    temporary.replace(path)
    return config.case_id, seed, float(result["elapsed_seconds"])


def run_stage(stage: str, configs: list[Experiment], seeds: list[int], workers: int) -> None:
    fingerprint = simulation_hash()
    jobs, already = [], 0
    for config in configs:
        directory = case_path(stage, config)
        (directory / "seeds").mkdir(parents=True, exist_ok=True)
        metadata = {"config": asdict(config), "scenario_directory": SCENARIOS[config.mode],
                    "case_id": config.case_id, "seeds": seeds, "metrics": ALL_METRICS,
                    "covariance_ddof": 1, "code_sha256": fingerprint,
                    "example_seed": seeds[0], "stage": stage}
        json_write(directory / "config.json", metadata)
        for seed in seeds:
            path = directory / "seeds" / f"seed_{seed:06d}.npz"
            if seed_is_complete(path, fingerprint, config.iterations):
                already += 1
            else:
                jobs.append((asdict(config), seed, str(path), fingerprint, seed == seeds[0]))
    total = len(configs) * len(seeds)
    print(f"STAGE {stage}: {len(configs)} cases, {total} trajectories; "
          f"{already} reusable, {len(jobs)} pending; workers={workers}", flush=True)
    started, last_notice, finished = time.perf_counter(), 0.0, already
    status_path = RESULTS / "run_status.json"
    def update_status(state: str) -> None:
        json_write(status_path, {"state": state, "stage": stage, "completed": finished,
                                 "total": total, "elapsed_seconds": time.perf_counter() - started,
                                 "workers": workers, "code_sha256": fingerprint})
    update_status("running")
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(run_seed, job) for job in jobs]
            for future in as_completed(futures):
                case_id, seed, elapsed = future.result()
                finished += 1
                now = time.perf_counter()
                if now - last_notice >= 15 or finished == total:
                    print(f"  {stage} {finished}/{total} | "
                          f"wall={now-started:.1f}s | last_seed={seed} "
                          f"last_run={elapsed:.2f}s | {case_id}", flush=True)
                    update_status("running")
                    last_notice = now
    update_status("stage_complete")
    print(f"DONE {stage}: {finished}/{total}, {time.perf_counter()-started:.1f}s", flush=True)


def exploration_configs() -> list[Experiment]:
    configs = [Experiment("fixed_initial", r=0, k=k, iterations=300) for k in K_GRID]
    for r in (20, 50, 100):
        for mode in ("cumulative_real", "current_fresh"):
            configs.extend(Experiment(mode, r=r, k=k, iterations=300) for k in K_GRID)
    for p in (0.25, 0.5, 0.75):
        configs.extend(Experiment("unlabelled_mixture", k=k, real_fraction=p, iterations=300)
                       for k in K_GRID)
    configs.extend(Experiment("unlabelled_mixture", k=k, real_fraction=0, iterations=300)
                   for k in (0, 1))
    configs.extend(Experiment("cumulative_real", k=k, policy="fixed_count", iterations=300)
                   for k in (1, 5))
    configs.append(Experiment("frozen", r=0, iterations=300))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Full staged exploration/validation/stress study")
    parser.add_argument("--smoke", action="store_true", help="Small engineering verification run")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    RESULTS.mkdir(parents=True, exist_ok=True)
    from analyze_results import (analyze_stage, boundary_extensions, final_report,
                                 select_formal_configs, select_robustness_configs)
    if args.analyze_only:
        final_report()
        return
    environment = {"python": platform.python_version(), "executable": os.sys.executable,
                   "platform": platform.platform(), "numpy": np.__version__,
                   "workers": args.workers, "seed_scheme": "SeedSequence([seed, stream_id, d])",
                   "code_sha256": simulation_hash()}
    json_write(RESULTS / "environment.json", environment)
    if args.smoke:
        cases = [Experiment(mode, n0=200, r=20 if mode != "fixed_initial" else 0,
                            k=1, iterations=20, real_fraction=0.5 if mode == "unlabelled_mixture" else 1)
                 for mode in ("replace", "fixed_initial", "cumulative_real", "current_fresh", "unlabelled_mixture")]
        run_stage("_smoke", cases, [900001, 900002], args.workers)
        analyze_stage("_smoke", make_figures=False)
        return
    if not args.all:
        parser.error("Choose --all, --smoke, or --analyze-only")
    started = time.perf_counter()
    with threadpool_limits(limits=1):
        exploration = exploration_configs()
        run_stage("01_exploration", exploration, list(range(20)), args.workers)
        analyze_stage("01_exploration")
        extensions = boundary_extensions("01_exploration")
        if extensions:
            print(f"Extending {len(extensions)} boundary cases to k=10/20", flush=True)
            run_stage("01_exploration", exploration + extensions, list(range(20)), args.workers)
            analyze_stage("01_exploration")
        formal, selection = select_formal_configs("01_exploration")
        json_write(RESULTS / "selection_from_exploration.json", selection)
        baseline = [Experiment("replace", n0=n, r=0, iterations=1000)
                    for n in (10, 32, 100, 316, 1000)]
        run_stage("00_baseline", baseline, list(range(10000, 10100)), args.workers)
        analyze_stage("00_baseline")
        run_stage("02_validation", formal, list(range(10000, 10100)), args.workers)
        analyze_stage("02_validation")
        robustness = select_robustness_configs("01_exploration", "02_validation")
        run_stage("03_robustness", robustness, list(range(20000, 20030)), args.workers)
        analyze_stage("03_robustness")
        final_report()
    json_write(RESULTS / "run_status.json", {"state": "complete", "elapsed_seconds": time.perf_counter()-started,
                                             "code_sha256": simulation_hash()})
    print(f"ALL COMPLETE in {(time.perf_counter()-started)/60:.1f} min. "
          f"Read {RESULTS / '结果解读.md'}", flush=True)


if __name__ == "__main__":
    main()
