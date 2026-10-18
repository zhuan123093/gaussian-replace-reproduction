"""Verify saved scientific results against counts, source visibility, and budgets."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path

import numpy as np

from run_experiments import ALL_METRICS, RESULTS, json_write, simulation_hash


def verify_stage(stage: str) -> dict:
    index = {name: i for i, name in enumerate(ALL_METRICS)}
    cases = trajectories = generation_records = 0
    max_negative_relative = 0.0
    for meta_path in sorted((RESULTS / stage).glob("*/*/config.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        c = meta["config"]
        assert meta["code_sha256"] == simulation_hash(), str(meta_path)
        assert meta["covariance_ddof"] == 1
        t = np.arange(1, c["iterations"] + 1)
        for seed in meta["seeds"]:
            path = meta_path.parent / "seeds" / f"seed_{seed:06d}.npz"
            with np.load(path, allow_pickle=False) as saved:
                assert str(saved["fingerprint"]) == meta["code_sha256"]
                values = saved["metrics"]
                assert values.shape == (len(t), len(ALL_METRICS))
                assert np.isfinite(values).all()
                assert np.all(values[:, index["w2_per_dim"]] >= 0)
                assert np.all(values[:, index["trace_ratio"]] >= 0)
                get = lambda name: values[:, index[name]]
                for name in ("n_real_fit", "n_arrivals", "n_real_arrivals", "n_synthetic_arrivals",
                             "n_synthetic_extra", "n_train", "policy_basis_count"):
                    assert np.all(get(name) == np.round(get(name)))
                if c["mode"] == "replace":
                    np.testing.assert_array_equal(get("n_train"), c["n0"])
                    np.testing.assert_array_equal(get("n_real_fit")[1:], 0)
                    np.testing.assert_array_equal(get("cumulative_real_acquired"), c["n0"])
                elif c["mode"] == "frozen":
                    np.testing.assert_array_equal(get("n_train")[1:], 0)
                    np.testing.assert_allclose(values[:, :13], np.repeat(values[:1, :13], len(t), axis=0), atol=1e-13)
                else:
                    expected_basis = (np.full(len(t), c["n0"]) if c["mode"] == "fixed_initial" else
                                      c["n0"] + (t-1)*c["r"] if c["mode"] == "cumulative_real" else
                                      np.full(len(t), c["r"]))
                    expected_basis[0] = c["n0"]
                    np.testing.assert_array_equal(get("policy_basis_count"), expected_basis)
                    if c["policy"] == "fixed_count":
                        expected_extra = np.full(len(t), round(c["k"]*(c["n0"]+c["r"])))
                    else:
                        expected_extra = np.round(c["k"] * expected_basis)
                    expected_extra[0] = 0
                    np.testing.assert_array_equal(get("n_synthetic_extra"), expected_extra)
                    np.testing.assert_array_equal(get("n_train"), expected_basis + expected_extra)
                    if c["mode"] == "unlabelled_mixture":
                        assert np.all((get("n_real_arrivals") >= 0) & (get("n_real_arrivals") <= c["r"]))
                        np.testing.assert_array_equal(get("n_real_arrivals")[1:] + get("n_synthetic_arrivals")[1:], c["r"])
                        # m is independent of the hidden source composition.
                        np.testing.assert_array_equal(get("n_synthetic_extra")[1:], round(c["k"]*c["r"]))
                    else:
                        np.testing.assert_array_equal(get("n_real_fit"), expected_basis)
                np.testing.assert_allclose(get("cumulative_real_acquired"),
                                           c["n0"] + np.cumsum(get("n_real_arrivals")), atol=0)
                np.testing.assert_array_equal(get("cumulative_synthetic"),
                                              np.cumsum(get("n_synthetic_extra") + get("n_synthetic_arrivals")))
                np.testing.assert_array_equal(get("cumulative_training_rows"), np.cumsum(get("n_train")))
                max_negative_relative = max(max_negative_relative, float(get("largest_negative_relative").max()))
                trajectories += 1
                generation_records += len(t)
        cases += 1
    return {"stage": stage, "cases": cases, "trajectories": trajectories,
            "generation_records": generation_records, "max_negative_relative": max_negative_relative,
            "finite_metrics_and_counts": "passed", "data_retention_and_sampling_budgets": "passed"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", help="Verify only a completed stage; repeatable")
    args = parser.parse_args()
    stages = args.stage or ["00_baseline", "01_exploration", "02_validation", "03_robustness"]
    verified = [verify_stage(stage) for stage in stages]
    report = {"verified_at_utc": datetime.now(timezone.utc).isoformat(),
              "code_sha256": simulation_hash(), "stages": verified,
              "all_checks_passed": all(row["cases"] > 0 for row in verified)}
    json_write(RESULTS / ("verification.json" if not args.stage else "verification_partial.json"), report)
    environment_path = RESULTS / "environment.json"
    if environment_path.exists():
        environment = json.loads(environment_path.read_text(encoding="utf-8"))
        environment["packages"] = {name: importlib.metadata.version(name)
                                   for name in ("numpy", "scipy", "pandas", "matplotlib", "threadpoolctl", "pytest")}
        json_write(environment_path, environment)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
