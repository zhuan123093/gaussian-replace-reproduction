"""Audit final directional tails using every saved covariance in the pilot.

Averages over random directions can hide contraction in a particular direction.
The minimum-variance direction is chosen from the fitted covariance, not used
by the learner. It need not minimize the noncentered tail probability.
"""
from pathlib import Path
import argparse
import json

import numpy as np
import pandas as pd
from scipy.special import ndtr
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1]/"results"/"parameter_memory_pilot",
                        help="Directory of an existing parameter-memory run")
    root = parser.parse_args().output
    metadata = json.loads((root/"metadata.json").read_text(encoding="utf-8"))
    rows = []
    q3 = 2*ndtr(-3.)
    with threadpool_limits(limits=1):
        for config in metadata["configs"]:
            directory = root/f"n0_{config['n0']}_p_{config['p']:g}_{config['method']}"
            for seed in metadata["seeds"]:
                with np.load(directory/f"seed_{seed:06d}.npz") as saved:
                    assert str(saved["fingerprint"]) == metadata["fingerprint"]
                    mu, cov = saved["final_mean"], saved["final_covariance"]
                    mu0, cov0 = saved["initial_mean"], saved["initial_covariance"]
                eigen0, vectors0 = np.linalg.eigh(cov0)
                inverse_root = (vectors0/np.sqrt(eigen0)) @ vectors0.T
                relative = inverse_root @ cov @ inverse_root
                eigenvalues, eigenvectors = np.linalg.eigh((relative+relative.T)/2)
                direction = inverse_root @ eigenvectors[:, 0]
                variance = max(0., float(eigenvalues[0]))
                shift = float(direction @ (mu-mu0))
                if variance > 0:
                    centered_tail = 2*ndtr(-3/np.sqrt(variance))/q3
                    fixed_initial_tail = (ndtr((-3-shift)/np.sqrt(variance))
                                          + ndtr((shift-3)/np.sqrt(variance)))/q3
                else:
                    centered_tail = 0.
                    fixed_initial_tail = float(abs(shift)>3)/q3
                rows.append({"n0": config["n0"], "p": config["p"], "method": config["method"],
                             "seed": seed, "minimum_relative_variance": variance,
                             "tail_ratio_centered_min_variance_direction": centered_tail,
                             "tail_ratio_fixed_initial_event_min_variance_direction": fixed_initial_tail})
    frame = pd.DataFrame(rows)
    frame.to_csv(root/"directional_tail_audit_seeds.csv", index=False)
    summary = frame.groupby(["n0", "p", "method"]).agg(
        minimum_relative_variance_mean=("minimum_relative_variance", "mean"),
        fixed_initial_event_tail_ratio_median=("tail_ratio_fixed_initial_event_min_variance_direction", "median"),
        fixed_initial_event_tail_ratio_min=("tail_ratio_fixed_initial_event_min_variance_direction", "min"),
        fixed_initial_event_tail_ratio_max=("tail_ratio_fixed_initial_event_min_variance_direction", "max"))
    summary.to_csv(root/"directional_tail_audit_summary.csv")
    print(summary.to_string())


if __name__ == "__main__":
    main()
