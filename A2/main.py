"""One entry point for A2 experiments, analysis and checks.

Run `python A2/main.py --help` from the project directory. Child processes use
the same interpreter; their imports and result paths do not depend on the cwd.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
COMMANDS = {
    "study": ("run_experiments.py", "Main Gaussian study: --all / --smoke / --analyze-only"),
    "analyze": ("analyze_results.py", "Rebuild the report; --all also rebuilds stage summaries and figures"),
    "verify": ("checks/verify_outputs.py", "Verify saved trajectories; optional --stage NAME"),
    "memory": ("memory/run_parameter_memory.py", "Run the parameter-memory pilot; new output: results/parameter_memory"),
    "audit": ("memory/audit_parameter_memory.py", "Audit directional tails; --output selects a saved memory run"),
    "theory": ("theory/validate_theory.py", "Check exact moment theory and saved simulations"),
    "spectral": ("theory/validate_spectral_bounds.py", "Check covariance spectral bounds"),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name, (_, help_text) in COMMANDS.items():
        # Forward options, including --help, to the original command.
        subparsers.add_parser(name, help=help_text,
                             description=help_text,
                             add_help=name in {"theory", "spectral"})
    args, forwarded = parser.parse_known_args()
    if args.command in {"theory", "spectral"} and forwarded:
        parser.error(f"{args.command} takes no extra options")
    environment = os.environ.copy()
    import_paths = [str(ROOT), str(ROOT / "theory")]
    if environment.get("PYTHONPATH"):
        import_paths.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(import_paths)
    # Keep caller-relative --output paths relative to the caller's directory.
    return subprocess.run(
        [sys.executable, str(ROOT / COMMANDS[args.command][0]), *forwarded],
        env=environment,
    ).returncode


if __name__ == "__main__":
    raise SystemExit(main())
