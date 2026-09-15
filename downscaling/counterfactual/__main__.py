"""Reproducible synthetic shift + variance experiment; no training or downloads."""

import argparse
import shlex
import subprocess

import numpy as np
import pandas as pd
import xarray as xr

from .analysis import analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/counterfactual/synthetic.json")
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--run-ref", help="Persistent run identifier (defaults to output path)")
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)

    def series(start, end):
        time = pd.date_range(start, end, freq="D")
        years = time.year - time.year.min()
        anomalies = rng.normal(0, 0.7, years.max() + 1)[years] + rng.normal(0, 2, time.size)
        seasonal = 8 - 8 * np.cos(2 * np.pi * (time.dayofyear.to_numpy() - 15) / 365.25)
        return xr.DataArray(
            seasonal + anomalies, dims="time", coords={"time": time}, attrs={"units": "degC"}
        )

    hist = series("1981-01-01", "2000-12-31")
    obs = hist * 0.7 - 3
    target = series("2041-01-01", "2060-12-31") * 1.2 + 2
    obs.attrs["units"] = target.attrs["units"] = "degC"
    result = analyze(
        hist,
        obs,
        target,
        index="spring_tmin",
        baseline="synthetic observations 1981-2000",
        reference=f"synthetic-v1 seed={args.seed}; obs=0.7*hist-3; target=1.2*climate+2",
        run_ref=args.run_ref or args.output,
        seed=args.seed,
    )
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout
    )
    for metric in result.metrics:
        metric.update(
            git_sha=sha,
            git_dirty=dirty,
            command=shlex.join(
                [
                    "python",
                    "-m",
                    "downscaling.counterfactual",
                    "--seed",
                    str(args.seed),
                    "--output",
                    args.output,
                    "--run-ref",
                    args.run_ref or args.output,
                ]
            ),
        )
    print(result.to_json(args.output))


if __name__ == "__main__":
    main()
