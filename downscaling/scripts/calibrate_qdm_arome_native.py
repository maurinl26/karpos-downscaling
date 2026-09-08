#!/usr/bin/env python3
"""Fit the AROME-native QDM joblib for serving (voie B, #105 / #70).

Drop-in replacement for the v0 CERRA-calibrated joblib consumed by karpos-engine
``scripts/apply_qdm_to_arome.py``: same ``QuantileDeltaMapping(kind='delta',
by_month=True)`` object, so serving is a joblib swap + ``source_version`` bump to
``v1-arome-native``.

Difference vs v0: calibrated on ``(AROME-HD night-min at station, Sencrop night-min)``
pooled pairs (predictor = the operational model itself, not CERRA-lapse; target =
the assimilated analysis at the station point, which equals the station obs). This
is the GENERALIZED transfer (pooled over stations) served to ungauged parcels.

Monotonicity note: QDM.transform is monotone, so applying it per-hour and taking
the night-min downstream equals applying it to the night-min — consistent with the
``/alerts/frost`` and ``parcels-tmin`` endpoints that reduce AROME to a night min.

Measured lift (LOYO x LOSO, 2023-2025, seuil -2.2 C, baseline AROME HD CSI 0.40):
generalized CSI 0.40->0.46, bias annulé (cf. reports/qdm_arome_native/metrics.json).

Output: <out>.joblib (QuantileDeltaMapping) + <out>.metadata.json.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from downscaling.karpos_slr.quantile_mapping import QuantileDeltaMapping
from downscaling.utils.artifacts import write_artifact_metadata

DEF_SENC = "/Users/loicmaurin/kDrive/karpos_datasets/data/raw/sencrop"
DEF_AR = "/Users/loicmaurin/kDrive/karpos_datasets/output/arome_openmeteo_backfill/arome_hd_stations_2023-2025.parquet"
DEF_OUT = "/Users/loicmaurin/kDrive/karpos_datasets/output/qdm/arome_native_v1.joblib"


def night(d):
    return pd.Timestamp(d) + pd.Timedelta("20h"), pd.Timestamp(d) + pd.Timedelta(
        "1D"
    ) + pd.Timedelta("8h")


def load_obs(senc, years):
    rows = []
    for yr in years:
        sdf = pd.read_csv(sorted(glob.glob(f"{senc}/{yr}.csv/part-*.csv"))[0])
        sdf["timestamp"] = pd.to_datetime(
            sdf["timestamp"], utc=True, errors="coerce"
        ).dt.tz_localize(None)
        sdf = sdf[sdf["temperature_source"] == "station"]
        for d in pd.date_range(f"{yr}-01-01", f"{yr}-04-30", freq="D"):
            a, b = night(d)
            w = sdf[(sdf.timestamp >= a) & (sdf.timestamp <= b)]
            if w.empty:
                continue
            g = w.groupby("station_id")["temperature"].agg(["min", "count"])
            g = g[g["count"] >= 6]
            for st, r in g.iterrows():
                rows.append((str(st), pd.Timestamp(d), float(r["min"])))
    return pd.DataFrame(rows, columns=["station_id", "night", "obs_tmin"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sencrop", default=DEF_SENC)
    p.add_argument("--arome", default=DEF_AR)
    p.add_argument("--out", default=DEF_OUT)
    p.add_argument("--years", nargs="+", type=int, default=[2023, 2024, 2025])
    p.add_argument("--n-quantiles", type=int, default=100)
    a = p.parse_args()

    obs = load_obs(a.sencrop, a.years)
    ar = pd.read_parquet(a.arome)
    ar["station_id"] = ar.station_id.astype(str)
    df = ar.merge(obs, on=["station_id", "night"], how="inner").dropna(
        subset=["arome_tmin", "obs_tmin"]
    )
    print(
        f"{len(df)} pooled (AROME,obs) pairs · {df.station_id.nunique()} stations · years {a.years}"
    )

    # 1D DataArrays with a time coord (samples repeat dates across stations; QDM uses
    # time.dt.month for by_month stratification — cf. calibrate_qdm.py).
    t = pd.DatetimeIndex(df.night.values)
    modeled = xr.DataArray(df.arome_tmin.values, dims=["time"], coords={"time": t})
    observed = xr.DataArray(df.obs_tmin.values, dims=["time"], coords={"time": t})
    qdm = QuantileDeltaMapping(kind="delta", n_quantiles=a.n_quantiles, by_month=True).fit(
        modeled, observed
    )

    # sanity: transform must be monotone and reduce the cold bias
    chk = xr.DataArray(
        np.array([-6.0, -2.0, 0.0, 5.0]),
        dims=["time"],
        coords={"time": pd.date_range("2024-02-01", periods=4)},
    )
    corr = qdm.transform(chk).values

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(qdm, a.out)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        sha = None
    meta = {
        "artifact_type": "qdm",
        "fit_protocol": {
            "kind": qdm.kind,
            "by_month": qdm.by_month,
            "n_quantiles": qdm.n_quantiles,
            "wet_threshold": qdm.wet_threshold,
        },
        "issue": "maurinl26/karpos-downscaling#105",
        "source_version": "v1-arome-native",
        "kind": "delta",
        "by_month": True,
        "n_quantiles": a.n_quantiles,
        "predictor": "AROME-HD Open-Meteo night-min at stations (short-lead, #91)",
        "target": "Sencrop night-min (assimilated analysis at station point)",
        "train_years": a.years,
        "n_pairs": int(len(df)),
        "n_stations": int(df.station_id.nunique()),
        "months_calibrated": sorted(int(m) for m in qdm._mod_cdf),
        "sanity_transform_feb": {"in": [-6, -2, 0, 5], "out": [round(float(x), 2) for x in corr]},
        "git_sha": sha,
        "command": "uv run python -m downscaling.scripts.calibrate_qdm_arome_native",
    }
    write_artifact_metadata(a.out, meta)
    print(
        f"wrote {a.out} · months={meta['months_calibrated']} · sanity Feb {meta['sanity_transform_feb']}"
    )


if __name__ == "__main__":
    main()
