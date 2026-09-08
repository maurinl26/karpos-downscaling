#!/usr/bin/env python3
"""Fit QDM monthly transfer functions on (CERRA-lapse, Sencrop) point pairs.

C4.1 (issue maurinl26/karpos-downscaling#32). Fixes the placeholder
`pipe.calibrate(ref_ds, ref_ds)` in `recalibrate_karpos_slr.py`.

Pools (predicted_lapse_at_station, sencrop_tmin) point pairs across train years
× stations, fits `QuantileDeltaMapping(kind='delta', by_month=True)` and saves
as joblib. The fitted QDM is then loaded by `recalibrate_karpos_slr.py` via
`--qdm-joblib` and applied **after lapse-rate**, **before** RBF residual.

Inputs
------

    --cerra-atm-dir   <dir>      # contains cerra_atm_{year}.nc files
    --cerra-orog      <path|s3>  # CERRA orography (time-invariant)
    --dem             <NetCDF>   # 1 km DEM
    --sencrop         <bulk>     # Sencrop bulk root (local or s3://)
    --years           Y1 Y2 ...  # training years (e.g. 2022 2023 2024 2025)
    --out             <joblib>   # output joblib path

Output
------

    <out>.joblib                 # QuantileDeltaMapping with 12 monthly transfer functions
    <out>.metadata.json          # train years, n_pairs, git SHA, command

Notes
-----

- Lapse-rate downscaling runs WITHOUT QDM and WITHOUT RBF residual.
- Sampling at station = nearest-neighbor on the 1 km lapse-downscaled grid.
- Pooled pairs are flattened into 1D DataArrays with a time coord (samples
  may repeat dates across stations — QDM uses `time.dt.month` for the
  monthly stratification, not the unique values).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xarray as xr

from downscaling.karpos_slr.pipeline import KarposSLRPipeline
from downscaling.karpos_slr.quantile_mapping import QuantileDeltaMapping
from downscaling.prtihvi_wxc.sencrop import load_stations_catalog, load_timeseries
from downscaling.utils.artifacts import write_artifact_metadata

log = logging.getLogger("calibrate_qdm")


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except subprocess.CalledProcessError:
        return "unknown"


def _nightly_tmin(da: xr.DataArray) -> xr.DataArray:
    """Same convention as recalibrate_karpos_slr: shift -9h, daily min, time = morning date."""
    if (
        "valid_time" in da.dims
        and "time" not in da.dims
        or "valid_time" in da.coords
        and "time" not in da.coords
    ):
        da = da.rename({"valid_time": "time"})
    if da["time"].dtype.kind != "M":
        ref = pd.Timestamp("1900-01-01")
        da = da.assign_coords(time=ref + pd.to_timedelta(da["time"].values, unit="h"))
    da = da.assign_coords(time=da["time"] - pd.Timedelta("9h"))
    return da.resample(time="1D").min()


def _maybe_download_s3(uri_or_path: str, tmp_name: str) -> Path:
    if uri_or_path.startswith("s3://"):
        import s3fs

        local = Path(tempfile.gettempdir()) / tmp_name
        if local.exists():
            log.info("Réutilise %s (déjà téléchargé)", local)
            return local
        log.info("Téléchargement %s → %s", uri_or_path, local)
        fs = s3fs.S3FileSystem(
            endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("AWS_S3_ENDPOINT"),
        )
        fs.get(uri_or_path.replace("s3://", "", 1), str(local))
        return local
    return Path(uri_or_path)


def _load_cerra_year(cerra_path: Path, year: int) -> xr.DataArray:
    """Open CERRA atm yearly NetCDF, return nightly Tmin DataArray in °C, lat/lon dims."""
    ds = xr.open_dataset(cerra_path)
    t_var = next((v for v in ("t2m", "2t", "temperature_2m") if v in ds), None)
    if t_var is None:
        raise ValueError(f"No T2m-like variable in {cerra_path}")

    nightly = _nightly_tmin(ds[t_var])
    nightly = nightly.where(nightly.time.dt.year == year, drop=True)

    src_units = str(ds[t_var].attrs.get("units", "")).lower()
    nightly_median = float(np.nanmedian(nightly.values))
    if src_units in ("k", "kelvin") or nightly_median > 100:
        log.info("  %d: K → °C (median raw %.1f K)", year, nightly_median)
        nightly = nightly - 273.15
    nightly.attrs["units"] = "degC"

    rename = {}
    if "latitude" in nightly.dims:
        rename["latitude"] = "lat"
    if "longitude" in nightly.dims:
        rename["longitude"] = "lon"
    if rename:
        nightly = nightly.rename(rename)
    return nightly


def _maybe_init_wandb(args, bbox, stations_df, git_sha):
    """Initialise un run W&B si WANDB_PROJECT/WANDB_API_KEY est défini, sinon None.

    Pattern miroir de ``deep_learning/train.py::_build_logger`` mais en one-shot
    (pas via Lightning) : ``wandb.init()`` direct, ``run.log()`` pour les
    métriques calibration, ``run.log_artifact()`` pour le joblib en sortie,
    ``run.finish()`` à la fin.

    Gracieux : toute exception (wandb absent, network down, etc.) → log warning
    + retour None. Le script continue, la calibration tourne normalement, juste
    sans tracking — pas de régression sur les calibrations dev locales.

    Returns
    -------
    wandb.Run | None
    """
    if not (os.environ.get("WANDB_PROJECT") or os.environ.get("WANDB_API_KEY")):
        log.info("WANDB_PROJECT/WANDB_API_KEY absent — calibration sans W&B tracking.")
        return None
    try:
        import wandb  # noqa: PLC0415 — lazy, pas requis si tracking off
    except ImportError:
        log.warning("wandb non installé (pip install wandb) — calibration sans tracking.")
        return None
    try:
        return wandb.init(
            project=os.environ.get("WANDB_PROJECT", "karpos-downscaling"),
            entity=os.environ.get("WANDB_ENTITY"),
            job_type="calibrate-qdm",
            tags=["qdm", "karpos-slr", "calibration"] + [f"year-{y}" for y in args.years],
            config={
                # Hyperparams calibration
                "n_quantiles": args.n_quantiles,
                "kind": "delta",
                "by_month": True,
                # Datasets sources
                "years": args.years,
                "cerra_atm_dir": str(args.cerra_atm_dir),
                "cerra_orog": str(args.cerra_orog),
                "dem": str(args.dem),
                "sencrop_root": str(args.sencrop),
                # Géo
                "bbox_lat_min": bbox["lat_min"],
                "bbox_lat_max": bbox["lat_max"],
                "bbox_lon_min": bbox["lon_min"],
                "bbox_lon_max": bbox["lon_max"],
                "n_stations_bbox": int(len(stations_df)),
                # Reproductibilité — cf. feedback_wandb_reproducibility
                "git_sha": git_sha,
                "command": "uv run " + " ".join(sys.argv),
            },
        )
    except Exception as exc:  # noqa: BLE001 — auth/network/config, on dégrade
        log.warning("wandb.init() échoué (%s) — calibration sans tracking.", exc)
        return None


def _load_cerra_orog(orog_path: Path) -> xr.DataArray:
    """Load CERRA orography, normalize to lat/lon dims, drop time dim if present."""
    orog_ds = xr.open_dataset(orog_path)
    orog_da = None
    for orog_name in ("orography", "orog", "z", "surface_geopotential"):
        if orog_name in orog_ds:
            orog_da = orog_ds[orog_name]
            if orog_name in ("z", "surface_geopotential"):
                orog_da = orog_da / 9.80665
            for tdim in ("valid_time", "time"):
                if tdim in orog_da.dims:
                    orog_da = orog_da.isel({tdim: 0}, drop=True)
            rename = {}
            if "latitude" in orog_da.dims:
                rename["latitude"] = "lat"
            if "longitude" in orog_da.dims:
                rename["longitude"] = "lon"
            if rename:
                orog_da = orog_da.rename(rename)
            orog_da = orog_da.rename("orog")
            log.info(
                "Orographie: var=%s, shape=%s, mean=%.1f m",
                orog_name,
                orog_da.shape,
                float(np.nanmean(orog_da.values)),
            )
            return orog_da
    raise ValueError(f"No known orog variable in {orog_path}")


def main() -> int:
    p = argparse.ArgumentParser(
        description="Calibrate QDM on Sencrop vs lapse-rate downscaled CERRA"
    )
    p.add_argument(
        "--cerra-atm-dir",
        type=Path,
        required=True,
        help="Directory containing cerra_atm_<year>.nc files",
    )
    p.add_argument(
        "--cerra-orog", type=str, required=True, help="CERRA orography NetCDF (local or s3://)"
    )
    p.add_argument("--dem", type=Path, required=True)
    p.add_argument("--sencrop", type=str, required=True, help="bulk root (local or s3://)")
    p.add_argument(
        "--years",
        type=int,
        nargs="+",
        required=True,
        help="Training years (e.g. 2022 2023 2024 2025)",
    )
    p.add_argument(
        "--out", type=Path, required=True, help="Output joblib path (parent dir created if missing)"
    )
    p.add_argument("--n-quantiles", type=int, default=100)
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load orography + setup pipeline
    orog_local = _maybe_download_s3(args.cerra_orog, "cerra_orography.nc")
    orog_da = _load_cerra_orog(orog_local)

    pipe = KarposSLRPipeline(
        dem_path=args.dem,
        obs_ref_path=None,
        use_qdm=False,  # lapse-rate only, no QDM, no RBF
    )

    # 2. Load Sencrop stations catalog (use the union bbox across all CERRA grids)
    #    We pull stations once from the catalog filtered by the first year's bbox.
    first_cerra = args.cerra_atm_dir / f"cerra_atm_{args.years[0]}.nc"
    if not first_cerra.exists():
        log.error("CERRA atm not found: %s", first_cerra)
        return 2
    first_nightly = _load_cerra_year(first_cerra, args.years[0])
    bbox = {
        "lat_min": float(first_nightly.lat.min()),
        "lat_max": float(first_nightly.lat.max()),
        "lon_min": float(first_nightly.lon.min()),
        "lon_max": float(first_nightly.lon.max()),
    }
    log.info("Bbox stations: %s", bbox)
    stations_df = load_stations_catalog(args.sencrop, bbox=bbox)
    log.info("Stations Sencrop dans bbox: %d", len(stations_df))
    bucket_ids = [int(r["bucket_id"]) for _, r in stations_df.iterrows()]

    # Init W&B run après que bbox + stations soient connus (config complète).
    # `run` est None si tracking off → tous les sites `if run:` deviennent no-op.
    git_sha = _git_sha()
    run = _maybe_init_wandb(args, bbox, stations_df, git_sha)

    # 3. Iterate years, collect (date, station, pred_lapse, obs_sencrop) point pairs
    records: list[dict] = []
    for year in args.years:
        cerra_path = args.cerra_atm_dir / f"cerra_atm_{year}.nc"
        if not cerra_path.exists():
            log.warning("Skip year %d: %s absent", year, cerra_path)
            continue
        log.info("--- year %d ---", year)
        nightly = _load_cerra_year(cerra_path, year)

        # Lapse-rate downscaled grid per night
        ts_sencrop = load_timeseries(
            years=[year], root=args.sencrop, station_only=True, bucket_ids=bucket_ids
        )
        ts_sencrop["timestamp"] = pd.to_datetime(ts_sencrop["timestamp"], utc=True)
        ts_sencrop["night_date"] = (ts_sencrop["timestamp"] - pd.Timedelta("9h")).dt.date
        obs_per_night = (
            ts_sencrop.groupby(["night_date", "station_id"])["temperature"].min().reset_index()
        )

        for d in nightly.time.values:
            d_py = pd.Timestamp(d).date()
            slab = nightly.sel(time=d)
            ds_slab = xr.Dataset({"t2m": slab, "orog": orog_da})
            try:
                grid_fine = pipe.run(ds_slab, variables=["t2m"])["t2m"]
            except Exception as exc:
                log.warning("  %s: pipe.run failed (%s), skip", d_py, exc)
                continue

            # Sample at each station's nearest grid cell
            lat_grid = (
                grid_fine["lat"].values
                if "lat" in grid_fine.coords
                else grid_fine["latitude"].values
            )
            lon_grid = (
                grid_fine["lon"].values
                if "lon" in grid_fine.coords
                else grid_fine["longitude"].values
            )
            arr = grid_fine.values.squeeze()  # (H, W)

            night_obs = obs_per_night[obs_per_night["night_date"] == d_py]
            if night_obs.empty:
                continue
            for _, r in stations_df.iterrows():
                sid = int(r["bucket_id"])
                obs_row = night_obs[night_obs["station_id"] == sid]
                if obs_row.empty:
                    continue
                ii = int(np.argmin(np.abs(lat_grid - float(r["latitude"]))))
                jj = int(np.argmin(np.abs(lon_grid - float(r["longitude"]))))
                pred = float(arr[ii, jj])
                obs = float(obs_row["temperature"].iloc[0])
                if np.isnan(pred) or np.isnan(obs):
                    continue
                records.append(
                    {
                        "date": pd.Timestamp(d_py),
                        "station_id": sid,
                        "pred_lapse": pred,
                        "obs_sencrop": obs,
                    }
                )
        log.info("  total pairs so far: %d", len(records))

    if not records:
        log.error("No point pairs collected — aborting")
        return 2

    df = pd.DataFrame(records).sort_values("date").reset_index(drop=True)
    log.info(
        "Pool final: %d (date, station) pairs sur %d années",
        len(df),
        len(df["date"].dt.year.unique()),
    )

    # 4. Build 1D DataArrays with time coord (samples may repeat dates across stations).
    modeled = xr.DataArray(
        df["pred_lapse"].values.astype(np.float32),
        dims=("time",),
        coords={"time": pd.DatetimeIndex(df["date"].values)},
    )
    observed = xr.DataArray(
        df["obs_sencrop"].values.astype(np.float32),
        dims=("time",),
        coords={"time": pd.DatetimeIndex(df["date"].values)},
    )

    # 5. Fit QDM monthly delta
    log.info(
        "Fit QuantileDeltaMapping(kind='delta', by_month=True, n_quantiles=%d)…", args.n_quantiles
    )
    qdm = QuantileDeltaMapping(kind="delta", by_month=True, n_quantiles=args.n_quantiles)
    qdm.fit(modeled, observed)

    # Sanity check: per-month bias diff + collecte pour W&B
    log.info("Bilan calibration par mois (médiane lapse, médiane Sencrop, delta):")
    per_month_stats: dict[int, dict[str, float]] = {}
    for m in range(1, 13):
        mask = df["date"].dt.month == m
        if mask.sum() < 10:
            continue
        med_pred = float(np.nanmedian(df.loc[mask, "pred_lapse"]))
        med_obs = float(np.nanmedian(df.loc[mask, "obs_sencrop"]))
        delta = med_obs - med_pred
        log.info(
            "  mois %2d : n=%4d · lapse %+6.2f · sencrop %+6.2f · Δ %+6.2f",
            m,
            int(mask.sum()),
            med_pred,
            med_obs,
            delta,
        )
        per_month_stats[m] = {
            "n": int(mask.sum()),
            "median_lapse": med_pred,
            "median_sencrop": med_obs,
            "delta": delta,
        }

    # 6. Save joblib + metadata
    joblib.dump(qdm, args.out)
    log.info("QDM sauvegardée : %s", args.out)

    metadata = {
        "artifact_type": "qdm",
        "fit_protocol": {
            "kind": qdm.kind,
            "by_month": qdm.by_month,
            "n_quantiles": qdm.n_quantiles,
            "wet_threshold": qdm.wet_threshold,
        },
        "command": "uv run " + " ".join(sys.argv),
        "git_sha": git_sha,
        "years": args.years,
        "n_pairs": int(len(df)),
        "n_stations_bbox": int(len(stations_df)),
        "bbox": bbox,
        "n_quantiles": args.n_quantiles,
        "cerra_atm_dir": str(args.cerra_atm_dir),
        "cerra_orog": str(args.cerra_orog),
        "dem": str(args.dem),
        "sencrop_root": str(args.sencrop),
        "per_month_n": {int(m): int((df["date"].dt.month == m).sum()) for m in range(1, 13)},
    }
    meta_path = write_artifact_metadata(args.out, metadata)
    log.info("Metadata : %s", meta_path)

    # 7. Push métriques + joblib en artifact vers W&B (no-op si run is None)
    if run is not None:
        run.summary["n_pairs"] = int(len(df))
        run.summary["n_stations_bbox"] = int(len(stations_df))
        run.summary["n_years"] = len(args.years)
        run.summary["abs_delta_max"] = max(
            (abs(s["delta"]) for s in per_month_stats.values()), default=0.0
        )
        run.summary["abs_delta_mean"] = (
            float(np.mean([abs(s["delta"]) for s in per_month_stats.values()]))
            if per_month_stats
            else 0.0
        )
        # Métriques par mois (panneau "calibration/month_XX/..." dans W&B)
        for m, s in per_month_stats.items():
            run.log(
                {
                    f"calibration/month_{m:02d}/n_pairs": s["n"],
                    f"calibration/month_{m:02d}/median_lapse": s["median_lapse"],
                    f"calibration/month_{m:02d}/median_sencrop": s["median_sencrop"],
                    f"calibration/month_{m:02d}/delta": s["delta"],
                    "month": m,
                },
                step=m,
            )
        # Joblib en artifact (versioné, downloadable, link direct dans le run)
        try:
            import wandb  # noqa: PLC0415 — déjà importé en _maybe_init_wandb si run existe

            artifact = wandb.Artifact(
                name=f"qdm-baronnies-{git_sha[:7]}",
                type="qdm-mapper",
                description=(
                    f"QDM monthly delta calibrated on {len(args.years)} years "
                    f"({args.years[0]}-{args.years[-1]}) × "
                    f"{len(stations_df)} Sencrop stations Baronnies. "
                    f"n_pairs={len(df)}, n_quantiles={args.n_quantiles}."
                ),
                metadata=metadata,
            )
            artifact.add_file(str(args.out), name=args.out.name)
            artifact.add_file(str(meta_path), name=meta_path.name)
            run.log_artifact(artifact)
            log.info("Joblib + metadata loggés comme artifact W&B : %s", artifact.name)
        except Exception as exc:  # noqa: BLE001 — bénin
            log.warning("Échec log_artifact (%s) — joblib local OK, juste pas dans W&B.", exc)
        finally:
            run.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
