"""Climate contrasts, not causal attribution to anthropogenic forcing.

Reference observations define the baseline climate. QDM transfers the simulated
reference-to-target change onto that baseline. Bootstrap intervals condition on
that fitted transfer and quantify interannual sampling uncertainty only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from downscaling.karpos_slr.quantile_mapping import QuantileDeltaMapping

METHOD_REFERENCE = "Cannon et al. (2015), doi:10.1175/JCLI-D-14-00754.1"


def annual_index(
    series: xr.DataArray,
    index: str,
    *,
    months: tuple[int, ...] = (3, 4, 5),
    frost_threshold: float = 0.0,
) -> xr.DataArray:
    """Aggregate complete seasons of daily local Tmin or daytime Tmax.

    ``spring_tmin`` and ``daytime_tmax`` are annual seasonal means of supplied
    daily extrema; ``frost_days`` counts Tmin strictly below a Celsius threshold.
    Inputs must be one-dimensional, daily at midnight, Gregorian, in degC or K.
    Tmax must already have been computed over the caller's daytime window.
    Missing days/values, duplicate dates and partial seasons are rejected.
    All selected months belong to the same calendar year (no cross-year season).
    """
    if index not in {"spring_tmin", "frost_days", "daytime_tmax"}:
        raise ValueError("Unknown index")
    if not months or len(set(months)) != len(months) or any(m not in range(1, 13) for m in months):
        raise ValueError("months must contain distinct calendar months")
    if not np.isfinite(frost_threshold):
        raise ValueError("frost_threshold must be finite")
    if series.dims != ("time",) or series.size == 0:
        raise ValueError("Expected a nonempty local daily series with dims=('time',)")
    time = pd.DatetimeIndex(series.time.values)
    if (
        time.hasnans
        or not time.is_unique
        or not time.is_monotonic_increasing
        or not time.equals(time.normalize())
    ):
        raise ValueError("Dates must be unique, increasing daily midnights")
    units = series.attrs.get("units")
    if units not in {"degC", "K"}:
        raise ValueError("Explicit temperature units required: degC or K")
    selected = series.sel(time=series.time.dt.month.isin(months))
    if selected.size == 0 or not np.isfinite(selected.values).all():
        raise ValueError("Selected seasons must contain finite data")
    values, years = [], np.unique(selected.time.dt.year.values)
    for year in years:
        block = selected.sel(time=selected.time.dt.year == year)
        expected = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
        expected = expected[expected.month.isin(months)]
        if not pd.DatetimeIndex(block.time.values).equals(expected):
            raise ValueError(f"Incomplete season for year {year}")
        celsius = block.values - (273.15 if units == "K" else 0)
        values.append(
            float(np.sum(celsius < frost_threshold) if index == "frost_days" else np.mean(celsius))
        )
    return xr.DataArray(
        values,
        dims="year",
        coords={"year": years},
        attrs={"units": "days/year" if index == "frost_days" else "degC"},
    )


@dataclass(frozen=True)
class CounterfactualResult:
    """Every serialized estimate carries protocol, baseline and reference (#115)."""

    metrics: tuple[dict, ...]

    def to_dict(self) -> dict:
        """Return a JSON-ready versioned report with required provenance checked."""
        for metric in self.metrics:
            for key in ("protocol", "baseline", "reference", "run_ref"):
                if not metric.get(key):
                    raise ValueError(f"Missing metric provenance: {key}")
        return {"schema_version": 1, "metrics": list(self.metrics)}

    def to_json(self, path: str | Path) -> Path:
        """Persist estimates and uncertainty with their complete provenance."""
        payload = json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n"
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        return path


def _bootstrap_means(annual, n_bootstrap, block_years, rng):
    """Moving blocks of consecutive whole years, sampled independently by period."""
    n = annual.size
    starts = rng.integers(0, n - block_years + 1, size=(n_bootstrap, int(np.ceil(n / block_years))))
    positions = (starts[..., None] + np.arange(block_years)).reshape(n_bootstrap, -1)[:, :n]
    return annual.values[positions].mean(axis=1)


def analyze(
    modeled_reference: xr.DataArray,
    observed_reference: xr.DataArray,
    modeled_target: xr.DataArray,
    *,
    index: str,
    baseline: str,
    reference: str,
    run_ref: str,
    months: tuple[int, ...] = (3, 4, 5),
    frost_threshold: float = 0.0,
    n_bootstrap: int = 1000,
    block_years: int = 1,
    confidence: float = 0.95,
    seed: int = 0,
    n_quantiles: int = 100,
) -> CounterfactualResult:
    """Estimate baseline, QDM target and target-minus-baseline climate contrast.

    ``baseline`` names the reference climate/data; ``reference`` identifies the
    observation source/version; ``run_ref`` identifies the reproducible run.
    Periods and method are also recorded per metric.
    Model and observations must share reference dates; target dates are disjoint.
    Inputs may span different numbers of years. Bootstrap samples whole seasonal
    indices, preserving within-season dependence; block_years > 1 also preserves
    short interannual dependence. The QDM fit is held fixed: intervals exclude
    calibration, model, scenario and observation-system uncertainty. No causal
    attribution to a forcing or event probability attribution is inferred.
    """
    if not baseline.strip() or not reference.strip() or not run_ref.strip():
        raise ValueError("Named baseline, reference and run_ref are required")
    if not isinstance(n_bootstrap, int) or n_bootstrap < 2 or not 0 < confidence < 1:
        raise ValueError("n_bootstrap >= 2 and 0 < confidence < 1 required")
    if not isinstance(block_years, int) or block_years < 1:
        raise ValueError("block_years must be a positive integer")
    arrays = (modeled_reference, observed_reference, modeled_target)
    annual = [
        annual_index(a, index, months=months, frost_threshold=frost_threshold) for a in arrays
    ]
    if not modeled_reference.time.equals(observed_reference.time):
        raise ValueError("Model and observations must share reference dates")
    if np.intersect1d(modeled_reference.time.values, modeled_target.time.values).size:
        raise ValueError("Reference and target periods must be disjoint")
    for a in (annual[1], annual[2]):
        if a.size < max(2, block_years + 1) or np.any(np.diff(a.year.values) != 1):
            raise ValueError("Bootstrap requires consecutive years and more years than block_years")
    # Normalize units before estimating the temperature distributions.
    normalized = []
    for a in arrays:
        b = a.astype(float).copy()
        if a.attrs["units"] == "K":
            b = b - 273.15
        b.attrs["units"] = "degC"
        normalized.append(b.sel(time=b.time.dt.month.isin(months)))
    hist, obs, target = normalized
    qdm = QuantileDeltaMapping(kind="delta", by_month=True, n_quantiles=n_quantiles).fit(hist, obs)
    corrected = qdm.transform(target)
    target_annual = annual_index(corrected, index, months=months, frost_threshold=frost_threshold)
    base_annual = annual[1]
    rng = np.random.default_rng(seed)
    boot_base = _bootstrap_means(base_annual, n_bootstrap, block_years, rng)
    boot_target = _bootstrap_means(target_annual, n_bootstrap, block_years, rng)
    means = [float(base_annual.mean()), float(target_annual.mean())]
    common = {
        "protocol": "local-climate-QDM-monthly-v1/independent-moving-year-block-bootstrap",
        "baseline": baseline,
        "reference": reference,
        "run_ref": run_ref,
        "method_reference": METHOD_REFERENCE,
        "index": index,
        "months": list(months),
        "frost_threshold_degC": frost_threshold,
        "units": base_annual.attrs["units"],
        "reference_years": base_annual.year.values.tolist(),
        "target_years": target_annual.year.values.tolist(),
        "n_reference_years": base_annual.size,
        "n_target_years": target_annual.size,
        "bootstrap": {
            "replicates": n_bootstrap,
            "block_years": block_years,
            "confidence": confidence,
            "seed": seed,
            "qdm_refitted": False,
        },
        "n_quantiles": n_quantiles,
        "interpretation": "simulated climate contrast; not causal forcing attribution",
    }
    alpha = (1 - confidence) / 2
    metrics = []
    for name, value, sample in zip(
        ("reference_expected", "target_expected", "climate_delta"),
        (*means, means[1] - means[0]),
        (boot_base, boot_target, boot_target - boot_base),
        strict=True,
    ):
        metrics.append(
            {
                **common,
                "metric": name,
                "value": value,
                "confidence_interval": np.quantile(sample, [alpha, 1 - alpha]).tolist(),
            }
        )
    return CounterfactualResult(tuple(metrics))
