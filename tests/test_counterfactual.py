"""Local climate contrasts: synthetic oracle, calendar checks and provenance."""

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscaling.counterfactual import CounterfactualResult, analyze, annual_index


def _climate(start=2001, shift=0, scale=1):
    time = pd.date_range(f"{start}-01-01", f"{start + 7}-12-31", freq="D")
    # Identical annual cycles, leap days excluded from the spring analysis.
    values = scale * (time.day.to_numpy() - 16) + shift
    return xr.DataArray(
        values.astype(float), dims="time", coords={"time": time}, attrs={"units": "degC"}
    )


def _run(**kwargs):
    hist = _climate()
    obs = hist * 2 + 1
    obs.attrs["units"] = "degC"
    return analyze(
        hist,
        obs,
        _climate(2041, shift=2, scale=1.3),
        index="spring_tmin",
        baseline="synthetic observed reference",
        reference="fixture-v1",
        run_ref="test-counterfactual-v1",
        n_bootstrap=100,
        **kwargs,
    )


def test_shift_and_variance_change_matches_analytic_mean():
    metrics = _run().metrics
    x = _climate(2041)
    expected = annual_index(x, "spring_tmin")
    expected_target = float(expected.mean()) * 2.3 + 3
    assert metrics[1]["value"] == pytest.approx(expected_target, abs=0.05)
    assert metrics[2]["value"] == pytest.approx(metrics[1]["value"] - metrics[0]["value"])
    assert metrics[1]["value"] != pytest.approx(float(expected.mean()) * 2.6 + 5, abs=0.1)


def test_bootstrap_reproducible_and_all_metrics_traced(tmp_path):
    result = _run(block_years=2, seed=127)
    assert result.to_dict() == _run(block_years=2, seed=127).to_dict()
    path = result.to_json(tmp_path / "result.json")
    data = json.loads(path.read_text())
    for metric in data["metrics"]:
        assert all(metric[k] for k in ("protocol", "baseline", "reference", "run_ref"))
        assert metric["bootstrap"]["qdm_refitted"] is False
        assert len(metric["confidence_interval"]) == 2
    with pytest.raises(ValueError, match="baseline"):
        CounterfactualResult(({"protocol": "p"},)).to_json(tmp_path / "bad.json")
    assert not (tmp_path / "bad.json").exists()
    missing_run = dict(result.metrics[0])
    del missing_run["run_ref"]
    with pytest.raises(ValueError, match="run_ref"):
        CounterfactualResult((missing_run,)).to_json(tmp_path / "missing-run.json")
    assert not (tmp_path / "missing-run.json").exists()


def test_bootstrap_captures_interannual_variability():
    from downscaling.counterfactual.analysis import _bootstrap_means

    annual = xr.DataArray(np.arange(20.0), dims="year")
    independent = _bootstrap_means(annual, 500, 1, np.random.default_rng(0))
    blocks = _bootstrap_means(annual, 500, 4, np.random.default_rng(0))
    assert blocks.std() > independent.std() > 0


@pytest.mark.parametrize("index", ["spring_tmin", "frost_days", "daytime_tmax"])
def test_indices_and_kelvin(index):
    c = _climate()
    kelvin = c + 273.15
    kelvin.attrs["units"] = "K"
    xr.testing.assert_allclose(annual_index(c, index), annual_index(kelvin, index))
    if index == "frost_days":
        np.testing.assert_allclose(annual_index(c, index), 45)


def test_no_change_has_zero_contrast():
    result = analyze(
        _climate(),
        _climate(),
        _climate(2041),
        index="frost_days",
        baseline="station baseline",
        reference="synthetic-v1",
        run_ref="test-no-change-v1",
        n_bootstrap=20,
    )
    assert result.metrics[2]["value"] == 0


@pytest.mark.parametrize("failure", ["missing", "nan", "duplicate", "units"])
def test_invalid_daily_input_rejected(failure):
    series = _climate()
    if failure == "missing":
        series = series.drop_sel(time="2001-03-15")
    elif failure == "nan":
        series.loc[{"time": "2001-03-15"}] = np.nan
    elif failure == "duplicate":
        series = xr.concat([series, series.isel(time=[0])], dim="time")
    else:
        series.attrs.clear()
    with pytest.raises(ValueError):
        annual_index(series, "frost_days")


def test_invalid_bootstrap_rejected():
    with pytest.raises(ValueError, match="more years"):
        _run(block_years=8)
