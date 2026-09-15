"""Tests de la correction de biais par cartographie des quantiles.

(``downscaling.karpos_slr.quantile_mapping``)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from downscaling.karpos_slr.quantile_mapping import (
    EmpiricalQuantileMapping,
    QuantileDeltaMapping,
)


def _series(values, start="2000-01-01"):
    values = np.asarray(values, dtype=float)
    time = pd.date_range(start, periods=values.shape[0], freq="D")
    return xr.DataArray(values, dims=["time"], coords={"time": time}, name="t")


def test_eqm_identity_when_distributions_match():
    # modeled == observed → la fonction de transfert est l'identité.
    rng = np.random.default_rng(0)
    data = rng.normal(280.0, 5.0, size=400)
    modeled = _series(data)
    observed = _series(data.copy())
    eqm = EmpiricalQuantileMapping(n_quantiles=100, by_month=False)
    eqm.fit(modeled, observed)
    out = eqm.transform(modeled)
    np.testing.assert_allclose(out.values, modeled.values, atol=1e-6)
    assert out.attrs["bias_correction"] == "EQM"


def test_eqm_corrects_constant_bias():
    # observed = modeled + 3 K partout → EQM doit retirer ~3 K.
    rng = np.random.default_rng(1)
    base = rng.normal(280.0, 4.0, size=500)
    modeled = _series(base)
    observed = _series(base + 3.0)
    eqm = EmpiricalQuantileMapping(n_quantiles=200, by_month=False)
    eqm.fit(modeled, observed)
    out = eqm.transform(modeled)
    # La moyenne corrigée s'approche de la moyenne observée.
    np.testing.assert_allclose(float(out.mean()), float(observed.mean()), atol=0.2)


def test_eqm_transform_before_fit_raises():
    eqm = EmpiricalQuantileMapping(by_month=False)
    with pytest.raises(RuntimeError):
        eqm.transform(_series([1.0, 2.0, 3.0]))


def test_qdm_delta_identity_on_reference():
    # mod_ref == obs_ref, future == ref → QDM-delta ≈ identité.
    rng = np.random.default_rng(2)
    data = rng.normal(285.0, 3.0, size=400)
    ref = _series(data)
    qdm = QuantileDeltaMapping(kind="delta", n_quantiles=100, by_month=False)
    qdm.fit(ref, _series(data.copy()))
    out = qdm.transform(ref)
    np.testing.assert_allclose(out.values, ref.values, atol=1e-6)
    assert out.attrs["bias_correction"] == "QDM-delta"


def test_qdm_delta_corrects_bias_preserving_trend():
    # obs_ref = mod_ref - 2 K ; un futur décalé de +1 K doit garder son anomalie.
    rng = np.random.default_rng(3)
    base = rng.normal(280.0, 2.0, size=600)
    mod_ref = _series(base)
    obs_ref = _series(base - 2.0)
    qdm = QuantileDeltaMapping(kind="delta", n_quantiles=200, by_month=False)
    qdm.fit(mod_ref, obs_ref)
    future = _series(base + 1.0)
    out = qdm.transform(future)
    # Correction du biais (-2) tout en conservant la dérive (+1) → ~ -1 K net.
    np.testing.assert_allclose(float(out.mean() - future.mean()), -2.0, atol=0.3)


def test_qdm_ratio_keeps_precipitation_nonnegative():
    rng = np.random.default_rng(4)
    mod = np.abs(rng.gamma(2.0, 2.0, size=500))
    obs = mod * 1.5  # observations plus humides
    qdm = QuantileDeltaMapping(kind="ratio", n_quantiles=100, by_month=False, wet_threshold=0.1)
    qdm.fit(_series(mod), _series(obs))
    out = qdm.transform(_series(mod))
    assert bool((out.values >= 0.0).all())
    assert out.attrs["bias_correction"] == "QDM-ratio"
    # correction multiplicative → moyenne tirée vers le haut
    assert float(out.mean()) > float(_series(mod).mean())


def test_qdm_invalid_kind_raises():
    with pytest.raises(ValueError):
        QuantileDeltaMapping(kind="nope")


def test_qdm_transform_before_fit_raises():
    qdm = QuantileDeltaMapping(kind="delta", by_month=False)
    with pytest.raises(RuntimeError):
        qdm.transform(_series([1.0, 2.0]))


@pytest.mark.parametrize("scale,shift", [(1, 2), (1.4, 0), (1.4, 2), (1, 0)])
def test_climate_qdm_preserves_quantile_changes_against_cannon_formula(scale, shift):
    # Independent analytical oracle: Q_hist(p)=10p-5; Q_obs(p)=2Q_hist(p)+1.
    x = np.linspace(-5, 5, 1001)
    hist, obs, future = _series(x), _series(2 * x + 1), _series(scale * x + shift)
    out = QuantileDeltaMapping(by_month=False).fit(hist, obs).transform(future)
    expected = 2 * x + 1 + (scale * x + shift - x)
    np.testing.assert_allclose(out, expected, atol=1e-12)  # includes extreme quantiles
    qm = EmpiricalQuantileMapping(by_month=False).fit(hist, obs).transform(future)
    if shift:
        assert float(out.median()) == pytest.approx(1 + shift)
        assert float(qm.median()) == pytest.approx(1 + 2 * shift)


def test_ratio_preserves_wet_quantile_ratio_and_dry_days():
    wet = np.linspace(1, 10, 101)
    hist, obs = _series(np.r_[np.zeros(20), wet]), _series(np.r_[np.zeros(20), wet**2])
    future = _series(np.r_[np.zeros(20), wet * 2, np.nan])
    out = QuantileDeltaMapping(kind="ratio", by_month=False).fit(hist, obs).transform(future)
    np.testing.assert_allclose(out[:20], 0)
    np.testing.assert_allclose(out[20:-1], wet**2 * 2, atol=1e-12)
    assert np.isnan(out[-1])


def test_climate_ties_and_integer_input():
    hist = _series(np.arange(20))
    future = xr.DataArray(np.full(20, 12, dtype=int), dims="time")
    out = QuantileDeltaMapping(by_month=False).fit(hist, hist * 2).transform(future)
    np.testing.assert_allclose(out, 21.5)  # tied future rank = 0.5
    assert out.dtype.kind == "f"


def test_climate_months_refit_and_missing_group():
    jan = _series(np.arange(20))
    feb = _series(np.arange(20), "2000-02-01")
    qdm = QuantileDeltaMapping().fit(jan, jan + 1)
    np.testing.assert_allclose(qdm.transform(jan + 2), jan + 3)
    with pytest.raises(ValueError, match="group 2"):
        qdm.transform(feb)
    qdm.fit(feb, feb + 4)
    with pytest.raises(ValueError, match="group 1"):
        qdm.transform(jan)


def test_weather_serialization_and_batch_invariance(tmp_path):
    import joblib

    from downscaling.karpos_slr.quantile_mapping import WeatherQuantileMapping

    hist = _series(np.linspace(-5, 5, 1001))
    model = WeatherQuantileMapping(by_month=False).fit(hist, hist * 2 + 1)
    path = tmp_path / "weather.joblib"
    joblib.dump(model, path)
    from downscaling.utils.artifacts import load_validated_artifact, write_artifact_metadata

    write_artifact_metadata(
        path,
        {
            "artifact_type": "qdm",
            "fit_protocol": {"kind": "delta", "by_month": False, "n_quantiles": 100},
        },
    )
    loaded, _ = load_validated_artifact(path, expected_type="qdm")
    future = _series([-8, -2, 0, 2, 8])
    # Historical QM inside bounds, constant bias extrapolation outside.
    np.testing.assert_allclose(loaded.transform(future), [-12, -3, 1, 5, 14])
    for i in range(future.size):
        np.testing.assert_allclose(
            loaded.transform(future.isel(time=[i])), loaded.transform(future)[i]
        )


def test_legacy_artifact_keeps_weather_semantics(tmp_path):
    import joblib

    hist = _series(np.linspace(-5, 5, 1001))
    legacy = QuantileDeltaMapping(by_month=False).fit(hist, hist * 2 + 1)
    del legacy._climate_version  # state of artifacts produced before #127
    path = tmp_path / "legacy.joblib"
    joblib.dump(legacy, path)
    with pytest.warns(UserWarning, match="Legacy QDM"):
        out = joblib.load(path).transform(_series([2]))
    assert float(out[0]) == pytest.approx(5)


def test_weather_missing_month_is_identity():
    from downscaling.karpos_slr.quantile_mapping import WeatherQuantileMapping

    jan = _series(np.arange(20))
    feb = _series([2.0], "2000-02-01")
    out = WeatherQuantileMapping().fit(jan, jan + 1).transform(feb)
    np.testing.assert_allclose(out, feb)


def test_climate_preserves_coordinates_and_attributes_for_pooled_grid():
    hist = _series(np.arange(20.0)).expand_dims(site=["a", "b"])
    target = hist + 3
    target.attrs["units"] = "degC"
    out = QuantileDeltaMapping(by_month=False).fit(hist, hist - 1).transform(target)
    assert out.dims == target.dims
    assert out.coords.equals(target.coords)
    assert out.attrs["units"] == "degC"
    np.testing.assert_allclose(out, target - 1)


def test_ratio_constant_wet_ties_and_all_dry_target():
    hist = _series(np.ones(20))
    qdm = QuantileDeltaMapping(kind="ratio", by_month=False).fit(hist, hist * 3)
    np.testing.assert_allclose(qdm.transform(hist * 2), 6)
    np.testing.assert_allclose(qdm.transform(hist * 0), 0)
    with pytest.raises(ValueError, match="two finite"):
        qdm.transform(_series([2]))


def test_weather_ratio_nonregression():
    from downscaling.karpos_slr.quantile_mapping import WeatherQuantileMapping

    hist = _series(np.arange(1.0, 11.0))
    qm = WeatherQuantileMapping(kind="ratio", by_month=False, n_quantiles=9).fit(hist, hist**2)
    np.testing.assert_allclose(qm.transform(_series([0.0, 2.0, 5.0, 12.0])), [0, 4, 25, 120])
