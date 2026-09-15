"""
Correction de biais par cartographie des quantiles.

Deux méthodes implémentées
--------------------------
EQM — Empirical Quantile Mapping
    Corrige le biais en appliquant la fonction de transfert empirique
    F_obs⁻¹(F_mod(x)).  Simple, robuste, mais ne préserve pas le signal
    de tendance climatique.

QDM — Quantile Delta Mapping  (Cannon et al. 2015)
    Préserve les variations relatives (précipitations) ou absolues (température)
    autour du quantile climatologique. Recommandé pour les projections climatiques
    sur une fenêtre simulée complète ; pas pour une émission météo.

Usage typique
-------------
    # Calibration sur la période 1985-2010 (ERA5 → CERRA comme "observations")
    eqm = EmpiricalQuantileMapping(n_quantiles=100)
    eqm.fit(era5_hist, cerra_hist)
    era5_corrected = eqm.transform(era5_future)

    # QDM pour les précipitations
    qdm = QuantileDeltaMapping(kind="ratio", n_quantiles=100)
    qdm.fit(era5_hist, cerra_hist)
    era5_corrected = qdm.transform(era5_future)
"""

from __future__ import annotations

import warnings

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d


class EmpiricalQuantileMapping:
    """
    Empirical Quantile Mapping (EQM).

    QM météo : calibration modèle/observations contemporaines, sans delta
    climatique ni distribution future. La correction est apprise séparément
    pour chaque mois calendaire afin
    de conserver le cycle saisonnier.

    Parameters
    ----------
    n_quantiles:
        Nombre de quantiles (défaut 100).
    extrapolation:
        Méthode d'extrapolation hors de la plage de calibration :
        'constant' (valeurs extrêmes clampées) ou 'linear'.
    by_month:
        Si True (défaut), ajuste une fonction de transfert par mois calendaire.
    """

    def __init__(
        self,
        n_quantiles: int = 100,
        extrapolation: str = "constant",
        by_month: bool = True,
    ):
        self.n_quantiles = n_quantiles
        self.extrapolation = extrapolation
        self.by_month = by_month
        self._transfer: dict[int, interp1d] = {}  # clé = mois (1-12) ou 0 = global

    # ------------------------------------------------------------------
    def fit(self, modeled: xr.DataArray, observed: xr.DataArray) -> EmpiricalQuantileMapping:
        """
        Calibre la fonction de transfert.

        Parameters
        ----------
        modeled:
            Champ modélisé (ERA5 ou CERRA basse résolution) — période historique.
            Dim temporelle : 'time'.
        observed:
            Champ de référence (CERRA haute résolution, SAFRAN…) — même période.
        """
        months = list(range(1, 13)) if self.by_month else [0]
        quantiles = np.linspace(0, 100, self.n_quantiles + 1)

        for m in months:
            if m == 0:
                mod_vals = modeled.values.ravel()
                obs_vals = observed.values.ravel()
            else:
                mod_vals = modeled.sel(time=modeled.time.dt.month == m).values.ravel()
                obs_vals = observed.sel(time=observed.time.dt.month == m).values.ravel()

            mask = np.isfinite(mod_vals) & np.isfinite(obs_vals)
            mod_q = np.percentile(mod_vals[mask], quantiles)
            obs_q = np.percentile(obs_vals[mask], quantiles)

            fill_val = (obs_q[0], obs_q[-1]) if self.extrapolation == "constant" else "extrapolate"
            self._transfer[m] = interp1d(
                mod_q, obs_q, kind="linear", bounds_error=False, fill_value=fill_val
            )
        return self

    def transform(self, modeled: xr.DataArray) -> xr.DataArray:
        """
        Applique la correction de biais.

        Parameters
        ----------
        modeled:
            Champ à corriger (même variable que lors du fit).

        Returns
        -------
        xr.DataArray corrigé, mêmes dims/coords que `modeled`.
        """
        if not self._transfer:
            raise RuntimeError("Appeler fit() avant transform().")

        result = modeled.copy(deep=True)
        months = list(range(1, 13)) if self.by_month else [0]

        for m in months:
            if m == 0:
                idx = slice(None)
            else:
                idx = modeled.time.dt.month == m
            block = modeled.sel(time=idx)
            tf = self._transfer[m]
            corrected = xr.apply_ufunc(tf, block, dask="parallelized", output_dtypes=[float])
            result.loc[{"time": idx}] = corrected.values

        result.attrs.update(modeled.attrs)
        result.attrs["bias_correction"] = "EQM"
        return result


class QuantileDeltaMapping:
    """
    Quantile Delta Mapping (QDM — Cannon et al. 2015).

    Climat uniquement : fit(mod_ref, obs_ref), puis transform(mod_future) sur
    une fenêtre simulée représentative entière (jamais une émission météo).
    Au rang p = F_mod_future(x), applique obs_ref_ppf(p) + x - mod_ref_ppf(p)
    ou obs_ref_ppf(p) * x / mod_ref_ppf(p). Aucune observation future requise.

    Les distributions sont regroupées par mois calendaire via time.dt.month
    (calendriers xarray, sans appariement des dates) et toutes les dimensions
    spatiales sont mises en commun : sélectionner un site avant le fit local.
    Quantiles linéaires incluant 0 et 1 ; rang moyen des ex æquo. Les queues
    utilisent la correction au quantile extrême, sans extrapoler les PPF.
    Les groupes climatiques non calibrés lèvent une erreur.

    En mode ratio, les CDF sont conditionnelles aux jours humides (> seuil) ;
    les jours secs restent nuls. Le ratio est préservé sur les quantiles humides,
    pas la fréquence humide : aucun ajustement stochastique d'occurrence.
    Utiliser WeatherQuantileMapping pour la calibration opérationnelle QM.

    Parameters
    ----------
    kind:
        'delta' pour variables additives (température), 'ratio' pour multiplicatives
        (précipitations).
    n_quantiles:
        Nombre de quantiles.
    by_month:
        Ajustement mensuel (défaut True).
    wet_threshold:
        Seuil (mm) sous lequel un jour est considéré sec — utilisé uniquement
        si kind='ratio' pour éviter la division par zéro.
    """

    def __init__(
        self,
        kind: str = "delta",
        n_quantiles: int = 100,
        by_month: bool = True,
        wet_threshold: float = 0.1,
    ):
        if kind not in ("delta", "ratio"):
            raise ValueError("kind doit être 'delta' ou 'ratio'.")
        if not isinstance(n_quantiles, int) or n_quantiles < 2:
            raise ValueError("n_quantiles doit être un entier >= 2.")
        if not np.isfinite(wet_threshold) or wet_threshold <= 0:
            raise ValueError("wet_threshold doit être positif et fini.")
        self._climate_version = 1
        self.kind = kind
        self.n_quantiles = n_quantiles
        self.by_month = by_month
        self.wet_threshold = wet_threshold

        # Fonctions de transfert calibrées
        self._mod_cdf: dict[int, interp1d] = {}  # F_mod_ref(x)  → quantile
        self._mod_ppf: dict[int, interp1d] = {}  # F_mod_ref⁻¹(q) → valeur
        self._obs_ppf: dict[int, interp1d] = {}  # F_obs_ref⁻¹(q) → valeur

    # ------------------------------------------------------------------
    def fit(self, modeled_ref: xr.DataArray, observed_ref: xr.DataArray) -> QuantileDeltaMapping:
        """
        Calibre les distributions de référence.

        Parameters
        ----------
        modeled_ref:
            ERA5/CERRA basse résolution sur la période de référence.
        observed_ref:
            Référence haute résolution (CERRA fine, SAFRAN…) — même période.
        """
        self._climate_version = 1
        self._mod_cdf.clear()
        self._mod_ppf.clear()
        self._obs_ppf.clear()
        months = list(range(1, 13)) if self.by_month else [0]
        quantiles = np.linspace(0, 100, self.n_quantiles + 1)
        q01 = quantiles / 100.0  # [0, 1]

        for m in months:
            if m == 0:
                mod_vals = modeled_ref.values.ravel()
                obs_vals = observed_ref.values.ravel()
            else:
                mod_vals = modeled_ref.sel(time=modeled_ref.time.dt.month == m).values.ravel()
                obs_vals = observed_ref.sel(time=observed_ref.time.dt.month == m).values.ravel()

            if self.kind == "ratio":
                mod_vals = mod_vals[mod_vals > self.wet_threshold]
                obs_vals = obs_vals[obs_vals > self.wet_threshold]

            mask_m = np.isfinite(mod_vals)
            mask_o = np.isfinite(obs_vals)
            # Skip months sans données suffisantes (campagne Sencrop limitée Fév-Mai
            # par exemple) — identité météo, erreur explicite en chemin climat.
            if mask_m.sum() < 10 or mask_o.sum() < 10:
                continue
            mod_q = np.percentile(mod_vals[mask_m], quantiles)
            obs_q = np.percentile(obs_vals[mask_o], quantiles)

            # CDF modèle : valeur → quantile
            self._mod_cdf[m] = interp1d(
                mod_q, q01, kind="linear", bounds_error=False, fill_value=(q01[0], q01[-1])
            )
            # PPF modèle : quantile → valeur (nécessaire au terme delta de QDM)
            self._mod_ppf[m] = interp1d(
                q01, mod_q, kind="linear", bounds_error=False, fill_value=(mod_q[0], mod_q[-1])
            )
            # PPF observations : quantile → valeur
            self._obs_ppf[m] = interp1d(
                q01, obs_q, kind="linear", bounds_error=False, fill_value=(obs_q[0], obs_q[-1])
            )
        return self

    def transform(self, modeled_future: xr.DataArray) -> xr.DataArray:
        """Correct an entire simulated climate window using its own empirical CDF.

        Output keeps coordinates, shape and attributes; nonfinite inputs become
        NaN. At least two finite (wet for ratio) samples per active group are
        required. Changing the window changes the ranks and thus the correction.
        Legacy joblib objects retain their historical weather mapping with a
        warning; refit to explicitly opt into climate QDM.
        """
        if not hasattr(self, "_climate_version"):
            warnings.warn(
                "Legacy QDM artifact: using weather QM; refit for climate QDM.",
                UserWarning,
                stacklevel=2,
            )
            return self._transform_weather(modeled_future)
        if not self._mod_ppf:
            raise RuntimeError("Appeler fit() avant transform().")
        result = modeled_future.astype(float).copy(deep=True)
        months = np.unique(modeled_future.time.dt.month) if self.by_month else [0]
        for m in months:
            if m not in self._mod_ppf:
                raise ValueError(
                    f"Climate group {m} has insufficient reference samples (minimum 10)."
                )
            idx = slice(None) if m == 0 else modeled_future.time.dt.month == m
            block = modeled_future.sel(time=idx).values
            valid = np.isfinite(block)
            wet = valid & (block > self.wet_threshold) if self.kind == "ratio" else valid
            corrected = np.full(block.shape, np.nan)
            if self.kind == "ratio":
                corrected[valid & ~wet] = 0.0
            if wet.any():
                values = np.sort(block[wet])
                if values.size < 2:
                    raise ValueError("Climate window needs at least two finite (wet) samples.")
                unique, first, counts = np.unique(values, return_index=True, return_counts=True)
                ranks = (first + (counts - 1) / 2) / (values.size - 1)
                tau = np.interp(block[wet], unique, ranks)
                obs = self._obs_ppf[m](tau)
                hist = self._mod_ppf[m](tau)
                corrected[wet] = (
                    obs + block[wet] - hist if self.kind == "delta" else obs * block[wet] / hist
                )
            result.loc[{"time": idx}] = corrected
        result.attrs["bias_correction"] = f"QDM-{self.kind}"
        result.attrs["rank_distribution"] = "modeled_future"
        return result

    def _transform_weather(self, modeled_future: xr.DataArray) -> xr.DataArray:
        """
        Applique le QM historique météo (compatibilité des calibrateurs).

        Parameters
        ----------
        modeled_future:
            Émission météo à corriger, sans estimation de distribution future.

        Returns
        -------
        xr.DataArray corrigé.
        """
        if not self._mod_cdf:
            raise RuntimeError("Appeler fit() avant transform().")

        result = modeled_future.astype(float).copy(deep=True)
        months = list(range(1, 13)) if self.by_month else [0]

        for m in months:
            # Mois sans fonction de transfert calibrée (fit a skip → données insuffisantes) :
            # on laisse le bloc tel quel (identité). Cohérent avec une campagne Sencrop
            # qui ne couvre que Fév-Mai par exemple.
            if m not in self._mod_cdf:
                continue
            idx = slice(None) if m == 0 else (modeled_future.time.dt.month == m)
            block = modeled_future.sel(time=idx).values  # numpy

            tau = self._mod_cdf[m](block)  # quantile du futur dans la CDF modèle de réf
            x_ref = self._obs_ppf[m](tau)  # valeur bias-corrigée (PPF observations)
            x_mod = self._mod_ppf[m](tau)  # valeur modèle de réf au même quantile

            if self.kind == "delta":
                # QM dans la plage historique ; biais constant au-delà des bornes.
                delta = block - x_mod
                corrected = x_ref + delta
            else:
                # Anomalie multiplicative ; évite la division par zéro.
                denom = np.where(np.abs(x_mod) > self.wet_threshold, x_mod, self.wet_threshold)
                corrected = x_ref * np.where(np.abs(denom) > 0, block / denom, 1.0)
                corrected = np.maximum(corrected, 0.0)

            result.loc[{"time": idx}] = corrected

        result.attrs.update(modeled_future.attrs)
        result.attrs["bias_correction"] = f"QM-weather-{self.kind}"
        return result


class WeatherQuantileMapping(QuantileDeltaMapping):
    """Operational QM fitted on model/observation pairs from the same period.

    Historical ranks only: no climate delta and no future-window dependency.
    Retains the previous calibrators' mapping, including constant additive bias
    (or multiplicative factor) beyond historical bounds, wet-sample fitting and
    identity for uncalibrated months. ``kind`` describes the tail correction,
    not climate trend preservation. EQM above instead clamps tails by default.
    Historical artifact names may still contain 'qdm' for serving compatibility.
    """

    def transform(self, modeled: xr.DataArray) -> xr.DataArray:
        """Apply the fixed historical transfer to any operational batch."""
        return self._transform_weather(modeled)


# ---------------------------------------------------------------------------
# BCSD — Bias Correction & Spatial Disaggregation (version simplifiée)
# ---------------------------------------------------------------------------


def bcsd_temperature(
    t_coarse: xr.DataArray,
    t_ref_coarse: xr.DataArray,
    t_ref_fine: xr.DataArray,
    n_quantiles: int = 100,
) -> xr.DataArray:
    """
    BCSD pour la température :
    1. QDM en basse résolution (ERA5 → CERRA coarse)
    2. Désagrégation spatiale par interpolation bilinéaire +
       anomalie haute fréquence de la référence climatologique.

    Parameters
    ----------
    t_coarse:
        Température ERA5 à corriger/désagréger.
    t_ref_coarse:
        Climatologie de référence basse résolution (même grille que t_coarse).
    t_ref_fine:
        Climatologie de référence haute résolution (grille cible).

    Returns
    -------
    xr.DataArray température haute résolution corrigée.
    """
    # Étape 1 : QDM basse résolution
    qdm = QuantileDeltaMapping(kind="delta", n_quantiles=n_quantiles)
    qdm.fit(t_coarse, t_ref_coarse)
    t_bc = qdm.transform(t_coarse)

    # Étape 2 : interpolation bilinéaire de t_bc sur la grille fine
    lat_fine = t_ref_fine.coords.get("lat", t_ref_fine.coords.get("latitude"))
    lon_fine = t_ref_fine.coords.get("lon", t_ref_fine.coords.get("longitude"))
    lat_name = "latitude" if "latitude" in t_bc.coords else "lat"
    lon_name = "longitude" if "longitude" in t_bc.coords else "lon"

    t_bc_fine = t_bc.interp({lat_name: lat_fine, lon_name: lon_fine}, method="linear")

    # Étape 3 : ajout de l'anomalie spatiale de la référence fine
    # clim_ref_fine_monthly — la variabilité sub-maille
    clim_coarse_monthly = t_ref_coarse.groupby("time.month").mean("time")
    clim_fine_monthly = t_ref_fine.groupby("time.month").mean("time")
    clim_coarse_fine = clim_coarse_monthly.interp(
        {lat_name: lat_fine, lon_name: lon_fine}, method="linear"
    )
    spatial_anomaly = clim_fine_monthly - clim_coarse_fine  # (12, y, x)

    # Ajouter l'anomalie mensuelle correspondante
    months = t_bc_fine.time.dt.month
    anom = spatial_anomaly.sel(month=months).drop_vars("month")
    return (t_bc_fine + anom).rename("t2m_bcsd")
