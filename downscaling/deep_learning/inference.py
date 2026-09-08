"""
Inférence du modèle de descente d'échelle sur nouvelles données ERA5 / CERRA.

Gestion des grands domaines
----------------------------
Pour les domaines dépassant la taille des tuiles d'entraînement, on utilise
une inférence par tuiles avec chevauchement et recombinaison par moyenne
pondérée (fenêtre de Hann) pour éviter les artefacts aux jointures.

Usage CLI
---------
    python -m downscaling.deep_learning.inference \
        --checkpoint checkpoints/best_model.pt \
        --era5-sl    data/era5/era5_sl_20210427.nc \
        --dem-attrs  data/dem/dem_attributes.nc \
        --stats      checkpoints/normalization_stats.json \
        --out        output/dl_downscaled_20210427.nc
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import xarray as xr

try:
    import torch
except ImportError as e:
    raise ImportError("PyTorch requis : pip install torch") from e

from downscaling.config import load_config

from .dataset import DEFAULT_MET_VARS, prepare_inference_batch
from .model import build_model

log = logging.getLogger(__name__)


def _sha256_file(path: str | Path) -> str:
    """Hash a local inference artefact for output lineage."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Inférence par tuiles avec recombinaison Hann
# ---------------------------------------------------------------------------


def hann_window_2d(size: int) -> np.ndarray:
    """Fenêtre de Hann 2D pour la recombinaison sans artefacts."""
    w1d = np.hanning(size).astype(np.float32)
    return np.outer(w1d, w1d)


def _tile_starts(length: int, tile_size: int, stride: int) -> list[int]:
    """Return starts that cover the complete axis, including the last edge."""
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    last = length - tile_size
    if starts[-1] != last:
        starts.append(last)
    return starts


def tiled_inference(
    model: torch.nn.Module,
    x_met: torch.Tensor,
    x_dem: torch.Tensor,
    tile_size: int = 64,
    overlap: int = 16,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Découpe x_met et x_dem en tuiles, applique le modèle, recombine.

    Parameters
    ----------
    x_met:
        Champs météo (1, C_met, H, W) — déjà normalisés.
    x_dem:
        Attributs MNT (1, C_dem, H, W) — déjà normalisés.
    tile_size:
        Taille des tuiles (pixels). Doit correspondre à la taille d'entraînement.
    overlap:
        Chevauchement entre tuiles (pixels).

    Returns
    -------
    torch.Tensor (1, C_out, H, W) — reconstruction complète.
    """
    if x_met.ndim != 4 or x_dem.ndim != 4 or x_met.shape[0] != 1 or x_dem.shape[0] != 1:
        raise ValueError("tiled_inference attend deux tenseurs de forme (1, C, H, W)")
    if x_met.shape[-2:] != x_dem.shape[-2:]:
        raise ValueError("Les grilles météo et MNT doivent avoir la même forme")
    if tile_size <= 0:
        raise ValueError("tile_size doit être strictement positif")
    if not 0 <= overlap < tile_size:
        raise ValueError("overlap doit être compris entre 0 et tile_size exclus")

    _, C_met, H, W = x_met.shape
    pad_h = max(0, tile_size - H)
    pad_w = max(0, tile_size - W)
    if pad_h or pad_w:
        import torch.nn.functional as F

        x_met = F.pad(x_met, (0, pad_w, 0, pad_h), mode="replicate")
        x_dem = F.pad(x_dem, (0, pad_w, 0, pad_h), mode="replicate")
    Hp, Wp = x_met.shape[-2:]
    _, C_out, _, _ = _infer_output_shape(model, C_met, x_dem.shape[1], tile_size, device)

    output = np.zeros((1, C_out, Hp, Wp), dtype=np.float32)
    weight = np.zeros((1, 1, Hp, Wp), dtype=np.float32)
    # Hann vaut zéro sur le bord ; un epsilon garantit une couverture non nulle
    # lorsque le domaine ne contient qu'une seule tuile.
    win = np.maximum(hann_window_2d(tile_size), np.finfo(np.float32).eps)
    win = win[np.newaxis, np.newaxis, :, :]  # (1, 1, T, T)

    stride = tile_size - overlap
    model.eval()
    with torch.inference_mode():
        for i0 in _tile_starts(Hp, tile_size, stride):
            for j0 in _tile_starts(Wp, tile_size, stride):
                i1, j1 = i0 + tile_size, j0 + tile_size
                tile_met = x_met[:, :, i0:i1, j0:j1].to(device)
                tile_dem = x_dem[:, :, i0:i1, j0:j1].to(device)
                pred = model(tile_met, tile_dem).cpu().numpy()
                output[:, :, i0:i1, j0:j1] += pred * win
                weight[:, :, i0:i1, j0:j1] += win

    # Normalise par le poids
    if np.any(weight <= 0):  # defensive assertion against future tiling changes
        raise RuntimeError("Le tuilage n'a pas couvert toute la grille")
    output /= weight
    return torch.from_numpy(output[:, :, :H, :W])


def _infer_output_shape(
    model: torch.nn.Module, c_in: int, dem_in: int, tile_size: int, device: torch.device
) -> tuple:
    """Infère le nombre de canaux de sortie en faisant passer un batch factice."""
    model.eval()
    with torch.no_grad():
        x = torch.zeros(1, c_in, tile_size, tile_size, device=device)
        d = torch.zeros(1, dem_in, tile_size, tile_size, device=device)
        out = model(x, d)
    return out.shape


# ---------------------------------------------------------------------------
# Pipeline d'inférence complet
# ---------------------------------------------------------------------------


class DLInferencePipeline:
    """
    Charge un checkpoint et applique le modèle sur un fichier ERA5/CERRA.

    Parameters
    ----------
    checkpoint_path:
        Fichier .pt sauvegardé par Trainer (contient model_state_dict).
    config:
        Dictionnaire de configuration (section 'deep_learning').
    stats_path:
        Fichier JSON des statistiques de normalisation.
    device:
        'cuda', 'mps' ou 'cpu'. Auto-détecté si None.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        config: dict,
        stats_path: str | Path,
        device: str | None = None,
    ):
        dl_cfg = config.get("deep_learning", config)
        self.met_vars = dl_cfg.get("met_vars", DEFAULT_MET_VARS)
        self.tile_size = dl_cfg.get("patch_size", 64)
        self.overlap = dl_cfg.get("overlap", 16)

        # Device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = torch.device(device)
        self.checkpoint_path = str(checkpoint_path)
        self.stats_path = str(stats_path)

        # Statistiques de normalisation
        with open(stats_path, encoding="utf-8") as f:
            raw = json.load(f)
        self.stats = {k: tuple(v) for k, v in raw.items()}
        self._validate_stats()

        # Modèle
        self.model = build_model(
            architecture=dl_cfg.get("architecture", "unet"),
            met_in_ch=len(self.met_vars),
            dem_in_ch=dl_cfg.get("dem_in_ch", 4),
            base_ch=dl_cfg.get("base_ch", 64),
            n_levels=dl_cfg.get("n_levels", 4),
            use_film=dl_cfg.get("use_film", True),
        )
        ckpt = torch.load(checkpoint_path, map_location=self.device)
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif "state_dict" in ckpt:
            state_dict = {
                k.removeprefix("model."): v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("model.")
            }
        else:
            raise ValueError("Checkpoint incompatible : model_state_dict/state_dict absent")
        if not state_dict:
            raise ValueError("Checkpoint incompatible : aucun poids modèle trouvé")
        self.model.load_state_dict(state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()
        log.info(f"Checkpoint chargé : {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")

    def _validate_stats(self) -> None:
        required = [*self.met_vars, "elevation", "slope", "aspect", "curvature"]
        missing = [v for v in required if v not in self.stats]
        if missing:
            raise ValueError(f"Statistiques de normalisation manquantes : {', '.join(missing)}")
        for name in required:
            values = self.stats[name]
            if len(values) != 2 or not all(math.isfinite(float(v)) for v in values):
                raise ValueError(f"Statistiques invalides pour {name}: {values!r}")
            if float(values[1]) <= 0:
                raise ValueError(f"Écart-type invalide pour {name}: {values[1]!r}")

    @staticmethod
    def _validate_inputs(coarse_ds: xr.Dataset, dem_ds: xr.Dataset, met_vars: list[str]) -> None:
        missing_met = [v for v in met_vars if v not in coarse_ds]
        if missing_met:
            raise ValueError(f"Variables météo absentes : {', '.join(missing_met)}")
        dem_vars = ("elevation", "slope", "aspect", "curvature")
        missing_dem = [v for v in dem_vars if v not in dem_ds]
        if missing_dem:
            raise ValueError(f"Variables MNT absentes : {', '.join(missing_dem)}")
        dem_shapes = {tuple(dem_ds[v].shape[-2:]) for v in dem_vars}
        shape = next(iter(dem_shapes), (0, 0))
        if len(dem_shapes) != 1 or shape[0] < 1 or shape[1] < 1:
            raise ValueError("Les variables MNT doivent partager une grille 2D non vide")

    def run(
        self,
        coarse_ds: xr.Dataset,
        dem_ds: xr.Dataset,
        output_vars: list[str] | None = None,
    ) -> xr.Dataset:
        """
        Applique le modèle sur toute la série temporelle.

        Parameters
        ----------
        coarse_ds:
            Dataset ERA5/CERRA basse résolution.
        dem_ds:
            Dataset attributs MNT (élévation, pente, aspect, courbure).
        output_vars:
            Variables à inclure dans le Dataset de sortie.
            Défaut = self.met_vars.

        Returns
        -------
        xr.Dataset haute résolution.
        """
        output_vars = output_vars or self.met_vars
        unknown_outputs = set(output_vars) - set(self.met_vars)
        if unknown_outputs:
            names = ", ".join(sorted(unknown_outputs))
            raise ValueError(f"Variables de sortie inconnues : {names}")
        self._validate_inputs(coarse_ds, dem_ds, self.met_vars)
        n_times = len(coarse_ds.time) if "time" in coarse_ds.dims else 1

        # Prépare le tenseur DEM une seule fois (constant dans le temps)
        _, x_dem = prepare_inference_batch(
            coarse_ds,
            dem_ds,
            self.met_vars,
            self.stats,
            time_idx=0,
            device=str(self.device),
        )

        H = x_dem.shape[-2]
        W = x_dem.shape[-1]
        C_out = len(self.met_vars)

        results = np.zeros((n_times, C_out, H, W), dtype=np.float32)

        log.info(f"Inférence sur {n_times} pas de temps…")
        for t in range(n_times):
            x_met, _ = prepare_inference_batch(
                coarse_ds,
                dem_ds,
                self.met_vars,
                self.stats,
                time_idx=t,
                device=str(self.device),
            )
            pred = tiled_inference(
                self.model,
                x_met,
                x_dem,
                tile_size=self.tile_size,
                overlap=self.overlap,
                device=self.device,
            ).numpy()
            if pred.shape[1] != C_out:
                raise ValueError(f"Le modèle produit {pred.shape[1]} canaux, {C_out} attendus")
            results[t] = pred[0]

        # Dénormalisation
        for ci, v in enumerate(self.met_vars):
            if v in self.stats:
                mu, sigma = self.stats[v]
                results[:, ci] = results[:, ci] * sigma + mu

        # Construction du Dataset xarray de sortie
        time_coord = coarse_ds.time if "time" in coarse_ds.dims else None
        lat = dem_ds.coords.get("lat", dem_ds.coords.get("latitude"))
        lon = dem_ds.coords.get("lon", dem_ds.coords.get("longitude"))

        data_vars = {}
        for ci, v in enumerate(self.met_vars):
            if v not in output_vars:
                continue
            if time_coord is not None:
                da = xr.DataArray(
                    results[:, ci],
                    dims=["time", "y", "x"],
                    coords={"time": time_coord},
                )
            else:
                da = xr.DataArray(results[0, ci], dims=["y", "x"])

            if lat is not None and lat.ndim == 2:
                da = da.assign_coords(lat=(["y", "x"], lat.values))
            if lon is not None and lon.ndim == 2:
                da = da.assign_coords(lon=(["y", "x"], lon.values))
            data_vars[v] = da

        ds_out = xr.Dataset(data_vars)
        ds_out.attrs["downscaling_method"] = "deep_learning (DEM-conditioned U-Net)"
        ds_out.attrs["model_checkpoint"] = str(self.checkpoint_path)
        ds_out.attrs["model_checkpoint_sha256"] = _sha256_file(self.checkpoint_path)
        ds_out.attrs["normalization_stats"] = self.stats_path
        ds_out.attrs["normalization_stats_sha256"] = _sha256_file(self.stats_path)
        return ds_out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inférence du modèle DL de descente d'échelle")
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Overrides Hydra (ex: dl.patch_size=128)",
    )
    p.add_argument("--checkpoint", required=True, help="Fichier checkpoint .pt")
    p.add_argument("--era5-sl", required=True, help="ERA5 single-level NetCDF")
    p.add_argument("--dem-attrs", required=True, help="Attributs MNT NetCDF")
    p.add_argument("--stats", required=True, help="JSON statistiques normalisation")
    p.add_argument("--out", required=True, help="Fichier NetCDF de sortie")
    p.add_argument("--device", default=None)
    p.add_argument("--tile-size", type=int, default=None)
    p.add_argument("--overlap", type=int, default=16)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main():
    args = _build_parser().parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    cfg = load_config(args.override)

    if args.tile_size:
        cfg.setdefault("deep_learning", {})["patch_size"] = args.tile_size
    cfg.setdefault("deep_learning", {})["overlap"] = args.overlap

    pipeline = DLInferencePipeline(
        checkpoint_path=args.checkpoint,
        config=cfg,
        stats_path=args.stats,
        device=args.device,
    )

    coarse_ds = xr.open_dataset(args.era5_sl, engine="netcdf4")
    dem_ds = xr.open_dataset(args.dem_attrs, engine="netcdf4")

    ds_out = pipeline.run(coarse_ds, dem_ds)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(args.out)
    log.info(f"Sortie écrite dans {args.out}")


if __name__ == "__main__":
    main()
