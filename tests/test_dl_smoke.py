"""Small CPU train -> checkpoint -> inference smoke test."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("lightning.pytorch")
xr = pytest.importorskip("xarray")

from downscaling.deep_learning.inference import DLInferencePipeline  # noqa: E402
from downscaling.deep_learning.lightning_module import (  # noqa: E402
    DownscalingDataModule,
    DownscalingLitModule,
)
from downscaling.deep_learning.model import build_model  # noqa: E402
from downscaling.deep_learning.train import build_trainer  # noqa: E402


class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, n: int = 4, size: int = 16):
        generator = torch.Generator().manual_seed(7)
        self.x = torch.randn(n, 5, size, size, generator=generator)
        self.dem = torch.randn(n, 4, size, size, generator=generator)
        self.y = torch.randn(n, 5, size, size, generator=generator)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.dem[index], self.y[index]


def test_cpu_train_checkpoint_inference(tmp_path) -> None:
    model = build_model(
        architecture="unet",
        met_in_ch=5,
        dem_in_ch=4,
        base_ch=4,
        n_levels=2,
        use_film=True,
    )
    module = DownscalingLitModule(model, max_epochs=1, warmup_epochs=0)
    data = DownscalingDataModule(TensorDataset(), batch_size=2, seed=7)
    trainer = build_trainer(
        {"accelerator": "cpu", "devices": 1, "precision": "32-true", "num_workers": 0},
        max_epochs=1,
        patience=1,
        checkpoint_dir=tmp_path / "checkpoints",
    )
    trainer.fit(module, datamodule=data)
    checkpoint = trainer.checkpoint_callback.best_model_path
    assert checkpoint

    stats = {
        name: [0.0, 1.0]
        for name in ("t2m", "tp", "u10", "v10", "sp")
    }
    stats.update(
        {name: [0.0, 1.0] for name in ("elevation", "slope", "aspect", "curvature")}
    )
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps(stats))
    cfg = {
        "deep_learning": {
            "architecture": "unet",
            "met_vars": ["t2m", "tp", "u10", "v10", "sp"],
            "dem_in_ch": 4,
            "base_ch": 4,
            "n_levels": 2,
            "patch_size": 16,
            "overlap": 4,
            "use_film": True,
        }
    }
    pipeline = DLInferencePipeline(checkpoint, cfg, stats_path, device="cpu")
    rng = np.random.default_rng(8)
    coarse = xr.Dataset(
        {
            name: (("time", "y", "x"), rng.normal(size=(1, 17, 19)))
            for name in stats
            if name in pipeline.met_vars
        },
        coords={"time": [0]},
    )
    dem = xr.Dataset(
        {
            name: (("y", "x"), rng.normal(size=(17, 19)))
            for name in ("elevation", "slope", "aspect", "curvature")
        }
    )
    output = pipeline.run(coarse, dem)
    assert output.sizes == {"time": 1, "y": 17, "x": 19}
    assert all(np.isfinite(output[name]).all() for name in pipeline.met_vars)
    assert output.attrs["model_checkpoint_sha256"]
    assert output.attrs["normalization_stats_sha256"]
