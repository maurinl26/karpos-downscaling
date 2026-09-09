"""CPU contract tests for the deep-learning inference path."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from downscaling.deep_learning.inference import tiled_inference  # noqa: E402


class OnesModel(torch.nn.Module):
    def forward(self, x_met, x_dem):  # noqa: D401
        return torch.ones((x_met.shape[0], 1, x_met.shape[2], x_met.shape[3]))


@pytest.mark.parametrize("shape", [(10, 12), (70, 83)])
def test_tiled_inference_covers_small_and_non_divisible_domains(shape) -> None:
    h, w = shape
    model = OnesModel()
    x_met = torch.zeros((1, 2, h, w))
    x_dem = torch.zeros((1, 4, h, w))

    output = tiled_inference(model, x_met, x_dem, tile_size=32, overlap=8)

    assert output.shape == (1, 1, h, w)
    assert torch.allclose(output, torch.ones_like(output))


def test_tiled_inference_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        tiled_inference(
            OnesModel(),
            torch.zeros((1, 2, 8, 8)),
            torch.zeros((1, 4, 8, 8)),
            tile_size=8,
            overlap=8,
        )
