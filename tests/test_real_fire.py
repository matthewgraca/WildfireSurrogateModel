"""Tests for the real-fire inference primitives (real_fire)."""

import numpy as np
import pytest
import torch

from wildfire_simulator.real_fire import build_sample, fire_burn_step_wind


class _Recorder(torch.nn.Module):
    """Mock model that records the injected wind slice and returns a full mask."""

    def __init__(self, value=0.8):
        super().__init__()
        self.value = value
        self.seen_wind = None
        self.seen_t = None

    def forward(self, x):
        # x: (1, 14, 512, 512) after the 6-px constant pad. Channel 13 is the
        # time channel; sample at the crop center (256,256) to avoid the pad.
        self.seen_wind = x[0, 10:12, 256:258, 256:258].detach().clone()
        self.seen_t = x[0, 13, 256, 256].detach().clone()
        return [torch.full((1, 1, 512, 512), self.value, device=x.device)]


def _landscape(h=700, w=700):
    """8-band synthetic landscape with distinct per-band constant values."""
    land = np.zeros((8, h, w), dtype=np.float32)
    for band in range(8):
        land[band] = band * 10.0 + 1.0
    return land


def test_build_sample_shape_and_channel_order():
    land = _landscape()
    wu = np.full(land.shape[-2:], 5.0, dtype=np.float32)
    wv = np.full(land.shape[-2:], -3.0, dtype=np.float32)
    sample = build_sample(land, wu, wv, 350, 350, foliar_moisture=110.0,
                          dt=0.125, max_val_fat=1439.8)

    assert sample.shape == (13, 500, 500)
    assert sample.dtype == np.float32
    # Channel order must match WildfireDataset._load_raw_channels:
    # [mask, fat, elev, slope, aspect, fuel, cc, sh, cbh, cbd, wu, wv, fm]
    means = sample.mean(axis=(1, 2))
    # Only the seed cell is lit -> negligible but non-zero mean.
    assert np.allclose(means[:2], [0.0, 0.0], atol=1e-3)
    assert np.allclose(means[2:10],
                       [1.0, 11.0, 21.0, 31.0, 41.0, 51.0, 61.0, 71.0],
                       atol=1e-3)
    assert np.allclose(means[10:12], [5.0, -3.0], atol=1e-3)
    assert np.allclose(means[12], 110.0, atol=1e-3)
    # Seed: only the ignition cell (crop center) is lit, FAT = dt*max_val_fat.
    assert sample[0, 250, 250] == 1.0
    assert sample[1, 250, 250] == np.float32(0.125 * 1439.8)


def test_fire_burn_step_wind_injects_time_varying_wind():
    model = _Recorder(value=0.8)
    inputs = torch.zeros((1, 13, 500, 500))
    dt = 0.125

    win_u_a = torch.full((500, 500), 0.25)
    win_v_a = torch.full((500, 500), -0.1)
    out1 = fire_burn_step_wind(dt, model, inputs, dt, win_u_a, win_v_a)
    wind_a = model.seen_wind.clone()

    win_u_b = torch.full((500, 500), -0.75)
    win_v_b = torch.full((500, 500), 0.5)
    out2 = fire_burn_step_wind(dt, model, out1, dt, win_u_b, win_v_b)
    wind_b = model.seen_wind.clone()

    # The network must see per-step wind that differs, not a stale static field.
    assert not torch.allclose(wind_a, wind_b)
    assert wind_b[0].mean().item() == pytest.approx(-0.75, abs=1e-4)
    # Threshold: 0.8 > 0.5 -> whole mask burns; FAT assigned t+dt on new cells.
    assert out1.shape == (1, 13, 500, 500)
    assert (out1[0, 0] == 1.0).all()
    assert (out1[0, 1] == (dt + dt)).all()


def test_fire_burn_step_wind_preserves_input_unchanged():
    """The input tensor is not mutated (mirrors fire_burn_step deepcopy)."""
    model = _Recorder(value=0.2)   # below threshold -> no new burn
    inputs = torch.zeros((1, 13, 500, 500))
    before = inputs.clone()
    fire_burn_step_wind(0.1, model, inputs, 0.1,
                        torch.full((500, 500), 0.5),
                        torch.full((500, 500), -0.5))
    assert torch.equal(inputs, before)
