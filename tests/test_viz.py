import numpy as np
import torch
import pytest

from wildfire_simulator.viz import (
    select_viz_indices,
    render_input_channels,
    render_mask_rollout,
    render_final_arrival_map,
    render_fat_montage,
)


def test_select_viz_indices_deterministic_and_sorted():
    a = select_viz_indices(100, n=10, seed=42)
    b = select_viz_indices(100, n=10, seed=42)
    assert a == b
    assert len(a) == 10
    assert all(0 <= i < 100 for i in a)
    assert len(set(a)) == 10
    assert a == sorted(a)


def test_select_viz_indices_seed_changes_selection():
    assert select_viz_indices(100, n=10, seed=42) != select_viz_indices(
        100, n=10, seed=7
    )


def test_select_viz_indices_all_when_n_covers_population():
    assert select_viz_indices(5, n=10) == [0, 1, 2, 3, 4]
    assert select_viz_indices(5, n=5) == [0, 1, 2, 3, 4]

def _sample_frame(seed=0):
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(500, 500, generator=g) > 0.9).float()
    fat = torch.rand(500, 500, generator=g)
    return torch.stack([mask, fat], dim=0)


def _history(n_steps=8, seed=0):
    return [_sample_frame(seed + i) for i in range(n_steps)]


def test_renderers_return_rgb_arrays():
    sample = torch.randn(13, 500, 500)
    arr = render_input_channels(sample, 0)
    assert arr.shape[2] == 3
    assert arr.dtype == np.uint8
    assert arr.shape[0] > 0 and arr.shape[1] > 0

    pred, gt = _history(), _history(seed=100)
    for fn in (render_mask_rollout, render_final_arrival_map):
        arr = fn(pred, gt, 0)
        assert arr.shape[2] == 3
        assert arr.dtype == np.uint8


def test_render_fat_montage():
    pred_fats = [torch.rand(500, 500) for _ in range(3)]
    gt_fats = [torch.rand(500, 500) for _ in range(3)]
    arr = render_fat_montage(pred_fats, gt_fats, [1, 2, 3], title="smoke — epoch 0")
    assert arr.shape[2] == 3
    assert arr.dtype == np.uint8


def test_render_fat_montage_mismatched_lengths():
    with pytest.raises(ValueError):
        render_fat_montage(
            [torch.rand(8, 8)], [torch.rand(8, 8), torch.rand(8, 8)], [0, 1]
        )
    with pytest.raises(ValueError):
        render_fat_montage([], [], [])
