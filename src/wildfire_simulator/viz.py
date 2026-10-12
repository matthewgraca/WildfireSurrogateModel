"""Visualization helpers for fire rollout results.

Shared between the training-time TensorBoard callback
(``callbacks.TensorBoardCallback``) and the post-hoc ``scripts/evaluate.py``
script. Every ``render_*`` function builds a matplotlib figure and returns
it as a uint8 ``(H, W, 3)`` RGB numpy array, ready for
``SummaryWriter.add_image`` or for saving to a file.
"""

import io

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image


def select_viz_indices(n_samples, n=10, seed=42):
    """Deterministic subset of ``n`` sample positions out of ``n_samples``.

    Uses the same torch-generator RNG convention as
    ``MultiSceneDataset.stratified_split`` so the selection is reproducible
    and stable across runs and epochs. Returns all positions (sorted) when
    ``n >= n_samples``.

    Args:
        n_samples: total number of samples to choose from.
        n: subset size.
        seed: RNG seed.

    Returns:
        Sorted list of int positions.
    """
    if n >= n_samples:
        return list(range(n_samples))
    gen = torch.Generator().manual_seed(seed)
    return sorted(torch.randperm(n_samples, generator=gen)[:n].tolist())


def _render_to_array(fig, dpi):
    """Rasterize an open figure to a uint8 RGB (H, W, 3) array, then close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return np.asarray(Image.open(buf).convert("RGB"))


def render_input_channels(sample, sample_idx, dpi=150):
    """Static input channels (landscape features) for a sample, as a 2x5 grid.

    Channel mapping (skip 0=mask, 1=FAT):
    2: Elevation, 3: Slope, 4: Aspect, 5: Fuel model (FBFM40)
    6: Canopy Cover, 7: Stand Height, 8: Canopy Base Height, 9: Canopy Bulk Density
    10: Wind U, 11: Wind V (per-cell fields when the trial has terrain-aware
    WindNinja sidecar grids; uniform broadcasts for v1 scalar-wind trials)
    12: Foliar moisture (scalar)

    Args:
        sample: raw (un-normalized) (13, H, W) tensor or ndarray crop.
        sample_idx: sample label used in the figure title.
        dpi: rasterization resolution of the returned image.

    Returns:
        uint8 (H, W, 3) RGB array.
    """
    channel_info = [
        (2, "Elevation", "terrain", "m"),
        (3, "Slope", "YlOrBr", "°"),
        (4, "Aspect", "hsv", "°"),
        (5, "Fuel Model\n(FBFM40)", "Set3", ""),
        (6, "Canopy Cover", "Greens", "%"),
        (7, "Stand Height", "Greens", "m"),
        (8, "Canopy Base\nHeight", "Greens", "m"),
        (9, "Canopy Bulk\nDensity", "Greens", "kg/m³"),
    ]

    sample_np = sample.numpy() if isinstance(sample, torch.Tensor) else sample

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))

    # Plot spatial channels (2-9)
    for i, (ch_idx, name, cmap, unit) in enumerate(channel_info):
        row, col = divmod(i, 5)
        ax = axes[row, col]
        data = sample_np[ch_idx]

        im = ax.imshow(data, cmap=cmap, aspect='equal')
        ax.set_title(name, fontsize=10)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if unit:
            cbar.set_label(unit, fontsize=8)

    # Wind quiver plot: channels 10/11 are U/V. For v2 trials these are
    # per-cell terrain-aware fields (WindNinja sidecar grids); for v1 they
    # are uniform broadcasts, so a coarse grid of identical arrows suffices.
    ax_wind = axes[1, 3]
    wind_u = sample_np[10]
    wind_v = sample_np[11]
    terrain_aware = float(np.ptp(wind_u)) > 0.0 or float(np.ptp(wind_v)) > 0.0
    wind_speed = np.sqrt(wind_u**2 + wind_v**2)
    mean_speed = float(wind_speed.mean())
    mean_dir = (np.degrees(np.arctan2(-float(wind_u.mean()), -float(wind_v.mean()))) + 360) % 360

    if terrain_aware:
        # Subsample the actual per-cell field so the quiver stays readable.
        step = 25
        X, Y = np.meshgrid(np.arange(0, 500, step), np.arange(0, 500, step))
        U = wind_u[::step, ::step]
        V = wind_v[::step, ::step]
    else:
        grid_size = 10
        x = np.linspace(0, 500, grid_size)
        y = np.linspace(0, 500, grid_size)
        X, Y = np.meshgrid(x, y)
        U = np.full_like(X, float(wind_u[0, 0]))
        V = np.full_like(Y, float(wind_v[0, 0]))

    ax_wind.quiver(X, Y, U, V, scale=max(mean_speed, 1e-6) * 15, color='navy', alpha=0.7)
    ax_wind.set_xlim(0, 500)
    ax_wind.set_ylim(500, 0)
    ax_wind.set_aspect('equal')
    if terrain_aware:
        ax_wind.set_title(f"Wind (terrain-aware)\n{mean_speed:.0f} avg units @ {mean_dir:.0f}°", fontsize=10)
    else:
        ax_wind.set_title(f"Wind\n{mean_speed:.0f} units @ {mean_dir:.0f}°", fontsize=10)
    ax_wind.axis('off')

    # Foliar moisture
    ax_fm = axes[1, 4]
    foliar_moisture = float(sample_np[12, 0, 0])
    ax_fm.text(0.5, 0.5, f"Foliar\nMoisture\n\n{foliar_moisture:.0f}%",
               ha='center', va='center', fontsize=14,
               transform=ax_fm.transAxes)
    ax_fm.axis('off')

    fig.suptitle(f"Sample {sample_idx} — Static Input Channels", fontsize=13)
    fig.tight_layout()

    return _render_to_array(fig, dpi)


def render_mask_rollout(pred_history, gt_history, sample_idx, dpi=150):
    """Side-by-side fire mask rollout (predicted / ground truth / error).

    Args:
        pred_history: sequence of (mask, FAT) 2-channel frames (tensors or
            ndarrays), one per rollout time step.
        gt_history: matching ground-truth frames.
        sample_idx: sample label used in the figure title.
        dpi: rasterization resolution of the returned image.

    Returns:
        uint8 (H, W, 3) RGB array.
    """
    n_steps = len(pred_history)
    indices = np.linspace(0, n_steps - 1, min(8, n_steps), dtype=int)

    fig, axes = plt.subplots(3, len(indices), figsize=(len(indices) * 3, 9), squeeze=False)

    for col, idx in enumerate(indices):
        pred = pred_history[idx]
        gt = gt_history[idx] if idx < len(gt_history) else gt_history[-1]

        pred_mask = pred[0].numpy() if isinstance(pred, torch.Tensor) else pred[0]
        gt_mask = gt[0].numpy() if isinstance(gt, torch.Tensor) else gt[0]

        axes[0, col].imshow(np.where(pred_mask == 0, np.nan, pred_mask), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        axes[0, col].set_title(f"t={idx}", fontsize=8)
        axes[0, col].axis('off')

        axes[1, col].imshow(np.where(gt_mask == 0, np.nan, gt_mask), cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        axes[1, col].axis('off')

        error = np.abs(pred_mask - gt_mask)
        axes[2, col].imshow(error, cmap='Reds', vmin=0, vmax=1, aspect='auto')
        axes[2, col].axis('off')

    axes[0, 0].set_ylabel("Predicted", fontsize=10)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=10)
    axes[2, 0].set_ylabel("Error", fontsize=10)

    fig.suptitle(f"Sample {sample_idx} — Fire Mask Rollout", fontsize=12)
    fig.tight_layout()

    return _render_to_array(fig, dpi)


def render_final_arrival_map(pred_history, gt_history, sample_idx, dpi=150):
    """Final predicted vs ground truth arrival time map (3 panels).

    Args:
        pred_history: sequence of (mask, FAT) 2-channel frames; the last
            entry is the final rollout state.
        gt_history: matching ground-truth frames; the last entry is the
            final ground truth.
        sample_idx: sample label used in the figure title.
        dpi: rasterization resolution of the returned image.

    Returns:
        uint8 (H, W, 3) RGB array.
    """
    pred_final = pred_history[-1]
    gt_final = gt_history[-1]

    pred_arr = pred_final[1].numpy() if isinstance(pred_final, torch.Tensor) else pred_final[1]
    gt_arr = gt_final[1].numpy() if isinstance(gt_final, torch.Tensor) else gt_final[1]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axes[0].imshow(np.where(pred_arr == 0, np.nan, pred_arr), cmap='YlOrRd', aspect='auto')
    axes[0].set_title("Predicted Arrival Time")
    axes[0].axis('off')
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(np.where(gt_arr == 0, np.nan, gt_arr), cmap='YlOrRd', aspect='auto')
    axes[1].set_title("Ground Truth Arrival Time")
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    error = np.abs(pred_arr - gt_arr)
    im2 = axes[2].imshow(error, cmap='Reds', aspect='auto')
    axes[2].set_title(f"Absolute Error (MAE={error.mean():.4f})")
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    fig.suptitle(f"Sample {sample_idx} — Final Arrival Time Map", fontsize=12)
    fig.tight_layout()

    return _render_to_array(fig, dpi)


def render_fat_montage(pred_fats, gt_fats, sample_ids, title="", dpi=50):
    """Snapshot of final arrival-time maps for a set of samples.

    Two rows (predicted, ground truth) with one column per sample. All cells
    share a fixed [0, 1] color scale (normalized arrival time) so predicted
    and ground truth are directly comparable. Unburned pixels (FAT == 0)
    render as blank.

    Args:
        pred_fats: sequence of final predicted FAT maps (H, W), tensors or
            ndarrays, normalized time units.
        gt_fats: matching sequence of ground-truth FAT maps.
        sample_ids: one label per sample (used as cell titles).
        title: optional figure title (e.g. scene name and epoch).
        dpi: rasterization resolution of the returned image.

    Returns:
        uint8 (H, W, 3) RGB array.
    """
    if len(pred_fats) != len(gt_fats):
        raise ValueError(
            f"pred_fats ({len(pred_fats)}) and gt_fats ({len(gt_fats)}) must match"
        )
    if len(pred_fats) == 0:
        raise ValueError("render_fat_montage requires at least one sample")

    def _to_np(x):
        return x.numpy() if isinstance(x, torch.Tensor) else x

    n = len(pred_fats)
    fig, axes = plt.subplots(2, n, figsize=(3 * n, 7), squeeze=False)

    for col in range(n):
        pred = _to_np(pred_fats[col])
        gt = _to_np(gt_fats[col])
        for ax, fat in ((axes[0, col], pred), (axes[1, col], gt)):
            ax.imshow(
                np.where(fat == 0, np.nan, fat),
                cmap='YlOrRd', vmin=0, vmax=1, aspect='equal',
            )
            ax.axis('off')
        axes[0, col].set_title(f"sample {sample_ids[col]:02d}", fontsize=9)

    axes[0, 0].set_ylabel("Predicted", fontsize=10)
    axes[1, 0].set_ylabel("Ground Truth", fontsize=10)

    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()

    return _render_to_array(fig, dpi)
