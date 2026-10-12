"""
evaluate.py — Post-training evaluation script.

Loads a trained checkpoint, runs autoregressive inference on test samples,
produces visual comparisons (predicted vs ground truth fire arrival maps),
plots training curves from TensorBoard logs, and reports quantitative metrics
as formatted tables.

Usage:
    python evaluate.py --checkpoint ./checkpoints/best-model-99-0.01.pt
    python evaluate.py --checkpoint ./checkpoints/best-model-99-0.01.pt --num_samples 20
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image

from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.simulators import ForwardBurnSimulator, fire_burn_step
from wildfire_simulator.datasets import build_multiscene_dataset
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.config import load_config
from wildfire_simulator import viz


def load_model(checkpoint_path, device, in_channels=14):
    """Load model from checkpoint."""
    model = MK_UNet_Regression(
        in_channels=in_channels,
        out_channels=1,
        channels=[16, 32, 64, 96, 160],
        final_activation='sigmoid'
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    epoch = checkpoint.get('epoch', -1)
    return model, epoch


def get_test_dataset(config):
    """Load dataset and return the test split (same split as training).

    Mirrors the scene-building and validation-split logic in run.py so the
    evaluated test set matches what training held out.
    """
    dataset = build_multiscene_dataset(config)

    # Same stratified per-scene split as training (deterministic seed), so the
    # evaluated validation samples match what training held out, per scene.
    _, _, per_scene_val = dataset.stratified_split(val_frac=0.2, seed=42)

    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    return dataset, transform, per_scene_val


def run_inference(model, sample, transform, device, dt, max_t):
    """
    Run autoregressive rollout on a single sample.
    Returns predicted history and ground truth history at each time step.
    """
    burner = ForwardBurnProcess()

    # Transform the sample to normalized space (where t ∈ [0,1] is meaningful)
    sample_transformed = transform(sample)

    # Ground truth snapshots at each time step (burn in normalized space,
    # then inverse-transform to raw space to match pred_history)
    times = np.arange(dt, max_t, dt)
    gt_history = []
    for t in times:
        burned = burner(sample_transformed, t)
        gt_history.append(transform.inverse(burned))
    gt_history.append(sample)  # final state (raw)

    # Move sample to device and add batch dimension for inference
    # (simulator applies transform internally, expects raw data)
    sample_batched = sample.unsqueeze(0).to(device)

    # Predicted rollout using the simulator
    simulator = ForwardBurnSimulator(
        data=sample_batched,
        model=model,
        step=fire_burn_step,
        transform=transform,
        dt=dt * max_t,
        max_t=max_t,
        t0=dt * max_t
    )
    pred_history = simulator.run_to(max_t, return_history=True)

    # Remove batch dimension and move to CPU for metrics/visualization
    pred_history = [p.squeeze(0).cpu() for p in pred_history]

    return pred_history, gt_history, times


def compute_metrics(pred_frames, gt_frames):
    """
    Compute metrics between predicted and ground truth frame sequences.
    Compares channel 0 (fire mask) and channel 1 (arrival time).
    """
    metrics = {
        'mse_arrival': [],
        'mae_arrival': [],
        'iou': [],
        'dice': [],
        'precision': [],
        'recall': [],
    }

    for pred, gt in zip(pred_frames, gt_frames):
        pred_mask = pred[0].numpy() if isinstance(pred, torch.Tensor) else pred[0]
        gt_mask = gt[0].numpy() if isinstance(gt, torch.Tensor) else gt[0]
        pred_arr = pred[1].numpy() if isinstance(pred, torch.Tensor) else pred[1]
        gt_arr = gt[1].numpy() if isinstance(gt, torch.Tensor) else gt[1]

        # MSE / MAE on continuous values
        metrics['mse_arrival'].append(float(np.mean((pred_arr - gt_arr) ** 2)))
        metrics['mae_arrival'].append(float(np.mean(np.abs(pred_arr - gt_arr))))

        # Binarize masks (threshold at 0.5) for classification metrics
        pred_binary = (pred_mask > 0.5).astype(np.float32)
        gt_binary = (gt_mask > 0.5).astype(np.float32)

        intersection = float((pred_binary * gt_binary).sum())
        union = float(((pred_binary + gt_binary) > 0).sum())
        pred_sum = float(pred_binary.sum())
        gt_sum = float(gt_binary.sum())

        iou = intersection / (union + 1e-8)
        dice = 2 * intersection / (pred_sum + gt_sum + 1e-8)
        precision = intersection / (pred_sum + 1e-8)
        recall = intersection / (gt_sum + 1e-8)

        metrics['iou'].append(iou)
        metrics['dice'].append(dice)
        metrics['precision'].append(precision)
        metrics['recall'].append(recall)

    # Average over time steps
    return {k: float(np.mean(v)) for k, v in metrics.items()}


def _save_figure(arr, filepath):
    """Save a uint8 RGB array as a PNG, creating parent directories as needed."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(filepath)
    return filepath


def save_input_channels(sample, sample_idx, output_dir):
    """Save visualization of static input channels (landscape features) for a sample."""
    arr = viz.render_input_channels(sample, sample_idx)
    filepath = os.path.join(output_dir, f"input_channels_sample_{sample_idx:03d}.png")
    return _save_figure(arr, filepath)


def save_arrival_comparison(pred_history, gt_history, sample_idx, output_dir):
    """Save side-by-side arrival time comparison at multiple time steps."""
    arr = viz.render_mask_rollout(pred_history, gt_history, sample_idx)
    filepath = os.path.join(output_dir, f"comparison_sample_{sample_idx:03d}.png")
    return _save_figure(arr, filepath)


def save_final_arrival_map(pred_history, gt_history, sample_idx, output_dir):
    """Save final predicted vs ground truth arrival time map."""
    arr = viz.render_final_arrival_map(pred_history, gt_history, sample_idx)
    filepath = os.path.join(output_dir, f"arrival_map_sample_{sample_idx:03d}.png")
    return _save_figure(arr, filepath)


def plot_training_curves(logdir, output_dir):
    """Load TensorBoard logs and plot training/validation loss curves.
    Automatically detects and plots per-channel losses if available."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        print("  [!] tensorboard not installed, skipping training curve plot")
        return None, None, None

    train_dir = os.path.join(logdir, "train")
    val_dir = os.path.join(logdir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"  [!] Training logs not found at {logdir}")
        return None, None, None

    train_acc = EventAccumulator(train_dir)
    train_acc.Reload()
    val_acc = EventAccumulator(val_dir)
    val_acc.Reload()

    train_scalars = train_acc.Scalars("Loss")
    val_scalars = val_acc.Scalars("Loss")

    train_steps = [s.step for s in train_scalars]
    train_values = [s.value for s in train_scalars]
    val_steps = [s.step for s in val_scalars]
    val_values = [s.value for s in val_scalars]

    # Detect per-component losses
    available_tags = train_acc.Tags().get('scalars', [])
    val_tags = val_acc.Tags().get('scalars', [])

    # FireSenseNetLoss components
    fire_components = ['Loss/bce', 'Loss/dice', 'Loss/focal']
    has_fire_components = any(tag in available_tags for tag in fire_components)

    # Legacy HybridLoss components
    legacy_components = ['Loss/mask_bce', 'Loss/arrival_mse']
    has_legacy_components = any(tag in available_tags for tag in legacy_components)

    # DistanceMapLoss components
    dml_components = ['Loss/ce', 'Loss/penalty']
    has_dml_components = any(tag in available_tags for tag in dml_components)

    if has_fire_components:
        # Plot combined + 3 FireSenseNet components
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        # Combined loss
        axes[0].plot(train_steps, train_values, label='Train', linewidth=1.5)
        axes[0].plot(val_steps, val_values, label='Val', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Combined Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        component_info = [
            ('Loss/bce', 'Weighted BCE', 'Loss'),
            ('Loss/dice', 'Dice Loss', 'Loss'),
            ('Loss/focal', 'Focal Loss', 'Loss'),
        ]
        for idx, (tag, title, ylabel) in enumerate(component_info):
            ax = axes[idx + 1]
            if tag in available_tags:
                train_data = train_acc.Scalars(tag)
                ax.plot([s.step for s in train_data], [s.value for s in train_data], label='Train', linewidth=1.5)
            if tag in val_tags:
                val_data = val_acc.Scalars(tag)
                ax.plot([s.step for s in val_data], [s.value for s in val_data], label='Val', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle('Training & Validation Loss', fontsize=13)
        fig.tight_layout()

    elif has_legacy_components:
        # Plot combined + legacy per-channel on separate subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Combined loss
        axes[0].plot(train_steps, train_values, label='Train', linewidth=1.5)
        axes[0].plot(val_steps, val_values, label='Val', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Combined Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Mask BCE
        if 'Loss/mask_bce' in available_tags:
            train_mask = train_acc.Scalars("Loss/mask_bce")
            val_mask = val_acc.Scalars("Loss/mask_bce") if 'Loss/mask_bce' in val_tags else []
            axes[1].plot([s.step for s in train_mask], [s.value for s in train_mask], label='Train', linewidth=1.5)
            if val_mask:
                axes[1].plot([s.step for s in val_mask], [s.value for s in val_mask], label='Val', linewidth=1.5)
            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('BCE')
            axes[1].set_title('Mask Loss (BCE)')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        # Arrival MSE
        if 'Loss/arrival_mse' in available_tags:
            train_arr = train_acc.Scalars("Loss/arrival_mse")
            val_arr = val_acc.Scalars("Loss/arrival_mse") if 'Loss/arrival_mse' in val_tags else []
            axes[2].plot([s.step for s in train_arr], [s.value for s in train_arr], label='Train', linewidth=1.5)
            if val_arr:
                axes[2].plot([s.step for s in val_arr], [s.value for s in val_arr], label='Val', linewidth=1.5)
            axes[2].set_xlabel('Epoch')
            axes[2].set_ylabel('MSE')
            axes[2].set_title('Arrival Time Loss (MSE)')
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)

        fig.suptitle('Training & Validation Loss', fontsize=13)
        fig.tight_layout()

    elif has_dml_components:
        # Plot combined + 2 DistanceMapLoss components
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Combined loss
        axes[0].plot(train_steps, train_values, label='Train', linewidth=1.5)
        axes[0].plot(val_steps, val_values, label='Val', linewidth=1.5)
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Combined Loss (Distance Map)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        component_info = [
            ('Loss/ce', 'Cross-Entropy (unweighted)', 'Loss'),
            ('Loss/penalty', 'Boundary Penalty Term', 'Loss'),
        ]
        for idx, (tag, title, ylabel) in enumerate(component_info):
            ax = axes[idx + 1]
            if tag in available_tags:
                train_data = train_acc.Scalars(tag)
                ax.plot([s.step for s in train_data], [s.value for s in train_data], label='Train', linewidth=1.5)
            if tag in val_tags:
                val_data = val_acc.Scalars(tag)
                ax.plot([s.step for s in val_data], [s.value for s in val_data], label='Val', linewidth=1.5)
            ax.set_xlabel('Epoch')
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        fig.suptitle('Training & Validation Loss', fontsize=13)
        fig.tight_layout()
    else:
        # Single combined plot (legacy / BCELoss only)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(train_steps, train_values, label='Train Loss', linewidth=1.5)
        ax.plot(val_steps, val_values, label='Val Loss', linewidth=1.5)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training & Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        if len(train_values) > 0 and max(train_values) / (min(train_values) + 1e-10) > 100:
            ax.set_yscale('log')

    filepath = os.path.join(output_dir, "training_curves.png")
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return filepath, (train_steps, train_values), (val_steps, val_values)


def format_table(headers, rows, col_width=14):
    """Format a list of rows into an aligned text table."""
    sep = "+" + "+".join(["-" * (col_width + 2)] * len(headers)) + "+"
    header_row = "|" + "|".join(f" {h:^{col_width}} " for h in headers) + "|"
    lines = [sep, header_row, sep]
    for row in rows:
        formatted = []
        for val in row:
            if isinstance(val, float):
                formatted.append(f" {val:>{col_width}.6f} ")
            else:
                formatted.append(f" {str(val):>{col_width}} ")
        lines.append("|" + "|".join(formatted) + "|")
    lines.append(sep)
    return "\n".join(lines)


def build_per_sample_table(all_metrics):
    """Build a table with one row per sample."""
    headers = ["Sample", "IoU", "Dice", "Precision", "Recall", "MAE Arrival"]
    rows = []
    for i, m in enumerate(all_metrics):
        rows.append([
            i, m['iou'], m['dice'], m['precision'], m['recall'],
            m['mae_arrival']
        ])
    return format_table(headers, rows)


def _summary_table_rows(all_metrics):
    """Return (headers, rows) for the overall summary table."""
    headers = ["Metric", "Mean", "Std", "Min", "Max"]
    keys = ['mse_arrival', 'mae_arrival',
            'iou', 'dice', 'precision', 'recall']
    rows = []
    for key in keys:
        values = [m[key] for m in all_metrics]
        rows.append([key, np.mean(values), np.std(values), np.min(values), np.max(values)])
    return headers, rows


def build_summary_table(all_metrics):
    """Build a summary table with mean and std across all samples."""
    return format_table(*_summary_table_rows(all_metrics))


def _per_scene_summary_table_rows(metrics_by_scene):
    """Return (headers, rows) comparing key metrics across scenes (one row per scene).

    ``metrics_by_scene`` maps scene name -> list of per-sample metric dicts.
    This is the table that isolates new-regime (terrain-aware) performance.
    """
    headers = ["Scene", "N", "IoU", "Dice", "Precision", "Recall", "MAE Arrival"]
    keys = ['iou', 'dice', 'precision', 'recall', 'mae_arrival']
    rows = []
    for scene_name, metrics in metrics_by_scene.items():
        if not metrics:
            rows.append([scene_name, 0, *(float('nan') for _ in keys)])
            continue
        row = [scene_name, len(metrics)]
        for key in keys:
            row.append(float(np.mean([m[key] for m in metrics])))
        rows.append(row)
    return headers, rows


def build_per_scene_summary_table(metrics_by_scene):
    """Build a text table comparing key metrics across scenes (one row per scene)."""
    return format_table(*_per_scene_summary_table_rows(metrics_by_scene))


def build_training_table(train_data, val_data):
    """Build a table summarizing training curve stats."""
    train_steps, train_values = train_data
    val_steps, val_values = val_data

    headers = ["Statistic", "Train Loss", "Val Loss"]
    rows = [
        ["Total Epochs", len(train_steps), len(val_steps)],
        ["Final", train_values[-1], val_values[-1]],
        ["Best", min(train_values), min(val_values)],
        ["Best Epoch", train_steps[int(np.argmin(train_values))],
                       val_steps[int(np.argmin(val_values))]],
        ["Δ (final)", train_values[-1] - val_values[-1], "—"],
    ]
    return format_table(headers, rows)


def save_metrics_table_image(all_metrics, metrics_by_scene, filepath,
                             checkpoint=None, ckpt_epoch=None, dt=None, max_t=None):
    """Render the per-scene and overall summary tables as a single PNG image.

    Replaces the previous metrics.csv / summary.txt outputs.
    """
    scene_headers, scene_rows = _per_scene_summary_table_rows(metrics_by_scene)
    sum_headers, sum_rows = _summary_table_rows(all_metrics)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    ratios = [max(len(scene_rows), 1), len(sum_rows)]
    fig, axes = plt.subplots(2, 1, figsize=(10, 0.55 * (sum(ratios) + 2) + 1.8),
                             gridspec_kw={"height_ratios": ratios})

    meta = []
    if checkpoint:
        meta.append(f"checkpoint: {os.path.basename(checkpoint)}")
    if ckpt_epoch is not None:
        meta.append(f"epoch: {ckpt_epoch}")
    if dt is not None and max_t is not None:
        meta.append(f"dt: {dt}, max_t: {max_t}")
    title = f"Evaluation Results ({len(all_metrics)} samples)"
    if meta:
        title += "  —  " + "  |  ".join(meta)
    fig.suptitle(title, fontsize=13)

    for ax, panel_title, headers, rows in (
        (axes[0], "PER-SCENE SUMMARY", scene_headers, scene_rows),
        (axes[1], "OVERALL SUMMARY", sum_headers, sum_rows),
    ):
        ax.axis("off")
        ax.set_title(panel_title, fontsize=12, loc="left", pad=10)
        table = ax.table(
            cellText=[[fmt(v) for v in row] for row in rows],
            colLabels=headers,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        for (r, _), cell in table.get_celld().items():
            if r == 0:
                cell.set_facecolor("#333333")
                cell.set_text_props(color="white", fontweight="bold")
            elif r % 2 == 0:
                cell.set_facecolor("#f0f0f0")

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return filepath


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained wildfire model")
    parser.add_argument('experiment_dir', type=str, nargs='?', default=None,
                        help='Experiment directory (auto-detects checkpoint, logs, and outputs here)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (default: best checkpoint in experiment_dir/checkpoints/)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory to save evaluation outputs (default: experiment_dir)')
    parser.add_argument('--logdir', type=str, default=None,
                        help='TensorBoard log directory (default: experiment_dir/training/)')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of test samples to evaluate')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config.yaml in project root)')
    parser.add_argument('--dt', type=float, default=None,
                        help='Time step for rollout (default: from config)')
    parser.add_argument('--max_t', type=float, default=None,
                        help='Maximum time for rollout (default: from config)')
    args = parser.parse_args()

    # Resolve paths from experiment_dir if provided
    exp_dir = args.experiment_dir

    if args.checkpoint is None:
        if exp_dir is None:
            parser.error("Either experiment_dir or --checkpoint is required")
        # Find best checkpoint (lowest val_loss in filename)
        ckpt_dir = os.path.join(exp_dir, 'checkpoints')
        if not os.path.isdir(ckpt_dir):
            parser.error(f"No checkpoints directory found at {ckpt_dir}")
        ckpt_files = sorted(Path(ckpt_dir).glob("best-model-*.pt"))
        if not ckpt_files:
            parser.error(f"No checkpoint files found in {ckpt_dir}")
        # Pick the one with the lowest val_loss (last in sorted order by epoch, or parse the loss)
        def parse_val_loss(p):
            try:
                return float(p.stem.split('-')[-1])
            except ValueError:
                return float('inf')
        checkpoint = str(min(ckpt_files, key=parse_val_loss))
    else:
        checkpoint = args.checkpoint

    output_dir = args.output_dir if args.output_dir is not None else (exp_dir if exp_dir else './evaluation_results')
    logdir = args.logdir if args.logdir is not None else (os.path.join(exp_dir, 'training') if exp_dir else './training')

    config = load_config(args.config)
    # CLI args override config if provided
    dt = args.dt if args.dt is not None else config['dt']
    max_t = args.max_t if args.max_t is not None else config['max_t']

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # ─── Load Model ───────────────────────────────────────────────────────
    print(f"Checkpoint: {checkpoint}")
    model, ckpt_epoch = load_model(checkpoint, device, in_channels=config['in_channels'])
    print(f"Loaded epoch: {ckpt_epoch}")

    # ─── Load Data ────────────────────────────────────────────────────────
    print("\nLoading dataset...")
    full_dataset, transform, per_scene_val = get_test_dataset(config)
    print(f"  Dataset size:  {len(full_dataset)}")
    for scene_name, idxs in per_scene_val.items():
        print(f"  Val[{scene_name}]: {len(idxs)}")

    # ─── Training Curves ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" TRAINING CURVES")
    print("=" * 60)

    curves_path, train_data, val_data = plot_training_curves(logdir, output_dir)
    if curves_path and train_data and val_data:
        print(build_training_table(train_data, val_data))
        print(f"\n  Saved: {curves_path}")
    else:
        print("  No training logs found.")

    # ─── Inference & Metrics (per scene) ──────────────────────────────────
    print("\n" + "=" * 60)
    print(f" INFERENCE (up to {args.num_samples} val samples per scene)")
    print("=" * 60)

    all_metrics = []
    metrics_by_scene = {}

    for scene_name, val_idxs in per_scene_val.items():
        n_scene = min(args.num_samples, len(val_idxs))
        metrics_by_scene[scene_name] = []
        if n_scene == 0:
            print(f"\n  Scene '{scene_name}': no validation samples, skipping.")
            continue
        print(f"\n  Scene '{scene_name}' ({n_scene} samples):")

        # Per-scene visualization subfolders keep outputs organized.
        mask_dir = os.path.join(output_dir, scene_name, "mask")
        fat_dir = os.path.join(output_dir, scene_name, "fat")
        input_dir = os.path.join(output_dir, scene_name, "input_channels")

        for j in range(n_scene):
            global_idx = val_idxs[j]
            sample = full_dataset[global_idx]  # raw; run_inference transforms internally
            print(f"    [{j + 1:3d}/{n_scene}] Running rollout...", end=" ", flush=True)

            pred_history, gt_history, times = run_inference(
                model, sample, transform, device, dt, max_t
            )

            min_len = min(len(pred_history), len(gt_history))
            metrics = compute_metrics(pred_history[:min_len], gt_history[:min_len])

            all_metrics.append(metrics)
            metrics_by_scene[scene_name].append(metrics)

            print(f"IoU={metrics['iou']:.4f}  Dice={metrics['dice']:.4f}  "
                  f"MAE_arr={metrics['mae_arrival']:.4f}")

            save_arrival_comparison(pred_history, gt_history, j, mask_dir)
            save_final_arrival_map(pred_history, gt_history, j, fat_dir)
            save_input_channels(sample, j, input_dir)

    # ─── Per-Scene Summary (the new-regime signal) ────────────────────────
    print("\n" + "=" * 60)
    print(" PER-SCENE SUMMARY")
    print("=" * 60)
    print(build_per_scene_summary_table(metrics_by_scene))

    # ─── Overall Summary Table ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(" SUMMARY (across all evaluated samples)")
    print("=" * 60)
    print(build_summary_table(all_metrics))

    # ─── Save to Disk ─────────────────────────────────────────────────────
    table_path = save_metrics_table_image(
        all_metrics, metrics_by_scene,
        os.path.join(output_dir, "metrics_table.png"),
        checkpoint=checkpoint, ckpt_epoch=ckpt_epoch, dt=dt, max_t=max_t,
    )

    print(f"\n  Results saved:")
    print(f"    Metrics table:  {table_path}")
    print(f"    Visuals:        {output_dir}/<scene>/")


if __name__ == "__main__":
    main()
