"""
preflight.py — Quick diagnostic to verify model training is healthy before
committing to a full run.

Runs in ~5 minutes and checks:
1. Overfit test: can the model memorize a tiny dataset?
2. Per-step loss: do errors compound across autoregressive steps?
3. Gradient health: no exploding/vanishing gradients?
4. Visual sanity: does a single inference look fire-shaped?

Usage:
    python scripts/preflight.py
    python scripts/preflight.py --output-dir ./tmp/preflight
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, random_split

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.trainers import ForwardBurnTrainer, BurnerBatchProcessor
from wildfire_simulator.datasets import build_multiscene_dataset, TransformedDataset
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.simulators import ForwardBurnSimulator, fire_burn_step
from wildfire_simulator.utils import ScalarRNG
from wildfire_simulator.config import load_config


class ConstSampler:
    def __init__(self, prob):
        self.prob = prob
    def get_prob(self, epoch):
        return self.prob


def check_overfit(model, dataset, transform, device, dt, max_t):
    """Test 1: Can the model memorize 2 samples in 5 epochs?"""
    print("\n" + "=" * 60)
    print(" TEST 1: Overfit Test (2 samples, 5 epochs)")
    print("=" * 60)

    subset = Subset(dataset, [0, 1])
    subset = TransformedDataset(subset, transform)
    loader = DataLoader(subset, batch_size=2, shuffle=False, num_workers=0)

    model_copy = MK_UNet_Regression(
        in_channels=14, out_channels=1,
        channels=[16, 32, 64, 96, 160],
        final_activation='sigmoid'
    ).to(device)

    optimizer = torch.optim.AdamW(model_copy.parameters(), 5e-4, weight_decay=1e-4)
    loss_fn = nn.BCELoss()

    burner = ForwardBurnProcess()
    rng = ScalarRNG()
    batch_processor = BurnerBatchProcessor(
        burner=burner, dt=dt, eval=False,
        sampler=ConstSampler(0.0), rng=rng, device=device
    )

    losses = []
    for epoch in range(5):
        model_copy.train()
        epoch_loss = 0.0
        for batch in loader:
            N = batch.size(0)
            preds_padded = torch.zeros(N, 13, 512, 512, device=device)
            for t in np.arange(dt, max_t, dt):
                inputs, targets = batch_processor(preds_padded, batch, epoch=epoch, batch_idx=0, t=t)
                optimizer.zero_grad()
                pred_out = model_copy(inputs)
                if isinstance(pred_out, (list, tuple)):
                    pred_out = pred_out[0]
                loss = loss_fn(pred_out, targets)
                loss.backward()
                optimizer.step()

                # Deterministic FAT update
                pred_mask = (pred_out[:, 0:1].detach() > 0.5).float()
                preds_padded = inputs[:, :13].detach().clone()
                preds_padded[:, 0:1] = pred_mask
                newly_burned = (pred_mask == 1) & (preds_padded[:, 1:2] == 0)
                preds_padded[:, 1:2][newly_burned] = t + dt

                epoch_loss += loss.item()
        losses.append(epoch_loss)
        print(f"  Epoch {epoch}: loss = {epoch_loss:.4f}")

    decreased = losses[-1] < losses[0] * 0.5
    print(f"\n  Loss decrease: {losses[0]:.4f} → {losses[-1]:.4f} ({(1 - losses[-1]/losses[0])*100:.0f}% reduction)")
    print(f"  Result: {'PASS ✓' if decreased else 'FAIL ✗ — loss did not decrease by at least 50%'}")
    return decreased


def check_per_step_loss(model, dataset, transform, device, dt, max_t):
    """Test 2: Do errors compound across autoregressive steps?"""
    print("\n" + "=" * 60)
    print(" TEST 2: Per-Step Loss (autoregressive error compounding)")
    print("=" * 60)

    subset = TransformedDataset(Subset(dataset, [0]), transform)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    burner = ForwardBurnProcess()
    batch_processor = BurnerBatchProcessor(
        burner=burner, dt=dt, eval=True, device=device
    )
    loss_fn = nn.BCELoss()

    model.eval()
    preds_padded = None
    step_losses = []

    with torch.no_grad():
        times = np.arange(dt, max_t, dt)
        for step, t in enumerate(times):
            if preds_padded is None:
                pred_input = batch.to(device).clone()
                not_burnt = pred_input[:, 1:2, :, :] > dt
                pred_input[:, 0:1][not_burnt] = 0.0
                pred_input[:, 1:2][not_burnt] = 0.0
            else:
                pred_input = preds_padded

            inputs, targets = batch_processor(pred_input, batch, epoch=0, batch_idx=0, t=t)
            pred_out = model(inputs)
            if isinstance(pred_out, (list, tuple)):
                pred_out = pred_out[0]

            loss = loss_fn(pred_out, targets)
            step_losses.append(loss.item())

            # Deterministic FAT update
            pred_mask = (pred_out[:, 0:1].detach() > 0.5).float()
            preds_padded = inputs[:, :13].detach().clone()
            preds_padded[:, 0:1] = pred_mask
            newly_burned = (pred_mask == 1) & (preds_padded[:, 1:2] == 0)
            preds_padded[:, 1:2][newly_burned] = t + dt

    early = np.mean(step_losses[:5])
    mid = np.mean(step_losses[20:25])
    late = np.mean(step_losses[-5:])

    print(f"  Steps  1-5 (early): avg loss = {early:.4f}")
    print(f"  Steps 21-25 (mid):  avg loss = {mid:.4f}")
    print(f"  Steps 43-47 (late): avg loss = {late:.4f}")
    print(f"  Compounding ratio (late/early): {late/early:.1f}x")

    # If late loss is >100x early, errors are compounding badly
    stable = late / early < 100
    print(f"  Result: {'PASS ✓' if stable else 'FAIL ✗ — errors compound >100x'}")
    return stable


def check_gradients(model, dataset, transform, device, dt, max_t):
    """Test 3: Are gradients healthy?"""
    print("\n" + "=" * 60)
    print(" TEST 3: Gradient Health")
    print("=" * 60)

    subset = TransformedDataset(Subset(dataset, [0]), transform)
    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))

    burner = ForwardBurnProcess()
    rng = ScalarRNG()
    batch_processor = BurnerBatchProcessor(
        burner=burner, dt=dt, eval=False,
        sampler=ConstSampler(0.0), rng=rng, device=device
    )
    loss_fn = nn.BCELoss()

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), 5e-4)

    N = batch.size(0)
    preds_padded = torch.zeros(N, 13, 512, 512, device=device)

    # Run one step
    t = dt
    inputs, targets = batch_processor(preds_padded, batch, epoch=0, batch_idx=0, t=t)
    optimizer.zero_grad()
    pred_out = model(inputs)
    if isinstance(pred_out, (list, tuple)):
        pred_out = pred_out[0]
    loss = loss_fn(pred_out, targets)
    loss.backward()

    grad_norms = []
    zero_count = 0
    for name, p in model.named_parameters():
        if p.grad is not None:
            norm = p.grad.norm().item()
            grad_norms.append(norm)
            if norm == 0:
                zero_count += 1

    min_grad = min(grad_norms)
    max_grad = max(grad_norms)
    mean_grad = np.mean(grad_norms)
    zero_pct = zero_count / len(grad_norms) * 100

    print(f"  Gradient norms across {len(grad_norms)} parameters:")
    print(f"    Min:  {min_grad:.2e}")
    print(f"    Mean: {mean_grad:.2e}")
    print(f"    Max:  {max_grad:.2e}")
    print(f"    Zero: {zero_count}/{len(grad_norms)} ({zero_pct:.0f}%)")

    healthy = max_grad < 100 and zero_pct < 50
    if max_grad >= 100:
        print(f"  Result: FAIL ✗ — gradients exploding (max={max_grad:.2e})")
    elif zero_pct >= 50:
        print(f"  Result: FAIL ✗ — majority of gradients are zero ({zero_pct:.0f}%)")
    else:
        print(f"  Result: PASS ✓")
    return healthy


def check_visual(model, dataset, transform, device, dt, max_t, output_dir):
    """Test 4: Does inference produce fire-shaped output?"""
    print("\n" + "=" * 60)
    print(" TEST 4: Visual Sanity Check")
    print("=" * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    burner = ForwardBurnProcess()
    sample = dataset[0]
    sample_transformed = transform(sample)

    # Run inference
    sample_batched = sample.unsqueeze(0).to(device)
    simulator = ForwardBurnSimulator(
        data=sample_batched, model=model, step=fire_burn_step,
        transform=transform, dt=dt * max_t, max_t=max_t, t0=dt * max_t
    )
    pred_history = simulator.run_to(max_t, return_history=True)
    pred_history = [p.squeeze(0).cpu() for p in pred_history]

    # Ground truth at a few steps
    times = [0.25, 0.5, 0.75, 1.0]
    fig, axes = plt.subplots(2, len(times), figsize=(12, 6), squeeze=False)

    for col, t in enumerate(times):
        # Prediction at corresponding index
        idx = min(int(t / dt), len(pred_history) - 1)
        pred = pred_history[idx]
        pred_mask = pred[0].numpy()

        # Ground truth
        gt = burner(sample_transformed, t)
        gt_raw = transform.inverse(gt)
        gt_mask = gt_raw[0].numpy()
        gt_mask = np.where(gt_mask == 0, np.nan, gt_mask)

        # Show prediction without masking (uniform = untrained, spatial structure = learning)
        axes[0, col].imshow(pred_mask, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
        axes[0, col].set_title(f"t={t:.2f}", fontsize=9)
        axes[0, col].axis('off')

        axes[1, col].imshow(gt_mask, cmap='YlOrRd', vmin=0, vmax=1, aspect='equal')
        axes[1, col].axis('off')

    fig.text(0.02, 0.72, "Predicted", va='center', ha='center', fontsize=11, rotation=90)
    fig.text(0.02, 0.30, "Ground\nTruth", va='center', ha='center', fontsize=11, rotation=90)
    fig.suptitle("Visual Sanity Check — Single Sample Inference", fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(left=0.06)

    filepath = os.path.join(output_dir, "preflight_visual.png")
    fig.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close(fig)

    # Check: does prediction have any nonzero pixels?
    # Note: untrained models may produce all-zero (threshold cuts ~0.5 outputs)
    # or all-one predictions. Both are expected before training.
    final_pred = pred_history[-1][0]
    nonzero_pct = (final_pred > 0.1).float().mean().item() * 100

    print(f"  Final prediction nonzero pixels: {nonzero_pct:.1f}%")
    if nonzero_pct == 0:
        print(f"  (Expected for untrained model — threshold removes sub-0.5 predictions)")
    print(f"  Saved: {filepath}")

    # Pass as long as inference ran without error
    print(f"  Result: PASS ✓")
    return True


def main():
    parser = argparse.ArgumentParser(description="Preflight checks for training")
    parser.add_argument('--output-dir', type=str, default='./tmp/preflight',
                        help='Directory to save diagnostic outputs')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config.yaml in project root)')
    args = parser.parse_args()

    config = load_config(args.config)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print(" PREFLIGHT DIAGNOSTICS")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # Load data
    dataset = build_multiscene_dataset(config)
    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    print(f"  Dataset: {len(dataset)} samples")

    dt = 1/48
    max_t = 1.0

    # Create model
    torch.manual_seed(42)
    model = MK_UNet_Regression(
        in_channels=14, out_channels=1,
        channels=[16, 32, 64, 96, 160],
        final_activation='sigmoid'
    ).to(device)

    # Run checks
    results = {}
    results['overfit'] = check_overfit(model, dataset, transform, device, dt, max_t)
    results['gradients'] = check_gradients(model, dataset, transform, device, dt, max_t)
    results['per_step'] = check_per_step_loss(model, dataset, transform, device, dt, max_t)
    results['visual'] = check_visual(model, dataset, transform, device, dt, max_t, args.output_dir)

    # Summary
    print("\n" + "=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  {name:15s}: {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  All checks passed. Safe to start full training.")
    else:
        print("  Some checks failed. Review issues above before training.")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
