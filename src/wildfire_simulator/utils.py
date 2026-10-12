import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

def save_frame(data, filepath):
    n_channels = data.shape[0]

    # Determine a roughly square grid that can hold all channels
    ncols = int(np.ceil(np.sqrt(n_channels)))
    nrows = int(np.ceil(n_channels / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2, nrows * 2))

    # Flatten axes for easy indexing into a 1‑D array
    axes = np.atleast_1d(axes).flatten()

    # Plot each channel
    for i in range(n_channels):
        ax = axes[i]
        ax.imshow(data[i], cmap='gray', aspect='auto')
        ax.axis('off')

    # Hide any unused subplots
    for j in range(n_channels, nrows * ncols):
        axes[j].axis('off')

    # Ensure the output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


def save_comparison(data_pred, data_true, filepath):
    n_samples = data_pred.shape[0]

    # Two rows (predictions and true) and n_samples columns
    fig, axes = plt.subplots(
        2, n_samples, figsize=(n_samples * 2, 4), squeeze=False
    )

    for i in range(n_samples):
        ax_pred = axes[0, i]
        ax_pred.imshow(data_pred[i], cmap='gray', aspect='auto')
        ax_pred.axis('off')

        ax_true = axes[1, i]
        ax_true.imshow(data_true[i], cmap='gray', aspect='auto')
        ax_true.axis('off')

    # Ensure the output directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(filepath, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)


class ScalarRNG:
    def __init__(self):
        self.generator = torch.Generator()

    def seed(self, seed_value):
        self.generator.manual_seed(seed_value)

    def rand(self):
        return torch.rand((), generator=self.generator)

