"""Tests for the real-data normalization transform (finetune.transforms)."""

import numpy as np
import pytest
import torch

from finetune.transforms import ZScorePerChannel


def test_zscore_per_channel_standardizes_and_round_trips():
    """Z-score per channel, with log1p for the channels flagged log."""
    transform = ZScorePerChannel(mean_vals=[10.0, 0.0], std_vals=[2.0, 1.0],
                                 log_mask=[False, True])
    x = torch.tensor([[[10.0, 12.0]], [[0.0, 3.0]]])      # (C, 1, 2)
    out = transform(x)

    assert out[0, 0, 0].item() == 0.0                     # (10 - 10) / 2
    assert out[0, 0, 1].item() == 1.0
    assert out[1, 0, 0].item() == 0.0                     # log1p(0) = 0
    assert out[1, 0, 1].item() == pytest.approx(np.log1p(3.0), rel=1e-6)  # log space
    assert torch.allclose(transform.inverse(out), x, atol=1e-6)

    # batch layout, and the same values as the single-sample call
    batched = transform(x.unsqueeze(0))
    assert batched.shape == (1, 2, 1, 2)
    assert torch.allclose(batched, out.unsqueeze(0), atol=1e-6)


def test_zscore_per_channel_keeps_nan_and_rejects_a_bad_mask():
    transform = ZScorePerChannel(mean_vals=[0.0], std_vals=[1.0])
    x = torch.tensor([[[np.nan, 0.0]]])
    assert torch.isnan(transform(x)[0, 0, 0])
    assert transform(x)[0, 0, 1].item() == 0.0

    with pytest.raises(ValueError, match="log_mask length"):
        ZScorePerChannel([0.0, 0.0], [1.0, 1.0], log_mask=[True])
