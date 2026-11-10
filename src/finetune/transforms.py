"""Normalization for the real-data channels.

Per-channel z-score statistics come from the product's own ``norm_cache``
(``{channel: {mean, std, log}}``, one entry per channel in ``channels.json``),
so the real-data track owns the transform rather than widening
``wildfire_simulator.transforms`` (which normalizes MTT crops).
"""

import numpy as np
import torch


class ZScorePerChannel:
    """Per-channel z-score normalization, with optional log1p for skewed channels.

    ``mean``/``std``/``log`` are per-channel arrays. Channels flagged ``log``
    are passed through ``log1p(clamp(x, min=0))`` *before* standardizing: a
    monotone transform that keeps zero at zero (so a zero-valued cell stays at
    the channel's affine zero point) while compressing the long right tail.
    ``log`` is a hard requirement to specify — a skewed channel silently
    standardized in linear space trains poorly.

    NaN is propagated (not clamped), so downstream code decides how to fill
    cells a window does not cover.

    Works on ``(C, H, W)`` and ``(B, C, H, W)`` inputs, matching
    ``wildfire_simulator.transforms.MinMaxPerChannel``'s layout convention.
    """

    def __init__(self, mean_vals, std_vals, log_mask=None):
        self.mean_vals = torch.as_tensor(mean_vals, dtype=torch.float32)
        self.std_vals = torch.as_tensor(std_vals, dtype=torch.float32)
        if log_mask is None:
            log_mask = np.zeros(len(self.mean_vals), dtype=bool)
        self.log_mask = torch.as_tensor(np.asarray(log_mask), dtype=torch.bool)
        if self.log_mask.numel() != self.mean_vals.numel():
            raise ValueError(
                f"log_mask length ({self.log_mask.numel()}) must match the "
                f"number of channels ({self.mean_vals.numel()})"
            )

    def __call__(self, img):
        return self.transform(img)

    def _get_params(self, img):
        C = img.shape[-3]
        if img.dim() == 3:
            return (self.mean_vals.to(img.device).view(C, 1, 1),
                    self.std_vals.to(img.device).view(C, 1, 1),
                    self.log_mask.to(img.device).view(C, 1, 1))
        return (self.mean_vals.to(img.device).view(1, C, 1, 1),
                self.std_vals.to(img.device).view(1, C, 1, 1),
                self.log_mask.to(img.device).view(1, C, 1, 1))

    def transform(self, img):
        mean_vals, std_vals, log_mask = self._get_params(img)
        x = img
        if bool(log_mask.any()):
            # clamp(min=0) is safe for NaN (torch.clamp propagates NaN).
            x = torch.where(log_mask, torch.log1p(x.clamp(min=0)), x)
        return (x - mean_vals) / std_vals.clamp(min=1e-8)

    def inverse(self, img):
        mean_vals, std_vals, log_mask = self._get_params(img)
        x = img * std_vals + mean_vals
        if bool(log_mask.any()):
            x = torch.where(log_mask, torch.expm1(x), x)
        return x
