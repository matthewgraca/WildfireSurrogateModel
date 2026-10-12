import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt


class HybridLoss(nn.Module):
    """
    Combined loss for fire spread prediction:
    - BCE on channel 0 (fire mask — binary classification)
    - MSE on channel 1 (arrival time — continuous regression)
    """
    def __init__(self, mask_weight=1.0, arrival_weight=1.0):
        super().__init__()
        self.bce = nn.BCELoss()
        self.mse = nn.MSELoss()
        self.mask_weight = mask_weight
        self.arrival_weight = arrival_weight
        # Expose individual losses for logging
        self.last_mask_loss = 0.0
        self.last_arrival_loss = 0.0

    def forward(self, pred, target):
        mask_loss = self.bce(pred[:, 0:1], target[:, 0:1])
        arrival_loss = self.mse(pred[:, 1:2], target[:, 1:2])
        self.last_mask_loss = (self.mask_weight * mask_loss).item()
        self.last_arrival_loss = (self.arrival_weight * arrival_loss).item()
        return self.mask_weight * mask_loss + self.arrival_weight * arrival_loss


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2.0 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class FocalLoss(nn.Module):
    """Focal loss for binary segmentation (handles class imbalance)."""
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target == 1, pred, 1 - pred)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        return (focal_weight * bce).mean()


class FireSenseNetLoss(nn.Module):
    """
    Combined loss from FireSenseNet (arxiv 2604.07675):
    - Weighted BCE (positive class weight)
    - Dice loss (spatial overlap)
    - Focal loss (focus on hard boundary pixels)

    L = bce_weight * BCE_w + dice_weight * Dice + focal_weight * Focal
    """
    def __init__(self, bce_weight=0.4, dice_weight=0.3, focal_weight=0.3,
                 pos_weight=3.0, focal_gamma=2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.pos_weight = pos_weight

        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(gamma=focal_gamma)

        # Expose individual weighted losses for logging
        self.last_bce = 0.0
        self.last_dice = 0.0
        self.last_focal = 0.0

    def forward(self, pred, target):
        # Weighted BCE: fire pixels weighted by pos_weight
        weight = torch.where(target == 1, self.pos_weight, 1.0)
        bce = F.binary_cross_entropy(pred, target, weight=weight, reduction='mean')

        dice = self.dice_loss(pred, target)
        focal = self.focal_loss(pred, target)

        # Log weighted contributions
        self.last_bce = (self.bce_weight * bce).item()
        self.last_dice = (self.dice_weight * dice).item()
        self.last_focal = (self.focal_weight * focal).item()
        return self.bce_weight * bce + self.dice_weight * dice + self.focal_weight * focal


class DistanceMapLoss(nn.Module):
    """
    Distance Map Loss for binary segmentation (Caliva et al., MIDL 2019,
    arXiv:1908.03679).

    A boundary-penalizing cross-entropy: each pixel's negative log-likelihood
    is weighted by (1 + d(x)), where d(x) is derived from distance transforms
    of the ground-truth mask (and its complement). Pixels adjacent to the mask
    boundary are weighted up to 2x, pixels far from the boundary keep weight 1,
    steering the network's focus toward hard-to-segment boundary regions
    (here: the fire front).

    Distance map construction, per sample (paper Eq. 2 and Fig. 1):
      inner = distance_transform_edt(mask)   # dist of interior pixels to the
                                             # nearest background pixel
      outer = distance_transform_edt(~mask)  # dist of exterior pixels to the
                                             # nearest mask pixel
    Each is inverted so boundary pixels are high and deep interior/exterior
    pixels are low: w = 1 - dist / dist.max() (0 where the side is empty),
    then combined:
      d(x) = w_inner(x) if mask(x) else w_outer(x)      (d in [0, 1])

    Loss:
      L = mean( (1 + d) * NLL ),  NLL = -y log p - (1 - y) log(1 - p)

    The "+1" floor is the paper's mitigation of the vanishing-gradient issue:
    every pixel retains at least the plain cross-entropy penalty.

    Degenerate samples (all-background or all-foreground) have no boundary,
    so no penalty is applied to them (plain cross-entropy instead).

    The map depends only on the target, so it is a constant weight (no grad).
    It is computed on CPU via scipy at each forward call; the target changes
    every autoregressive step, so there is no cross-step caching.
    """
    def __init__(self):
        super().__init__()
        # Expose individual losses for logging (trainer captures last_*)
        self.last_ce = 0.0
        self.last_penalty = 0.0

    def forward(self, pred, target):
        # pred, target: (N, 1, H, W); pred in [0, 1], target binary
        nll = F.binary_cross_entropy(pred, target, reduction='none')
        d = self._distance_maps(target)
        weighted = nll * (1.0 + d)
        self.last_ce = nll.mean().item()
        self.last_penalty = (d * nll).mean().item()
        return weighted.mean()

    def _distance_maps(self, target):
        t = target.detach().cpu().numpy()
        if t.ndim == 4:
            t = t[:, 0]  # (N, 1, H, W) -> (N, H, W)
        mask = t > 0.5
        n, h, w = mask.shape
        maps = np.zeros((n, h, w), dtype=np.float32)
        for i in range(n):
            m = mask[i]
            # No boundary (all-background or all-foreground): the distance
            # maps are undefined (EDT would measure distances to the array
            # border), so no penalty is applied to this sample.
            if not (m.any() and m.size - int(m.sum()) > 0):
                continue
            inner = distance_transform_edt(m)
            outer = distance_transform_edt(~m)
            w_in = np.zeros_like(inner, dtype=np.float32)
            w_out = np.zeros_like(outer, dtype=np.float32)
            # Both maps are non-empty: the guard above guarantees the mask
            # has interior and exterior pixels, so each EDT has real
            # reference pixels (inner_max, outer_max > 0).
            w_in[m] = 1.0 - inner[m] / inner.max()
            w_out[~m] = 1.0 - outer[~m] / outer.max()
            maps[i] = w_in + w_out  # disjoint supports, so max == sum
        return torch.from_numpy(maps).unsqueeze(1).to(
            device=target.device, dtype=target.dtype
        )

