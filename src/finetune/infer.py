"""Full-life autoregressive inference on real-data chains.

One rollout per fire: seed with that fire's first *real* observed perimeter
(``finetune.dataset.ARTIFACT_THRESHOLD`` filters the pre-fire placeholder), then step
the model forward once per remaining sample, feeding each step its own
meteorology (the chains carry it) and its own predicted mask. The rollout covers
the fire's whole recorded life, so the per-fire metrics are "one IoU per fire"
plus "one mean per-step IoU per fire".

The result (:class:`FireRollout`) carries everything metrics and visualizations
need: per-step predicted and true masks, first-burn (arrival) hour maps, the
elevation under the fire and the mean wind over the rollout.
"""

from __future__ import annotations

import numpy as np
import torch

from finetune import dataset as rd
from finetune.trainer import HOURS_PER_T_UNIT
from wildfire_simulator.trainers import _pad_to_multiple

# Wind channels shown by the wind-coherence figure: terrain-adjusted
# (WindNinja) wind is what drives the local spread, so it is preferred, with
# the HRRR 10 m wind as fallback.
_WIND_CHANNEL_PAIRS = (("wn_u10", "wn_v10"), ("u10", "v10"))


def _wind_channels(store):
    """Local indices of the (u, v) wind channels in the store's channel list."""
    for u_name, v_name in _WIND_CHANNEL_PAIRS:
        if u_name in store.channels and v_name in store.channels:
            return store.channels.index(u_name), store.channels.index(v_name)
    return None


def elevation_channel(store):
    """Local index of the ``elev`` channel, or ``None`` when not loaded."""
    return store.channels.index("elev") if "elev" in store.channels else None


def build_model(store, model_channels=None, final_activation="sigmoid"):
    """The baseline's MK-UNet: one mask output, ``len(store.channels) + 1`` in.

    The extra input channel is the step-time channel the trainer appends
    (``t_hours / HOURS_PER_T_UNIT``); keeping that "+1" here means the training
    and evaluation scripts can never disagree about the input width.
    """
    from wildfire_simulator.models import MK_UNet_Regression

    return MK_UNet_Regression(
        in_channels=len(store.channels) + 1,
        out_channels=1,
        channels=list(model_channels or [16, 32, 64, 96, 160]),
        final_activation=final_activation,
    )


def arrival_hours(truths, extent_hours, seed=None) -> np.ndarray:
    """First-burn hour per cell (NaN where the cell never burns).

    ``truths[k]`` is the cumulative burned mask at ``extent_hours[k]`` hours
    after the rollout start, so a cell's arrival is the first step that contains
    it. ``seed`` cells (the perimeter already burning at the rollout start) get
    hour 0 — they are fire *before* the forecast, not 12-hour growth.
    """
    arr = np.full(truths[0].shape, np.nan, dtype=np.float32)
    if seed is not None:
        arr[np.asarray(seed, dtype=bool)] = 0.0
    for k, mask in enumerate(truths):
        newly = mask & ~np.isfinite(arr)
        arr[newly] = extent_hours[k]
    return arr


@torch.no_grad()
def rollout_fire(model, store, fire_id, device=None, chain=None) -> "FireRollout":
    """Run one full-life autoregressive rollout for a fire.

    Args:
        model: trained MK-UNet (returns a list whose first item is the mask).
        store: :class:`RealDataStore` (supplies channels and normalization).
        fire_id: real-data fire id.
        device: torch device (defaults to the model's parameter device).
        chain: pre-built full-life chain (skipped when given, so a caller can
            reuse a chain it already read).

    Returns:
        :class:`FireRollout`.
    """
    from finetune.metrics import FireRollout

    device = device or next(model.parameters()).device
    if chain is None:
        chain = rd.build_full_life_chain(store, fire_id)

    row = store.fire_row(fire_id)
    x = torch.from_numpy(chain.x)                    # (L, C, H, W) raw, NaN
    x = store.transform()(x)
    x = torch.nan_to_num(x, nan=0.0)
    x = x.to(device)
    L, C, H, W = x.shape

    perim0 = torch.from_numpy(chain.perim0).to(device) > 0.5
    state = perim0                                   # (H, W) bool
    t_hours = torch.from_numpy(chain.t_hours).to(device)

    was_training = model.training
    model.eval()
    preds = []
    for k in range(L):
        inp = x[k].unsqueeze(0).clone()              # (1, C, H, W)
        # Equal-shape assignment (no broadcasting): see the broadcast note in
        # finetune.trainer's module docstring.
        inp[:, 0:1] = state.to(inp.dtype).unsqueeze(0).unsqueeze(0)
        t_ch = (t_hours[k] / HOURS_PER_T_UNIT).view(1, 1)
        t_ch = t_ch.expand(1, 1, H, W)
        inputs = torch.cat([inp, t_ch], dim=1)
        inputs, _, _ = _pad_to_multiple(inputs, 32)
        out = model(inputs)
        if isinstance(out, (list, tuple)):
            out = out[0]
        pred_mask = out[0, 0, :H, :W] > 0.5          # (H, W)
        # Burned land does not unburn (ROLLOUT.md's inference loop).
        state = state | pred_mask
        preds.append(state.cpu().numpy().copy())
    model.train(was_training)

    wind = _wind_channels(store)
    wind_u = wind_v = None
    if wind is not None:
        ui, vi = wind
        # NaN (cells a later window did not cover) drops out of the mean.
        wind_u = np.nanmean(chain.x[:, ui], axis=0)
        wind_v = np.nanmean(chain.x[:, vi], axis=0)

    elev_idx = elevation_channel(store)
    elev = (chain.x[0, elev_idx] if elev_idx is not None
            else np.full((H, W), np.nan, dtype=np.float32))

    truth = [m for m in chain.y]
    return FireRollout(
        fire_id=str(fire_id),
        name=str(row.get("name", fire_id)),
        year=rd._fire_year(fire_id),
        state=str(row.get("state", "")),
        county=str(row.get("county", "")),
        start_step=chain.start,
        n_steps=L,
        pred=preds,
        truth=truth,
        scored=[m for m in chain.scored],
        perim0=chain.perim0,
        extent_hours=np.asarray(chain.extent_hours, dtype=np.float32),
        t_hours=np.asarray(chain.t_hours, dtype=np.float32),
        pred_arrival_hours=arrival_hours(preds, chain.extent_hours,
                                         seed=chain.perim0),
        truth_arrival_hours=arrival_hours(truth, chain.extent_hours,
                                          seed=chain.perim0),
        elev=elev,
        wind_u=wind_u,
        wind_v=wind_v,
        cell_deg=rd.CELL_DEG,
        cell_m=rd.CELL_M,
    )
