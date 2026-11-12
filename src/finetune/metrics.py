"""Modular per-fire metrics for the real-data baseline.

Metrics are registered functions of a :class:`FireRollout`, so an experiment can
add one without touching the evaluation script:

    @register_metric("my_metric", "what it measures")
    def _my_metric(r):
        return float(...)

``DEFAULT_METRICS`` is what the evaluation script reports unless ``--metrics``
names others (``--metrics all`` runs every registered one). The defaults are the
two headline IoUs plus the size-normalised pair that makes them interpretable:

* ``final_iou`` — the final model footprint against the fire's final burned
  footprint (every perimeter unioned to the last timestep);
* ``avg_step_iou`` — mean per-step IoU over the rollout, i.e. how well the model
  predicts step-by-step spread rather than just the end state;
* ``precision_final`` — of the cells the model called burned at the end, the
  fraction that really burned. This is the one that falls when the predicted
  union runs away, which ``final_iou`` alone hides (it degenerates into a ratio
  of true to predicted area once the prediction engulfs the fire);
* ``recall_final`` — of the cells that really burned, the fraction the model
  found. Stays high under that same saturation, so the pair separates
  "over-prediction" from "missed spread".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wildfire_simulator import real_spread as rs

# Metrics reported by default: the two headline IoUs plus the precision/recall
# pair that says *why* an IoU is low (over-prediction vs missed spread).
DEFAULT_METRICS = ("final_iou", "avg_step_iou", "precision_final", "recall_final")

_REGISTRY = {}


@dataclass
class FireRollout:
    """One fire's full-life rollout: predictions, truth and everything to draw.

    ``pred[k]`` is the model's cumulative footprint after step ``k`` and
    ``truth[k]`` the real cumulative burned footprint at the same step, both
    ``(135, 135)`` bool in the chain's start window. ``scored[k]`` marks the
    cells the real data can score at that step.
    """

    fire_id: str
    name: str
    year: int | None
    state: str
    county: str
    start_step: int
    n_steps: int
    pred: list
    truth: list
    scored: list
    perim0: np.ndarray
    extent_hours: np.ndarray
    t_hours: np.ndarray
    pred_arrival_hours: np.ndarray
    truth_arrival_hours: np.ndarray
    elev: np.ndarray
    wind_u: np.ndarray | None = None
    wind_v: np.ndarray | None = None
    cell_deg: float = 0.003
    cell_m: float = 330.0
    meta: dict = field(default_factory=dict)

    @property
    def n_cells(self) -> int:
        return int(self.truth[-1][self.scored[-1]].sum())

    @property
    def pred_cells(self) -> int:
        return int(self.pred[-1][self.scored[-1]].sum())

    @property
    def seed_cells(self) -> int:
        return int(self.perim0.sum())

    @property
    def duration_hours(self) -> float:
        return float(self.extent_hours[-1])


def register_metric(name, doc=""):
    """Register a metric function under ``name`` (decorator)."""
    def decorator(fn):
        if name in _REGISTRY:
            raise ValueError(f"metric {name!r} is already registered")
        fn.__doc__ = fn.__doc__ or doc
        _REGISTRY[name] = fn
        return fn
    return decorator


def available_metrics() -> list:
    """Sorted names of every registered metric."""
    return sorted(_REGISTRY)


def compute_metrics(rollout: FireRollout, names=DEFAULT_METRICS) -> dict:
    """Evaluate ``names`` on one rollout (``None``/``"all"`` = every metric)."""
    if names is None or names == "all":
        names = available_metrics()
    out = {}
    for name in names:
        if name not in _REGISTRY:
            raise KeyError(
                f"unknown metric {name!r}; available: {available_metrics()}")
        out[name] = float(_REGISTRY[name](rollout))
    out["n_steps"] = float(rollout.n_steps)
    return out


def aggregate(per_fire: dict, ignore=("n_steps",)) -> dict:
    """Summarize per-fire metrics: mean/median/std/min/max per metric.

    ``per_fire`` maps fire id -> ``{metric: value}``; metrics missing from any
    fire are dropped from the summary (a fire that could not be scored must not
    silently become a zero).
    """
    if not per_fire:
        return {}
    keys = sorted({k for m in per_fire.values() for k in m})
    summary = {}
    for key in keys:
        values = [m[key] for m in per_fire.values() if key in m]
        values = [v for v in values if v is not None and np.isfinite(v)]
        if not values:
            continue
        values = np.asarray(values, dtype=float)
        summary[key] = {
            "n": int(values.size),
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for key in ignore:
        summary.pop(key, None)
    return summary


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
@register_metric("final_iou",
                 "IoU of the final model footprint vs the real final footprint")
def _final_iou(r: FireRollout) -> float:
    return rs.iou(r.pred[-1], r.truth[-1], r.scored[-1])


@register_metric("avg_step_iou",
                 "Mean per-step IoU over the whole rollout")
def _avg_step_iou(r: FireRollout) -> float:
    return float(np.mean([rs.iou(p, t, s) for p, t, s in
                          zip(r.pred, r.truth, r.scored)]))


@register_metric("median_step_iou", "Median per-step IoU over the rollout")
def _median_step_iou(r: FireRollout) -> float:
    return float(np.median([rs.iou(p, t, s) for p, t, s in
                            zip(r.pred, r.truth, r.scored)]))


@register_metric("dice_final", "Dice of the final model vs real footprint")
def _dice_final(r: FireRollout) -> float:
    return rs.binary_metrics(r.pred[-1], r.truth[-1], r.scored[-1])["dice"]


@register_metric("precision_final",
                 "Precision of the final model footprint (over-prediction)")
def _precision_final(r: FireRollout) -> float:
    return rs.binary_metrics(r.pred[-1], r.truth[-1], r.scored[-1])["precision"]


@register_metric("recall_final",
                 "Recall of the final model footprint (under-prediction)")
def _recall_final(r: FireRollout) -> float:
    return rs.binary_metrics(r.pred[-1], r.truth[-1], r.scored[-1])["recall"]


@register_metric("growth_ratio",
                 "Predicted growth / real growth by the final step")
def _growth_ratio(r: FireRollout) -> float:
    real_growth = r.n_cells - r.seed_cells
    if real_growth <= 0:
        return float("nan")
    return (r.pred_cells - r.seed_cells) / real_growth


@register_metric("persistence_final_iou",
                 "Final IoU of the do-nothing (seed-only) footprint")
def _persistence_final_iou(r: FireRollout) -> float:
    return rs.iou(r.perim0, r.truth[-1], r.scored[-1])


@register_metric("arrival_mae_hours",
                 "Mean |predicted - real| first-burn time (hours, both burned)")
def _arrival_mae_hours(r: FireRollout) -> float:
    both = np.isfinite(r.pred_arrival_hours) & np.isfinite(r.truth_arrival_hours)
    if not both.any():
        return float("nan")
    return float(np.abs(r.pred_arrival_hours[both]
                        - r.truth_arrival_hours[both]).mean())
