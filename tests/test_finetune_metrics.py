"""Tests for the modular real-data metrics (finetune.metrics) and the rollout."""

import numpy as np
import pytest
import torch

from finetune import metrics as rm
from finetune.infer import (arrival_hours, build_model,
                                           rollout_fire)


class ConstMaskModel(torch.nn.Module):
    """Stub model: every cell gets the same probability (no growth if <= 0.5)."""

    def __init__(self, value):
        super().__init__()
        self.value = value
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        return [torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.value)]


def _rollout(pred_steps, truth_steps, scored, **kw):
    """Hand-built FireRollout for metric unit tests (tiny 2x2 windows)."""
    shape = (2, 2)
    n = len(pred_steps)
    rand = np.random.default_rng(0)
    kw.setdefault("pred_arrival_hours", np.full(shape, np.nan, np.float32))
    kw.setdefault("truth_arrival_hours", np.full(shape, np.nan, np.float32))
    return rm.FireRollout(
        fire_id="F", name="F", year=2020, state="CA", county="X",
        start_step=0, n_steps=n,
        pred=[np.asarray(p, bool) for p in pred_steps],
        truth=[np.asarray(t, bool) for t in truth_steps],
        scored=[np.asarray(s, bool) for s in scored],
        perim0=np.asarray(kw.pop("perim0", np.zeros(shape, bool)), bool),
        extent_hours=np.arange(1, n + 1) * 12.0,
        t_hours=np.arange(n) * 12.0,
        elev=np.full(shape, 100.0, np.float32),
        wind_u=np.full(shape, 1.0, np.float32),
        wind_v=np.full(shape, -1.0, np.float32),
        **kw,
    )


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------
def test_default_metrics_are_iou_plus_precision_and_recall():
    mask = np.array([[1, 1], [0, 0]], bool)
    other = np.array([[1, 1], [1, 0]], bool)
    rollout = _rollout([mask, mask], [mask, other],
                       [np.ones((2, 2), bool)] * 2, perim0=mask)
    out = rm.compute_metrics(rollout)
    assert set(out) == {"final_iou", "avg_step_iou", "precision_final",
                        "recall_final", "n_steps"}
    # step 0: identical -> 1.0 ; step 1: inter 2, union 3 -> 2/3
    assert out["avg_step_iou"] == pytest.approx((1.0 + 2 / 3) / 2)
    # "every perimeter unioned to the final timestep" = the last step's truth
    assert out["final_iou"] == pytest.approx(2 / 3)
    # at the last step: 2 predicted cells, both right -> precision 1.0;
    # 3 real cells, 2 found -> recall 2/3
    assert out["precision_final"] == pytest.approx(1.0)
    assert out["recall_final"] == pytest.approx(2 / 3)
    assert out["n_steps"] == 2.0


def test_all_metrics_and_a_metric_slots_in():
    mask = np.array([[1, 0], [0, 0]], bool)
    truth = np.array([[1, 1], [0, 0]], bool)
    rollout = _rollout([mask], [truth], [np.ones((2, 2), bool)],
                       perim0=mask,
                       pred_arrival_hours=np.array([[0.0, 12.0],
                                                    [np.nan, np.nan]], np.float32),
                       truth_arrival_hours=np.array([[0.0, 6.0],
                                                     [np.nan, np.nan]], np.float32))
    out = rm.compute_metrics(rollout, "all")
    assert out["precision_final"] == pytest.approx(1.0)   # 1 predicted, 1 right
    assert out["recall_final"] == pytest.approx(0.5)      # 2 real, 1 found
    # the do-nothing footprint is the prediction itself here
    assert out["persistence_final_iou"] == pytest.approx(0.5)
    # growth: real 2-1=1, predicted 1-1=0
    assert out["growth_ratio"] == pytest.approx(0.0)
    # both maps finite on 2 cells: |0-0| and |12-6| -> 3
    assert out["arrival_mae_hours"] == pytest.approx(3.0)

    @rm.register_metric("test_only_metric", "slots in without touching the script")
    def _extra(r):
        return 0.25

    try:
        assert rm.compute_metrics(rollout, ["test_only_metric"])[
            "test_only_metric"] == 0.25
        with pytest.raises(ValueError, match="already registered"):
            rm.register_metric("test_only_metric")(lambda r: 0.0)
    finally:
        rm._REGISTRY.pop("test_only_metric", None)
    assert "test_only_metric" not in rm.available_metrics()


def test_unknown_metric_fails_loudly():
    rollout = _rollout([np.zeros((2, 2), bool)], [np.zeros((2, 2), bool)],
                       [np.ones((2, 2), bool)])
    with pytest.raises(KeyError, match="unknown metric"):
        rm.compute_metrics(rollout, ["nope"])


def test_aggregate_reports_mean_median_and_drops_non_finite():
    per_fire = {
        "a": {"final_iou": 1.0, "avg_step_iou": 0.5},
        "b": {"final_iou": 0.0, "avg_step_iou": 0.5},
        "c": {"final_iou": float("nan"), "avg_step_iou": 0.5},
    }
    summary = rm.aggregate(per_fire)
    assert summary["final_iou"]["n"] == 2
    assert summary["final_iou"]["mean"] == pytest.approx(0.5)
    assert summary["final_iou"]["median"] == pytest.approx(0.5)
    assert summary["avg_step_iou"]["n"] == 3
    assert rm.aggregate({}) == {}


# ---------------------------------------------------------------------------
# arrival maps
# ---------------------------------------------------------------------------
def test_arrival_hours_marks_seed_and_first_burn_step():
    seed = np.array([[True, False], [False, False]])
    truths = [np.array([[True, False], [False, False]]),
              np.array([[True, True], [False, False]]),
              np.array([[True, True], [True, False]])]
    arr = arrival_hours(truths, np.array([12.0, 24.0, 36.0]), seed=seed)
    assert arr[0, 0] == 0.0        # already burning at the forecast start
    assert arr[0, 1] == 24.0       # first appears in step 1 -> its extent hour
    assert arr[1, 0] == 36.0
    assert np.isnan(arr[1, 1])     # never burned
    assert (arr[seed] == 0.0).all()   # the seed is fire *before* the forecast


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------
def test_rollout_without_growth_stays_at_the_seed(real_store):
    model = ConstMaskModel(0.2)          # never above the 0.5 threshold
    rollout = rollout_fire(model, real_store, "V5_2020_2020_00002",
                           device=torch.device("cpu"))
    assert rollout.n_steps == 3
    for mask in rollout.pred:
        assert np.array_equal(mask, rollout.perim0)
    assert rollout.truth[-1].sum() > rollout.perim0.sum()
    # arrival: seed at hour 0, nothing else predicted
    assert np.array_equal(np.isfinite(rollout.pred_arrival_hours), rollout.perim0)
    assert (rollout.pred_arrival_hours[rollout.perim0] == 0.0).all()


def test_rollout_grows_when_the_model_fires(real_store):
    model = ConstMaskModel(0.9)          # everything burns immediately
    rollout = rollout_fire(model, real_store, "V5_2020_2020_00002",
                           device=torch.device("cpu"))
    assert rollout.pred[-1].all()
    assert np.array_equal(np.isfinite(rollout.pred_arrival_hours), rollout.pred[-1])
    assert (rollout.pred_arrival_hours[rollout.pred[-1]
                                       & ~rollout.perim0] == 12.0).all()
    assert rm.compute_metrics(rollout)["final_iou"] < 1.0


def test_rollout_feeds_its_own_mask_and_drops_observations(real_store):
    model = ConstMaskModel(0.9)
    rollout_fire(model, real_store, "V5_2020_2020_00002",
                 device=torch.device("cpu"))
    chain = real_store.build_chain("V5_2020_2020_00002", 0, chain_len=None)
    frp = real_store.channels.index("frp")
    first = model.inputs[0][0, :, :135, :135]
    assert torch.equal(first[0] > 0.5, torch.from_numpy(chain.perim0))
    assert (first[frp] > 0).any()                 # observations at the first step
    second = model.inputs[1][0, :, :135, :135]
    assert (second[frp] == 0).all()               # none afterwards
    # the mask handed to step 2 is the union of the seed and step 1's forecast
    assert second[0].all()


def test_rollout_uses_the_first_real_fire_sample(real_store):
    """Fire A's step 0 is a 5-cell placeholder: the rollout must start after it."""
    model = ConstMaskModel(0.2)
    rollout = rollout_fire(model, real_store, "V5_2020_2020_00001",
                           device=torch.device("cpu"))
    assert rollout.start_step == 1
    assert rollout.n_steps == 3
    # arrival bands 0-1 of the fixture inside the window: 36 rows x 70 cols
    assert rollout.seed_cells == 36 * 70


def test_build_model_input_width_matches_the_store(real_store):
    model = build_model(real_store, [4, 8, 16, 32, 48])
    assert model.out_channels == 1
    x = torch.zeros(1, len(real_store.channels) + 1, 160, 160)
    out = model(x)
    assert out[0].shape == (1, 1, 160, 160)
    # sigmoid head: probabilities
    assert float(out[0].detach().min()) >= 0.0
    assert float(out[0].detach().max()) <= 1.0
