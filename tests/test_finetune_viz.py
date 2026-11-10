"""Tests for the real-data baseline figures (finetune.viz)."""

import numpy as np
import pytest
import torch

from finetune import viz as rv
from finetune.infer import rollout_fire


class ConstMaskModel(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return [torch.full((x.shape[0], 1, x.shape[2], x.shape[3]), self.value)]


@pytest.fixture()
def rollout(real_store):
    return rollout_fire(ConstMaskModel(0.9), real_store,
                        "V5_2020_2020_00002", device=torch.device("cpu"))


def test_figures_are_written_for_a_real_rollout(rollout, tmp_path):
    paths = [
        rv.render_final_arrival_comparison(rollout, tmp_path / "a.png", "x"),
        rv.render_wind_coherence(rollout, tmp_path / "w.png", "x"),
        rv.render_direct_comparison(rollout, tmp_path / "d.png", "x"),
    ]
    for path in paths:
        p = tmp_path / path.split("/")[-1]
        assert p.is_file() and p.stat().st_size > 5000


def test_direct_comparison_array_uses_the_class_palette(rollout):
    arr = rv.render_direct_comparison_array(
        rollout.pred[-1], rollout.truth[-1], rollout.scored[-1],
        rollout.perim0, title="t")
    assert arr.dtype == np.uint8 and arr.ndim == 3 and arr.shape[2] == 3
    px = arr.reshape(-1, 3)
    for color in (rv.COLOR_TP, rv.COLOR_FP, rv.COLOR_FN, rv.COLOR_START):
        rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
        assert (np.abs(px - rgb).sum(axis=1) == 0).sum() > 10


def test_classify_masks_without_coverage_keeps_every_cell():
    pred = np.zeros((2, 2), bool)
    truth = np.zeros((2, 2), bool)
    pred[0, 0] = truth[0, 0] = True     # TP
    pred[0, 1] = True                   # FP
    truth[1, 0] = True                  # FN
    classes = rv.classify_masks(pred, truth, None, np.zeros((2, 2), bool))
    assert classes.tolist() == [[2, 3], [4, 0]]


def test_classify_final_separates_all_four_classes():
    from finetune.metrics import FireRollout

    shape = (3, 3)
    valid = np.ones(shape, bool)
    truth = np.zeros(shape, bool)
    pred = np.zeros(shape, bool)
    start = np.zeros(shape, bool)
    truth[0, 0] = pred[0, 0] = start[0, 0] = True    # seed: TP recolored to start
    truth[0, 1] = pred[0, 1] = True                  # true positive
    pred[0, 2] = True                                # false positive
    truth[1, 0] = True                               # false negative
    r = FireRollout(
        fire_id="F", name="F", year=2020, state="CA", county="X", start_step=0,
        n_steps=1, pred=[pred], truth=[truth], scored=[valid], perim0=start,
        extent_hours=np.array([12.0]), t_hours=np.array([0.0]),
        pred_arrival_hours=np.full(shape, np.nan, np.float32),
        truth_arrival_hours=np.full(shape, np.nan, np.float32),
        elev=np.zeros(shape, np.float32),
    )
    classes = rv.classify_final(r)
    assert classes[0, 0] == 1        # start wins over true positive
    assert classes[0, 1] == 2
    assert classes[0, 2] == 3
    assert classes[1, 0] == 4
    assert classes[1, 1] == 0        # never burned


def test_unscored_cells_are_not_classified():
    """A prediction outside the real data's coverage must not become a class."""
    from finetune.metrics import FireRollout

    shape = (2, 2)
    r = FireRollout(
        fire_id="F", name="F", year=2020, state="CA", county="X", start_step=0,
        n_steps=1,
        pred=[np.ones(shape, bool)], truth=[np.zeros(shape, bool)],
        scored=[np.array([[True, False], [False, False]])],
        perim0=np.zeros(shape, bool),
        extent_hours=np.array([12.0]), t_hours=np.array([0.0]),
        pred_arrival_hours=np.full(shape, np.nan, np.float32),
        truth_arrival_hours=np.full(shape, np.nan, np.float32),
        elev=np.zeros(shape, np.float32),
    )
    classes = rv.classify_final(r)
    assert classes[0, 0] == 3        # scored false positive
    assert (classes[~np.array([[True, False], [False, False]])] == 0).all()


def test_arrival_colormap_scale_is_shared_and_ordered(rollout):
    """Yellow at the fire's start, red at its end: YlOrRd with vmin 0."""
    vmax = rv._arrival_vmax(rollout)
    assert vmax > 0
    finite = rollout.pred_arrival_hours[np.isfinite(rollout.pred_arrival_hours)]
    assert finite.max() <= vmax + 1e-6
    assert (rollout.pred_arrival_hours[rollout.perim0] == 0.0).all()


def _pixels(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB")).reshape(-1, 3)


def test_figures_encode_the_requested_color_coding(rollout, tmp_path):
    """The drawn colors are the request, not an approximation.

    ``direct_comparison`` uses the yellow/blue/red/gray class palette (exact RGB
    is visible in its legend swatches), the arrival maps use ``YlOrRd`` over
    hours-since-forecast-start (both colormap ends appear in their colorbars),
    and the wind figure actually draws a quiver field.
    """
    cmp_path = rv.render_final_arrival_comparison(rollout, tmp_path / "cmp.png")
    direct = rv.render_direct_comparison(rollout, tmp_path / "direct.png")
    wind = rv.render_wind_coherence(rollout, tmp_path / "wind.png")

    px = _pixels(direct)
    for label, color in (("true positive", rv.COLOR_TP),
                         ("false positive", rv.COLOR_FP),
                         ("false negative", rv.COLOR_FN),
                         ("fire at forecast start", rv.COLOR_START)):
        rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
        assert (np.abs(px - rgb).sum(axis=1) == 0).sum() > 10, label

    from matplotlib import colormaps

    px = _pixels(cmp_path)
    cmap = colormaps[rv.CMAP_ARRIVAL]
    # Both ends of YlOrRd are present (tolerance: the bar and the over-terrain
    # blend are resampled, so exact RGB is not guaranteed).
    for value in (0.0, 1.0):                     # Yl at the start, Rd at the end
        rgb = np.array(cmap(value)[:3]) * 255
        assert (np.abs(px - rgb).sum(axis=1) < 60).sum() > 100, value
    # and the fire pixels form the YlOrRd family: red >= green >= blue
    warm = (px[:, 0] > 180) & (px[:, 2] < 160) & (px[:, 0] >= px[:, 1])
    assert warm.sum() > 1000
    monotone = ((px[:, 0] >= px[:, 1]) & (px[:, 1] >= px[:, 2]))[warm]
    assert monotone.mean() > 0.7

    px = _pixels(wind)
    assert (px < 60).all(axis=1).sum() > 500     # quiver arrows
