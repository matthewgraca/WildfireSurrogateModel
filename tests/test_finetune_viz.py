"""Tests for the real-data baseline figures (finetune.viz)."""

from pathlib import Path

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


# ---------------------------------------------------------------------------
# aggregate analysis: metric vs fire length
# ---------------------------------------------------------------------------
def test_duration_bin_summary_bins_medians_and_spearman():
    n_steps = np.array([10, 10, 20, 20, 30, 30, 40, 40], dtype=float)
    series = {"m": np.array([1.0, 3.0, 2.0, 4.0, 5.0, 7.0, 6.0, 8.0])}
    s = rv.duration_bin_summary(series, n_steps, bins=2)

    assert [b["n"] for b in s["bins"]] == [4, 4]
    assert s["bins"][0]["lo"] == 10.0 and s["bins"][1]["hi"] == 40.0
    # first half {1,2,3,4}: median 2.5, quartiles 1.75 / 3.25
    assert s["bins"][0]["m"]["median"] == pytest.approx(2.5)
    assert s["bins"][0]["m"]["q1"] == pytest.approx(1.75)
    assert s["bins"][0]["m"]["q3"] == pytest.approx(3.25)
    # second half {5,6,7,8}
    assert s["bins"][1]["m"]["median"] == pytest.approx(6.5)
    # monotone-ish series (ties in both variables) -> strong positive rank correlation
    rho, p = s["spearman"]["m"]
    assert rho > 0.8 and p < 0.05


def test_duration_bin_summary_single_bin_and_empty():
    s = rv.duration_bin_summary({"m": np.array([1.0, 2.0])},
                                np.array([5.0, 9.0]), bins=1)
    assert len(s["bins"]) == 1 and s["bins"][0]["n"] == 2
    with pytest.raises(ValueError, match="at least one fire"):
        rv.duration_bin_summary({"m": np.array([])}, np.array([]))


def test_duration_metrics_figure_has_one_series_per_metric(tmp_path):
    rng = np.random.default_rng(0)
    n_steps = np.linspace(11, 174, 60)
    series = {name: rng.uniform(0, 1, n_steps.size)
              for name in ("avg_step_iou", "precision_final", "recall_final")}
    s = rv.duration_bin_summary(series, n_steps, bins=4)
    path = rv.render_duration_metrics(s, tmp_path / "d.png", title="t")
    assert Path(path).is_file() and Path(path).stat().st_size > 5000

    fig = rv.duration_metrics_figure(s, title="t")
    ax = fig.axes[0]
    labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert len(labels) == 3
    assert all("ρ" in text for text in labels)
    # no per-fire scatter: only the binned medians are drawn as data
    assert ax.get_xscale() == "log"
