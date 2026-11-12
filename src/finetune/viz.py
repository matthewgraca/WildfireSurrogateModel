"""Visualizations for the real-data baseline.

Three figures per fire, all overlaid on the elevation under the fire (read from
the real data — no basemap or API call):

* ``final_arrival_comparison`` — side-by-side real vs predicted fire arrival
  time (first-burn hours) over terrain,
* ``wind_coherence`` — the same pair with the mean wind over the rollout drawn
  as a quiver field, so spread direction can be read against the wind,
* ``direct_comparison`` — the final real and predicted footprints overlaid on
  terrain, classified cell by cell (true positive / false positive / false
  negative / already burning at the forecast start).

Arrival maps use ``YlOrRd`` on a shared hour scale: early fire approaches
yellow, late fire approaches red. Unburned cells are transparent.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, Normalize
from matplotlib.patches import Patch

CMAP_ARRIVAL = "YlOrRd"

# Direct-comparison classes (spec: yellow = both burned, blue = predicted only,
# red = real only, gray = the fire already burning when the forecast started).
COLOR_TP = "#ffd400"      # true positive
COLOR_FP = "#1f77b4"      # false positive (predicted grew, real did not)
COLOR_FN = "#d62728"      # false negative (real grew, predicted did not)
COLOR_START = "#9e9e9e"   # fire at forecast start


def _finite_elev(rollout):
    """Elevation with NaN cells filled by the local mean (for the relief)."""
    elev = np.asarray(rollout.elev, dtype=float)
    if not np.isfinite(elev).any():
        return np.zeros(elev.shape, dtype=float)
    return np.where(np.isfinite(elev), elev, np.nanmean(elev))


def _draw_terrain(ax, rollout, alpha=1.0):
    """Elevation relief as a background layer (hillshade + terrain colormap)."""
    elev = _finite_elev(rollout)
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.shade(elev, cmap=plt.get_cmap("terrain"), blend_mode="soft",
                   vert_exag=1.0, dx=rollout.cell_m, dy=rollout.cell_m)
    ax.imshow(rgb, origin="lower", interpolation="bilinear", alpha=alpha,
              extent=[0, elev.shape[1], 0, elev.shape[0]])


def _draw_arrival(ax, arr, vmax, alpha=0.8):
    """Arrival hours over the terrain layer (NaN cells stay transparent)."""
    masked = np.where(np.isfinite(arr), arr, np.nan)
    return ax.imshow(masked, cmap=CMAP_ARRIVAL, origin="lower", alpha=alpha,
                     vmin=0.0, vmax=vmax,
                     extent=[0, arr.shape[1], 0, arr.shape[0]])


def _arrival_vmax(rollout) -> float:
    """Shared color maximum for the predicted and real arrival maps."""
    finite = np.concatenate([
        rollout.pred_arrival_hours[np.isfinite(rollout.pred_arrival_hours)],
        rollout.truth_arrival_hours[np.isfinite(rollout.truth_arrival_hours)],
    ])
    if finite.size == 0:
        return max(rollout.duration_hours, 1.0)
    return float(max(finite.max(), 1e-6))


def _quiver_wind(ax, rollout, n_arrows=12):
    """Mean rollout wind as a quiver field; returns (mean_speed, mean_dir)."""
    if rollout.wind_u is None or rollout.wind_v is None:
        return None
    u = np.asarray(rollout.wind_u, dtype=float)
    v = np.asarray(rollout.wind_v, dtype=float)
    h, w = u.shape
    step = max(1, h // n_arrows)
    yy, xx = np.mgrid[0:h:step, 0:w:step]
    uu = u[::step, ::step]
    vv = v[::step, ::step]
    mean_speed = float(np.sqrt(uu ** 2 + vv ** 2).mean())
    # Wind direction (meteorological: where the wind comes FROM).
    mean_dir = (np.degrees(np.arctan2(-float(uu.mean()), -float(vv.mean())))
                + 360) % 360
    ax.quiver(xx, yy, uu, vv, angles="xy", scale_units="xy",
              scale=1.0 / (step * 0.35), color="black", alpha=0.75,
              width=0.003, headwidth=3.0)
    return mean_speed, mean_dir


def _titles(rollout, subtitle=""):
    head = f"{rollout.name} ({rollout.fire_id}) — {rollout.year} {rollout.state}"
    detail = (f"forecast start sample {rollout.start_step}, {rollout.n_steps} "
              f"steps / {rollout.duration_hours:.0f} h")
    if subtitle:
        detail = f"{detail}   |   {subtitle}"
    return head, detail


def _save(fig, out_png):
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return str(out_png)


def _final_arrival_figure(rollout, subtitle=""):
    """Figure for :func:`render_final_arrival_comparison` (no file I/O)."""
    vmax = _arrival_vmax(rollout)
    norm = Normalize(vmin=0.0, vmax=vmax)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
    for ax, arr, title in (
        (axes[0], rollout.truth_arrival_hours, "Real (FEDS) arrival time"),
        (axes[1], rollout.pred_arrival_hours, "Model arrival time"),
    ):
        _draw_terrain(ax, rollout)
        im = _draw_arrival(ax, arr, vmax)
        ax.set_title(title)
        ax.set_xlabel("east (cells)")
        ax.set_ylabel("north (cells)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    cbar.set_label("fire arrival (hours since forecast start)")
    head, detail = _titles(rollout, subtitle)
    fig.suptitle(f"{head}\n{detail}", fontsize=11)
    return fig


def render_final_arrival_comparison(rollout, out_png, subtitle=""):
    """Real vs predicted fire arrival time over terrain (shared hour scale)."""
    return _save(_final_arrival_figure(rollout, subtitle), out_png)


def render_wind_coherence(rollout, out_png, subtitle=""):
    """Arrival time maps with the rollout-mean wind quiver overlain."""
    vmax = _arrival_vmax(rollout)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6))
    wind_stats = None
    for ax, arr, title in (
        (axes[0], rollout.truth_arrival_hours, "Real (FEDS) arrival time"),
        (axes[1], rollout.pred_arrival_hours, "Model arrival time"),
    ):
        _draw_terrain(ax, rollout)
        im = _draw_arrival(ax, arr, vmax)
        wind_stats = _quiver_wind(ax, rollout)
        ax.set_title(title)
        ax.set_xlabel("east (cells)")
        ax.set_ylabel("north (cells)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.035, pad=0.02)
    label = "fire arrival (hours since forecast start)"
    if wind_stats is not None:
        speed, direction = wind_stats
        label += f"\nquiver: mean wind {speed:.1f} m/s from {direction:.0f}°"
    else:
        label += "\nquiver: wind channels not loaded"
    cbar.set_label(label)
    head, detail = _titles(rollout, subtitle)
    fig.suptitle(f"{head}\n{detail}", fontsize=11)
    return _save(fig, out_png)


def classify_masks(pred, truth, scored, perim0):
    """Classify cells of two final footprints: 0 none, 1 start, 2 TP, 3 FP, 4 FN.

    Predictions outside ``scored`` (cells the real data cannot score) are
    dropped before classifying, so an unscoreable guess never becomes a class.
    ``scored=None`` means every cell is scorable.
    """
    pred = np.asarray(pred, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    valid = (np.ones(truth.shape, dtype=bool) if scored is None
             else np.asarray(scored, dtype=bool))
    pred = pred & valid
    truth = truth & valid
    start = np.asarray(perim0, dtype=bool) & valid
    classes = np.zeros(truth.shape, dtype=np.uint8)
    classes[pred & truth] = 2
    classes[pred & ~truth] = 3
    classes[~pred & truth] = 4
    classes[start] = 1                      # the seed burns in both, so it wins
    return classes


def classify_final(rollout):
    """``classify_masks`` on a rollout's final footprints."""
    return classify_masks(rollout.pred[-1], rollout.truth[-1],
                          rollout.scored[-1], rollout.perim0)


def _direct_figure(classes, title, on_terrain=None):
    """Class-coloured footprint figure; ``on_terrain`` is a rollout or ``None``."""
    from matplotlib.colors import ListedColormap, BoundaryNorm

    cmap = ListedColormap(["#ffffff00", COLOR_START, COLOR_TP, COLOR_FP, COLOR_FN])
    norm = BoundaryNorm(np.arange(-0.5, 5.5, 1.0), cmap.N)

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    if on_terrain is not None:
        _draw_terrain(ax, on_terrain)
    ax.imshow(classes, cmap=cmap, norm=norm, origin="lower", alpha=1.0,
              interpolation="nearest",
              extent=[0, classes.shape[1], 0, classes.shape[0]])
    tp = int((classes == 2).sum())
    fp = int((classes == 3).sum())
    fn = int((classes == 4).sum())
    start = int((classes == 1).sum())
    ax.legend(
        handles=[
            Patch(facecolor=COLOR_TP, edgecolor="0.2", label=f"true positive ({tp})"),
            Patch(facecolor=COLOR_FP, edgecolor="0.2", label=f"false positive ({fp})"),
            Patch(facecolor=COLOR_FN, edgecolor="0.2", label=f"false negative ({fn})"),
            Patch(facecolor=COLOR_START, edgecolor="0.2",
                  label=f"fire at forecast start ({start})"),
        ],
        loc="upper right", fontsize=9, framealpha=0.85,
    )
    ax.set_xlabel("east (cells)")
    ax.set_ylabel("north (cells)")
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def render_direct_comparison(rollout, out_png, subtitle=""):
    """Final real vs predicted footprint on terrain, cell-classified."""
    head, detail = _titles(rollout, subtitle)
    fig = _direct_figure(
        classify_final(rollout),
        f"{head}\nFinal {rollout.n_steps} steps / "
        f"{rollout.duration_hours:.0f} h   |   {subtitle}",
        on_terrain=rollout,
    )
    return _save(fig, out_png)


def render_direct_comparison_array(pred, truth, scored, perim0, title,
                                   dpi=90):
    """Class-coloured final footprint as a uint8 RGB array (TensorBoard).

    Same colour coding as :func:`render_direct_comparison` but without terrain:
    the training loop only carries masks (no elevation), and a flat background
    keeps the tracked image legible at TensorBoard size.
    """
    from wildfire_simulator.viz import render_to_array

    fig = _direct_figure(classify_masks(pred, truth, scored, perim0), title)
    return render_to_array(fig, dpi)


# ---------------------------------------------------------------------------
# aggregate analysis: metric vs fire length
# ---------------------------------------------------------------------------
_DURATION_LABELS = {
    "avg_step_iou": ("mean per-step IoU", "C0"),
    "precision_final": ("precision (final footprint)", "C2"),
    "recall_final": ("recall (final footprint)", "C3"),
    "final_iou": ("final footprint IoU", "C1"),
    "median_step_iou": ("median per-step IoU", "C4"),
    "dice_final": ("Dice (final footprint)", "C5"),
    "growth_ratio": ("predicted / real growth", "C6"),
}


def duration_bin_summary(series, n_steps, bins=6) -> dict:
    """Summarize metrics over quantile bins of fire length.

    Quantile bins (not fixed-width) keep every bin populated despite the heavy
    right tail of fire durations. Returns the bin geometry, each series'
    median and inter-quartile range per bin, and the Spearman rank correlation
    of each series against the raw (unbinned) fire length — so a caller can
    print the table without touching matplotlib.

    Args:
        series: ``{name: values}`` with one value per fire.
        n_steps: fire length per fire (the x axis).
        bins: number of quantile bins.

    Returns:
        ``{"bins": [{"lo","hi","center","n", <name>: {"median","q1","q3"}}],
        "series": [names], "spearman": {name: (rho, p)}}``.
    """
    from scipy import stats

    x = np.asarray(n_steps, dtype=float)
    if x.size == 0:
        raise ValueError("duration_bin_summary needs at least one fire")
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, bins + 1)))
    rows = []
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        last = i == len(edges) - 2
        mask = (x >= lo) & (x <= hi if last else x < hi)
        if not mask.any():
            continue
        row = {"lo": float(lo), "hi": float(hi),
               "center": float(np.median(x[mask])), "n": int(mask.sum())}
        for name, values in series.items():
            v = np.asarray(values, dtype=float)[mask]
            row[name] = {"median": float(np.median(v)),
                         "q1": float(np.percentile(v, 25)),
                         "q3": float(np.percentile(v, 75))}
        rows.append(row)
    spearman = {}
    for name, values in series.items():
        rho, p = stats.spearmanr(x, np.asarray(values, dtype=float))
        spearman[name] = (float(rho), float(p))
    return {"bins": rows, "series": list(series), "spearman": spearman}


def duration_metrics_figure(summary, title=""):
    """Median-with-IQR lines of each metric against fire length (no scatter).

    One line per series: the median of a quantile bin, with the inter-quartile
    range as the vertical bar. Fire-length points are too dense to overplot
    readably (109+ fires), so the bins carry the trend and the spread.
    """
    from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter

    bins = summary["bins"]
    if not bins:
        raise ValueError("duration_metrics_figure needs a non-empty summary")
    fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=130)
    for i, name in enumerate(summary["series"]):
        label, color = _DURATION_LABELS.get(name, (name, f"C{i % 10}"))
        xs = [row["center"] for row in bins]
        med = np.array([row[name]["median"] for row in bins])
        lo = np.array([row[name]["q1"] for row in bins])
        hi = np.array([row[name]["q3"] for row in bins])
        rho, p = summary["spearman"][name]
        ax.errorbar(xs, med, yerr=[med - lo, hi - med], color=color, lw=2.0,
                    capsize=3, marker="o", ms=5, markeredgecolor="white",
                    markeredgewidth=0.7, zorder=5,
                    label=f"{label}    ρ = {rho:+.2f}  (p {p:.0e})")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([12, 24, 48, 96, 192]))
    ax.xaxis.set_major_formatter(FixedFormatter(["12", "24", "48", "96", "192"]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(min(row["lo"] for row in bins) * 0.9,
                max(row["hi"] for row in bins) * 1.1)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("fire length (12-hour steps; 2 steps = 1 day)")
    ax.set_ylabel("metric value")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if title:
        ax.set_title(title, fontsize=10.5)
    fig.tight_layout()
    return fig


def render_duration_metrics(summary, out_png, title=""):
    """Write the metric-vs-duration figure; returns the path."""
    return _save(duration_metrics_figure(summary, title), out_png)
