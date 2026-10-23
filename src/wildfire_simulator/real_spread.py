"""Compare the surrogate's inferred per-step fire spread against a real
per-step fire product (NASA FEDS perimeters on VIIRS).

This module is a thin, pure-data layer the adapter script
``scripts/compare_real_spread.py`` drives. It consumes a completed
``infer_real.py`` run (``results/rollout.npz`` + ``result.json`` + ``fire.json``
+ the georeferenced ``landscape_resampled.tif``) and a real-data folder, and
produces per-step / final IoU plus side-by-side spread and arrival visuals.

The model's raw per-step masks (``rollout.npz`` ``masks``) are used as the
honest prediction; the cleaned largest-component perimeter that
``infer_real.py`` derives for its CAL FIRE comparison is **not** used here.
Only the fire-spread portion of the real samples (per-step observed perimeter
channel 0 of ``X``, the ``y`` label, timestamps and window corners) is read; the
54 training-feature channels are ignored.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

# The repo's stray PROJ_LIB/PROJ_DATA (bundled Windows GDAL data) must be
# neutralized before any pyproj import, exactly as the farsite tools do.
for _proj_key in ("PROJ_LIB", "PROJ_DATA"):
    os.environ.pop(_proj_key, None)

import numpy as np
import pandas as pd

_WIN = 135          # real-data per-step window edge (cells)
_CELL_DEG = 0.003   # real-data cell size in degrees
_CROP = 500         # surrogate crop edge (cells)
_HALF = 250         # surrogate crop half-extent
_EPSG_5070 = "EPSG:5070"

_EPS = 1e-9

# The real-data t_input/t_label are the fire's *local* wall-clock time (the
# dataset carries no timezone column), while the surrogate window and the
# dataset's ignition_date are UTC. We map a fire's centroid longitude to a
# nominal IANA US zone and convert local -> UTC via zoneinfo (handles DST).
# Band edges are approximate for the irregular US timezone boundaries.
_LON_TZ_BANDS = [
    (-85.0, "America/New_York"),
    (-102.5, "America/Chicago"),
    (-110.0, "America/Denver"),
    (-130.0, "America/Los_Angeles"),
    (-150.0, "America/Anchorage"),
]
# Pre-fire samples carry a small constant placeholder perimeter (15 cells in
# practice); real fire detections are far larger. This is the cutoff used to
# anchor the chain at the first sample that actually holds fire.
_ARTIFACT_THRESHOLD = 20


def _lon_to_tz(lon) -> str:
    """Nominal IANA US timezone for a centroid longitude (approximate bands)."""
    for edge, zone in _LON_TZ_BANDS:
        if lon >= edge:
            return zone
    return "Pacific/Honolulu"


def _parse_local_utc(value, tz_name=None) -> datetime:
    """Parse a real-data timestamp (fire local wall time) into naive UTC.

    ``tz_name`` is an IANA zone (e.g. ``America/Los_Angeles``); when given the
    local time is shifted to UTC (handling DST). When ``None`` it is treated as
    UTC already.
    """
    dt = _parse_utc(value)
    if tz_name:
        from zoneinfo import ZoneInfo

        dt = (dt.replace(tzinfo=ZoneInfo(tz_name))
                 .astimezone(timezone.utc).replace(tzinfo=None))
    return dt


class FireMatchError(ValueError):
    """Raised when a real-data fire cannot be matched unambiguously."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _parse_utc(value) -> datetime:
    """Parse a timestamp string/``datetime`` into naive UTC.

    Handles ``2025-01-07 00:00:00`` (real-data), ``2025-01-07T17:00Z``
    (result.json) and any ISO-8601 offset.
    """
    if isinstance(value, datetime):
        dt = value
    else:
        s = str(value).strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        if " " in s and "T" not in s:
            s = s.replace(" ", "T", 1)
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _surrogate_crop_affine(land_transform, crs, lon, lat, cell_m):
    """Reproduce the 500x500 crop affine used by ``infer_real.py``.

    ``infer_real.py`` crops the landscape around the ignition pixel:
    ``crop_affine = land_transform * Affine.translation(cx - 250, cy - 250)``.
    """
    from rasterio.transform import Affine, rowcol
    from pyproj import Transformer

    trans = Transformer.from_crs(4326, crs, always_xy=True)
    x, y = trans.transform(lon, lat)
    cy, cx = rowcol(land_transform, x, y)
    return land_transform * Affine.translation(cx - _HALF, cy - _HALF)


def _fire_year(row) -> int:
    """Derive the year from a V5 fire id (``V5_<year>_<year>_<id>``)."""
    parts = str(row["fire_id"]).split("_")
    try:
        return int(parts[1]) if len(parts) > 1 else None
    except (ValueError, IndexError):
        return None


def _trailing_digits(value, n=6) -> str:
    """Return the last ``n`` digits of the digit-only string of ``value``."""
    digits = "".join(ch for ch in str(value) if ch.isdigit())
    return digits[-n:] if len(digits) >= n else digits


def _disambiguate(df, key, value):
    """Filter ``df`` (candidate real fires) by a disambiguator, return row or None."""
    if key == "inc":
        m = df[df["incident_number"].astype(str).map(_trailing_digits) ==
               _trailing_digits(value)]
    elif key == "state":
        m = df[df["state"].astype(str).str.upper() == str(value).strip().upper()]
    elif key == "county":
        m = df[df["county"].astype(str).str.upper() == str(value).strip().upper()]
    elif key == "fire_id":
        m = df[df["fire_id"].astype(str) == str(value)]
    elif key in ("lon", "lat"):
        if len(df) == 0:
            return None
        want = float(value)
        col = "centroid_lon" if key == "lon" else "centroid_lat"
        nearest = df.iloc[np.abs(df[col].astype(float) - want).argmin()]
        return nearest
    else:
        raise ValueError(f"unknown disambiguator {key!r}")
    if len(m) == 1:
        return m.iloc[0]
    return None


# ---------------------------------------------------------------------------
# public API
# ---------------------------------------------------------------------------
def load_fire(fires_csv, name=None, year=None, disambiguator=None,
              fire_id=None, run_inc=None):
    """Select a real-data fire row from ``fires.csv``.

    ``name``+``year`` are matched case-insensitively (year derived from the V5
    fire id). ``fire_id`` short-circuits name matching. ``disambiguator`` is a
    ``(key, value)`` pair used only to break name+year ties; ``run_inc`` is the
    run's ``inc_num`` used as a default incident-number tiebreak.
    """
    df = pd.read_csv(fires_csv)
    if fire_id is not None:
        m = df[df["fire_id"].astype(str) == str(fire_id)]
        if len(m) != 1:
            raise FireMatchError(
                f"no single real fire with fire_id {fire_id!r} (found {len(m)})")
        return m.iloc[0]

    df = df.copy()
    df["_yr"] = df.apply(_fire_year, axis=1)
    m = df[df["_yr"] == int(year)]
    m = m[m["name"].astype(str).str.strip().str.upper() ==
          str(name).strip().upper()]
    if len(m) == 0:
        raise FireMatchError(f"no real fire {name!r} year {year}")
    if len(m) == 1:
        return m.iloc[0]

    # ambiguous: try an explicit disambiguator, then the run's incident number
    if disambiguator is not None:
        row = _disambiguate(m, *disambiguator)
        if row is not None:
            return row
    if run_inc is not None:
        row = _disambiguate(m, "inc", run_inc)
        if row is not None:
            return row
    cands = m[["fire_id", "incident_number", "state", "county"]].to_string(index=False)
    raise FireMatchError(
        f"ambiguous: multiple real fires {name!r} year {year}; pass "
        f"--fire-id/--inc/--state/--county\n{cands}")


def shift_window(a, src, dst, fill):
    """Move a ``(..., H, W)`` array between windows with corners ``src``/``dst``.

    Verbatim from ROLLOUT.md.
    """
    dr, dc = dst[0] - src[0], dst[1] - src[1]
    H, W = a.shape[-2:]
    out = np.full(a.shape, fill, dtype=a.dtype)
    r0, r1 = max(0, -dr), min(H, H - dr)
    c0, c1 = max(0, -dc), min(W, W - dc)
    if r0 < r1 and c0 < c1:
        out[..., r0:r1, c0:c1] = a[..., r0 + dr:r1 + dr, c0 + dc:c1 + dc]
    return out


@dataclass
class Chain:
    """A real-data chain laid out in its start window.

    ``truths[0]`` is the *initial observed perimeter* (the small ignition fire)
    at ``truth_times[0]`` (the start sample's ``t_input``); ``truths[k]`` for
    ``k >= 1`` is the cumulative burned map at the ``k``-th label time. All
    times are naive UTC.
    """
    truths: np.ndarray       # (n, WIN, WIN) bool; frame 0 = initial perimeter
    scored: np.ndarray       # (n, WIN, WIN) bool; cells the real data can score
    truth_times: list        # (n,) naive UTC datetimes
    grid: dict               # lat0, lon0, row0, col0, cell for the start window
    start_perim: np.ndarray  # (WIN, WIN) bool; observed perimeter at the start step


def first_fire_step(samples_npz, threshold=_ARTIFACT_THRESHOLD) -> int:
    """Index of the first sample holding real fire (observed perimeter count
    above the pre-fire placeholder artifact); 0 if none."""
    z = np.load(samples_npz)
    counts = (z["X"][:, :, :, 0] > 0.5).sum(axis=(1, 2))
    idx = np.flatnonzero(counts > threshold)
    return int(idx[0]) if idx.size else 0


def load_chain(fire_id, samples_csv, samples_npz, start_step, steps,
               tz_name=None) -> Chain:
    """Build a cumulative-burned chain in the start window.

    Follows ROLLOUT.md except that frame 0 is the *initial observed perimeter*
    at the start sample's ``t_input``, then each label is accumulated. When
    ``tz_name`` (IANA zone) is given, the real-data local timestamps are
    converted to UTC; otherwise they are assumed already UTC.
    """
    z = np.load(samples_npz)
    X, y = z["X"], z["y"]
    rows = pd.read_csv(samples_csv)
    rows = rows[rows["fire_id"].astype(str) == str(fire_id)]
    rows = rows.sort_values("step").reset_index(drop=True)
    origin = list(zip(rows["row0"].astype(int), rows["col0"].astype(int)))
    if start_step >= len(X):
        raise ValueError(f"start_step {start_step} beyond {len(X)} samples")
    L = min(int(steps), len(X) - start_step)
    if L <= 0:
        raise ValueError("chain length <= 0")

    frame = origin[start_step]
    start_perim = X[start_step, :, :, 0] > 0.5
    # frame 0 = the initial observed fire (the small ignition perimeter)
    truths = [start_perim.copy()]
    scored = [np.ones((_WIN, _WIN), dtype=bool)]
    truth_times = [_parse_local_utc(rows.iloc[start_step]["t_input"], tz_name)]
    burned = start_perim.copy()
    for k in range(L):
        j = start_step + k
        label = shift_window(y[j] > 0, origin[j], frame, False)
        inside = shift_window(np.ones(label.shape, bool), origin[j], frame, False)
        burned = burned | label
        truths.append(burned.copy())
        scored.append(inside | burned)
        truth_times.append(_parse_local_utc(rows.iloc[j]["t_label"], tz_name))
    grid = {
        "lat0": float(rows.iloc[start_step]["lat0"]),
        "lon0": float(rows.iloc[start_step]["lon0"]),
        "row0": int(rows.iloc[start_step]["row0"]),
        "col0": int(rows.iloc[start_step]["col0"]),
        "cell": _CELL_DEG,
    }
    return Chain(np.stack(truths), np.stack(scored), truth_times, grid,
                 start_perim)


def surrogate_extent_times(start, end, times_norm) -> list:
    """Real (UTC) time each surrogate mask represents.

    Matches the ``render`` convention: the fire extent produced at loop time
    ``times[i]`` corresponds to ``times[i] + dt``, i.e. real time
    ``start + (extent_norm/max_t) * (end - start)``.
    """
    times_norm = np.asarray(times_norm, dtype=float)
    if times_norm.size < 2:
        raise ValueError("times_norm must have >= 2 entries")
    dt = times_norm[1] - times_norm[0]
    max_t = times_norm[-1] + dt
    extent_norm = times_norm + dt
    total = (end - start).total_seconds()
    return [start + timedelta(seconds=(v / max_t) * total) for v in extent_norm]


def align_truth_to_surrogate(truth_times, surr_times, favor_ahead=True) -> list:
    """Align each real-data step to a surrogate step by nearest real time.

    Monotonic (no lookahead): for each truth time, only surrogate steps at or
    after the previously-selected index are considered. On an exact tie between
    an earlier and a later surrogate step, the **later** one is chosen when
    ``favor_ahead`` (the model slightly ahead), else the earlier one.
    """
    idxs = []
    prev = 0
    n = len(surr_times)
    for tt in truth_times:
        best, best_dt = prev, None
        for i in range(prev, n):
            d = abs((surr_times[i] - tt).total_seconds())
            if best_dt is None or d < best_dt or (favor_ahead and d == best_dt):
                best, best_dt = i, d
        idxs.append(best)
        prev = best
    return idxs


def _surrogate_px(crop_affine, landscape_crs, grid):
    """(row_i, col_i, coverage) of each chain-window cell in the surrogate crop."""
    from pyproj import Transformer

    lon0, lat0, cell = grid["lon0"], grid["lat0"], grid["cell"]
    rr, cc = np.meshgrid(np.arange(_WIN, dtype=float),
                         np.arange(_WIN, dtype=float), indexing="ij")
    lat = lat0 + (rr + 0.5) * cell
    lon = lon0 + (cc + 0.5) * cell
    trans = Transformer.from_crs(4326, landscape_crs, always_xy=True)
    x, y = trans.transform(lon, lat)
    inv = ~crop_affine
    col_s = inv.a * x + inv.b * y + inv.c
    row_s = inv.d * x + inv.e * y + inv.f
    row_i = np.rint(row_s).astype(int)
    col_i = np.rint(col_s).astype(int)
    coverage = ((row_i >= 0) & (row_i < _CROP) &
                (col_i >= 0) & (col_i < _CROP))
    return row_i, col_i, coverage


def sample_surrogate_on_grid(mask, crop_affine, landscape_crs, grid):
    """Nearest-neighbour resample a 500x500 surrogate band onto the chain grid.

    Returns a ``(_WIN, _WIN) float32`` array in the chain window orientation
    (row 0 = south), 0 where the surrogate crop does not cover the cell.
    """
    row_i, col_i, cov = _surrogate_px(crop_affine, landscape_crs, grid)
    out = np.zeros((_WIN, _WIN), dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    out[cov] = mask[row_i[cov], col_i[cov]]
    return out


def surrogate_crop_coverage(crop_affine, landscape_crs, grid):
    """Bool ``(_WIN, _WIN)``: cells the surrogate crop actually covers."""
    return _surrogate_px(crop_affine, landscape_crs, grid)[2]


def iou(pred, truth, valid) -> float:
    """Cell-wise IoU restricted to ``valid`` (0.0 when the union is empty)."""
    pred_b = np.asarray(pred, dtype=bool)
    truth_b = np.asarray(truth, dtype=bool)
    union = ((pred_b | truth_b) & valid).sum()
    if union < _EPS:
        return 0.0
    inter = ((pred_b & truth_b) & valid).sum()
    return float(inter / union)


def real_arrival_map(truths, truth_times, chain_start_time):
    """First-burn time per cell, in hours since ``chain_start_time``; NaN if never."""
    arr = np.full((_WIN, _WIN), np.nan, dtype=np.float32)
    for k, tt in enumerate(truth_times):
        new = truths[k] & ~np.isfinite(arr)
        arr[new] = (tt - chain_start_time).total_seconds() / 3600.0
    return arr


def surrogate_arrival_map(masks_on_grid, surr_times, chain_start_time):
    """Model first-burn time per cell (hours since chain_start); NaN if never.

    Each cell's arrival is the real time of the first surrogate mask step that
    burned it (``surr_times[i]`` is the extent time of mask step ``i``). This
    matches the FAT the model records at burn time but avoids depending on the
    un-normalized FAT channel (which needs the training norm_stats to invert).
    """
    hours = np.full((_WIN, _WIN), np.nan, dtype=np.float32)
    for mask, st in zip(masks_on_grid, surr_times):
        burned = (np.asarray(mask, dtype=float) > 0.5) & ~np.isfinite(hours)
        hours[burned] = (st - chain_start_time).total_seconds() / 3600.0
    return hours


# ---------------------------------------------------------------------------
# visualization
# ---------------------------------------------------------------------------
def render_side_by_side_gif(pairs, grid, out_gif):
    """Write a per-step GIF: real (left) vs model (right), same extent.

    ``pairs`` is a list of ``(truth_time, surr_time, iou, pred, truth)`` where
    ``pred``/``truth`` are ``(_WIN, _WIN)`` 0/1 arrays aligned on ``grid``.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    from matplotlib.colors import ListedColormap
    from matplotlib.ticker import MaxNLocator

    lon0, lat0, cell = grid["lon0"], grid["lat0"], grid["cell"]
    extent = [lon0, lon0 + _WIN * cell, lat0, lat0 + _WIN * cell]
    cmap_real = ListedColormap(["#ffffff", "#e4572e"])
    cmap_model = ListedColormap(["#ffffff", "#17becf"])

    def _frame(truth_time, surr_time, iv, pred, truth, step_label):
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
        axes[0].imshow(truth.astype(float), cmap=cmap_real, vmin=0, vmax=1,
                       origin="lower", extent=extent, aspect="equal")
        axes[0].set_title(f"Real FEDS  {truth_time:%Y-%m-%d %H:%M}")
        axes[1].imshow(pred.astype(float), cmap=cmap_model, vmin=0, vmax=1,
                       origin="lower", extent=extent, aspect="equal")
        axes[1].set_title(f"Model  {surr_time:%Y-%m-%d %H:%M}")
        for ax in axes:
            ax.set_xlabel("lon")
            ax.set_ylabel("lat")
            ax.xaxis.set_major_locator(MaxNLocator(2))
            ax.yaxis.set_major_locator(MaxNLocator(2))
        fig.suptitle(f"{step_label}   IoU = {iv:.3f}")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    frames = []
    for k, (tt, st, iv, pred, truth) in enumerate(pairs):
        frames.append(_frame(tt, st, iv, pred, truth,
                             f"Spread step {k + 1}/{len(pairs)}"))
    if not frames:
        # a single placeholder frame so the output file exists
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.axis("off")
        ax.text(0.5, 0.5, "no real-data steps aligned in window",
                ha="center", va="center")
        fig.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=80)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    frames[0].save(out_gif, save_all=True, append_images=frames[1:], loop=0,
                   duration=int(max(100, 1200 / len(frames))))
    return out_gif, len(pairs)


def render_arrival_compare(real_arr, surr_arr, grid, out_png):
    """Two-panel arrival-time PNG sharing one color scale; NaN -> transparent."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from matplotlib.ticker import MaxNLocator

    lon0, lat0, cell = grid["lon0"], grid["lat0"], grid["cell"]
    extent = [lon0, lon0 + _WIN * cell, lat0, lat0 + _WIN * cell]
    finite = np.concatenate([real_arr[np.isfinite(real_arr)],
                             surr_arr[np.isfinite(surr_arr)]])
    vmax = float(finite.max()) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap = plt.get_cmap("YlOrRd")
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, arr, title in ((axes[0], real_arr, "Real fire arrival (h)"),
                           (axes[1], surr_arr, "Model fire arrival (h)")):
        ax.imshow(arr, cmap=cmap, norm=norm, origin="lower", extent=extent,
                  aspect="equal")
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_ylabel("lat")
        ax.xaxis.set_major_locator(MaxNLocator(2))
        ax.yaxis.set_major_locator(MaxNLocator(2))
    cbar = fig.colorbar(mappable, ax=axes, fraction=0.046, pad=0.04)
    cbar.set_label("fire arrival (hours since chain start)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_png


def render_combined_fire_map(run_dir, grid, real_arr, surr_arr, ign_lonlat,
                             out_png, title="", providers=("imagery", "osm"),
                             max_zoom=16, max_tiles=64):
    """One map layering (bottom to top): basemap, VIIRS fire arrival, model
    fire arrival, CALFIRE reference perimeter outline, ignition point.

    ``real_arr``/``surr_arr`` are ``(_WIN, _WIN)`` hour arrays on the chain
    window grid (row 0 = south); ``grid`` describes that window; ``ign_lonlat``
    is the ignition ``(lon, lat)``. The two arrival rasters share one
    ``vmin/vmax`` so they are directly comparable but use distinct colormaps
    (viridis for VIIRS, plasma for the model).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import geopandas as gpd
    from shapely.ops import transform as sh_transform
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from wildfire_simulator import real_fire as rf

    # CALFIRE reference perimeter -> EPSG:5070 -> EPSG:3857
    ref_gdf = gpd.read_file(os.path.join(run_dir, "reference_perimeter.shp"))
    ref_gdf = ref_gdf.to_crs(_EPSG_5070)
    ref_geom = ref_gdf.union_all().buffer(0)
    to3857 = rf._transformer(4326, 3857)
    ref_to3857 = rf._transformer(_EPSG_5070, 3857)
    ref_3857 = sh_transform(ref_to3857.transform, ref_geom)
    ign_3857 = to3857.transform(ign_lonlat[0], ign_lonlat[1])

    # chain-window corners -> EPSG:3857
    lon0, lat0, cell = grid["lon0"], grid["lat0"], grid["cell"]
    w = _WIN * cell
    corners = [(lon0, lat0), (lon0 + w, lat0), (lon0, lat0 + w), (lon0 + w, lat0 + w)]
    wpts = [to3857.transform(lon, lat) for lon, lat in corners]
    win_extent = (min(p[0] for p in wpts), max(p[0] for p in wpts),
                  min(p[1] for p in wpts), max(p[1] for p in wpts))

    # combined plot bounds = window + CALFIRE perimeter + ignition
    rb = ref_3857.bounds
    bounds = (min(win_extent[0], rb[0]), min(win_extent[2], rb[1]),
              max(win_extent[1], rb[2]), max(win_extent[3], rb[3]))
    zoom = rf._choose_zoom(bounds[2] - bounds[0], max_zoom)
    basemap = rf.fetch_basemap(providers, bounds, zoom, max_tiles)
    if basemap is None:
        print("  combined map: no basemap available; drawing on plain frame")

    # shared color scale across both arrival rasters
    finite = np.concatenate([real_arr[np.isfinite(real_arr)],
                             surr_arr[np.isfinite(surr_arr)]])
    vmax = float(finite.max()) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    norm = Normalize(vmin=0.0, vmax=vmax)
    cmap_v = plt.get_cmap("viridis")
    cmap_m = plt.get_cmap("YlOrRd")

    fig, ax = plt.subplots(figsize=(9, 8))
    if basemap is not None:
        mosaic, (bx0, bx1, by0, by1), _ = basemap
        ax.imshow(mosaic, extent=(bx0, bx1, by0, by1), interpolation="nearest",
                  zorder=0)
    ax.imshow(real_arr, cmap=cmap_v, norm=norm, extent=win_extent,
              origin="lower", aspect="equal", zorder=1, alpha=0.55)
    ax.imshow(surr_arr, cmap=cmap_m, norm=norm, extent=win_extent,
              origin="lower", aspect="equal", zorder=2, alpha=0.55)
    ax.add_patch(rf._path_patch(ref_3857, 0, 0, facecolor="none",
                                edgecolor="red", linewidth=1.6, zorder=3))
    ax.plot([ign_3857[0]], [ign_3857[1]], marker="*", markersize=18,
            color="magenta", markeredgecolor="black", zorder=4)

    pad = max(bounds[2] - bounds[0], bounds[3] - bounds[1], 1.0) * 0.05
    ax.set_xlim(bounds[0] - pad, bounds[2] + pad)
    ax.set_ylim(bounds[1] - pad, bounds[3] + pad)
    ax.set_aspect("equal")
    ax.set_title(f"{title}\nfire arrival (h since chain start):"
                 "  VIIRS vs model", fontsize=10)
    ax.tick_params(axis="both", which="both", bottom=False, labelbottom=False,
                   left=False, labelleft=False)

    divider = make_axes_locatable(ax)
    cax_v = divider.append_axes("right", size="4%", pad=0.10)
    cax_m = divider.append_axes("right", size="4%", pad=0.36)
    sm_v = ScalarMappable(norm=norm, cmap=cmap_v)
    sm_v.set_array([])
    sm_m = ScalarMappable(norm=norm, cmap=cmap_m)
    sm_m.set_array([])
    cb_v = fig.colorbar(sm_v, cax=cax_v)
    cb_v.set_label("VIIRS arrival (h)", fontsize=7)
    cb_v.ax.tick_params(labelsize=6)
    cb_m = fig.colorbar(sm_m, cax=cax_m)
    cb_m.set_label("Model arrival (h)", fontsize=7)
    cb_m.ax.tick_params(labelsize=6)

    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out_png
