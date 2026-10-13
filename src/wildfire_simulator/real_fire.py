"""real_fire - time-varying-wind sample builder, autoregressive step, renderer.

This module is the model-side half of the real-fire inference pipeline
(``scripts/infer_real.py``). It deliberately stays decoupled from the
farsite-docker tools: the driver (infer_real) computes the crop cell size,
ingests + resamples the landscape, runs WindNinja, and hands the raw arrays
here.

The three entry points mirror existing conventions exactly:

* ``build_sample`` reproduces ``WildfireDataset._load_raw_channels`` channel
  order (13 channels, crop 500x500 around the ignition) so the sample is
  indistinguishable from a training sample apart from the synthetic fire seed.
* ``fire_burn_step_wind`` is ``simulators.fire_burn_step`` with per-step wind
  injection on channels 10/11 (the training model only ever saw one static wind
  field; this is deliberately out-of-distribution).
* ``render`` draws the rollout over an XYZ basemap (imagery then osm), the real
  reference perimeter and the ignition marker, producing a multi-frame GIF plus
  a single arrival-time map.

The tile math, basemap fetch, and the offset-preserving polygon path
construction mirror ``farsite-docker/tools/compare_perimeter.py`` so both
renderers look consistent.
"""

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

_MERCATOR_EXTENT = 40075016.685578488   # EPSG:3857 world width, m
_TILE_SIZE = 256
_TILE_URLS = {
    "imagery": ("https://server.arcgisonline.com/ArcGIS/rest/services/"
                "World_Imagery/MapServer/tile/{z}/{y}/{x}"),
    "osm": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
}
_HEADERS = {"User-Agent": "wildfire-surrogate-infer/0.1"}
_CROP = 500
_HALF = 250


def build_sample(resampled_landscape, wind_u, wind_v, cy, cx,
                 foliar_moisture, dt=0.0, max_val_fat=1.0, half=_HALF):
    """Build the raw ``(13, 500, 500)`` float32 input crop.

    Mirrors ``WildfireDataset._load_raw_channels`` exactly:
    ``[fire_mask, fire_arr, elev, slope, aspect, fuel, cc, ch, cbh, cbd,
    wind_u, wind_v, foliar_moisture]``.

    Parameters
    ----------
    resampled_landscape : ndarray, shape (8, H, W)
        The resampled landscape bands (elevation, slope, aspect, fuel,
        canopy_cover, stand_height, canopy_base_height, canopy_bulk_density),
        best read via ``rasterio`` so band order matches band 1..8.
    wind_u, wind_v : ndarray, shape (H, W)
        First-hour wind U/V components (m/s convention as in the dataset).
    cy, cx : int
        Fire ignition pixel (row, col) in the resampled landscape.
    foliar_moisture : float
        Raw foliar-moisture scalar (training units, ~100-130).
    dt : float
        Normalized time step; the synthetic seed FAT is ``dt * max_val_fat`` so
        the normalized seed arrival equals ``dt``.
    max_val_fat : float
        ``norm_stats`` max for channel 1 (arrival time).
    """
    resampled_landscape = np.asarray(resampled_landscape, dtype=np.float32)
    if resampled_landscape.ndim != 3 or resampled_landscape.shape[0] != 8:
        raise ValueError(
            "resampled_landscape must be (8, H, W) 8-band float32, "
            f"got shape {resampled_landscape.shape}"
        )
    cy, cx = int(cy), int(cx)
    cy0, cy1 = cy - half, cy + half
    cx0, cx1 = cx - half, cx + half

    crops = [
        np.asarray(resampled_landscape[i][cy0:cy1, cx0:cx1], dtype=np.float32)
        for i in range(8)
    ]

    fire_mask = np.zeros((_CROP, _CROP), dtype=np.float32)
    fire_mask[250, 250] = 1.0
    fire_arr = np.zeros((_CROP, _CROP), dtype=np.float32)
    fire_arr[250, 250] = dt * max_val_fat

    wu = np.asarray(wind_u[cy0:cy1, cx0:cx1], dtype=np.float32)
    wv = np.asarray(wind_v[cy0:cy1, cx0:cx1], dtype=np.float32)
    fm = np.full((_CROP, _CROP), foliar_moisture, dtype=np.float32)

    return np.stack([fire_mask, fire_arr, *crops, wu, wv, fm], axis=0)


def fire_burn_step_wind(t, model, inputs_norm, dt, wind_u_norm, wind_v_norm):
    """One autoregressive step with per-step wind injected on channels 10/11.

    Identical to ``simulators.fire_burn_step`` (padding 6, threshold 0.5,
    ``FAT = t + dt``) except the normalized wind U/V channels are overwritten
    before the time channel is appended, so the time-varying wind reaches the
    network every step. ``inputs_norm`` is ``(1, 13, 500, 500)`` normalized;
    ``wind_u_norm``/``wind_v_norm`` are ``(500, 500)`` normalized grids.
    Returns ``(1, 13, 500, 500)`` (mirroring the ``fire_burn_step`` contract).
    """
    inputs = copy.deepcopy(inputs_norm)
    wind = torch.stack([wind_u_norm, wind_v_norm], dim=0).unsqueeze(0)
    inputs[:, 10:12] = wind
    # time channel
    inputs = torch.cat(
        (inputs, torch.full((1, 1, _CROP, _CROP), t, device=inputs.device)),
        dim=1,
    )  # (1, 14, 500, 500)
    inputs = F.pad(inputs, (6, 6, 6, 6, 0, 0), mode="constant", value=0)
    with torch.no_grad():
        pred = model(inputs)
        if isinstance(pred, (list, tuple)):
            pred = pred[0]
        pred_mask = (pred[0, 0].detach() > 0.5).float()
        inputs[0, 0, 6:-6, 6:-6] = pred_mask[6:-6, 6:-6]
        if dt is not None:
            current_fat = inputs[0, 1, 6:-6, 6:-6]
            newly_burned = (pred_mask[6:-6, 6:-6] == 1) & (current_fat == 0)
            inputs[0, 1, 6:-6, 6:-6][newly_burned] = t + dt
    return inputs[:, :13, 6:-6, 6:-6]


# ---------------------------------------------------------------------------
# raw numpy outputs + plain (pre-overlay) visualizations
# ---------------------------------------------------------------------------
CHANNEL_NAMES = [
    "fire_mask", "fire_arr", "elevation", "slope", "aspect", "fuel",
    "canopy_cover", "stand_height", "canopy_base_height", "canopy_bulk_density",
    "wind_u", "wind_v", "foliar_moisture",
]


def save_input_channels(sample, run_dir, npz_name="input_channels.npz",
                        png_name="input_channels.png"):
    """Save the raw (13, 500, 500) input channels to an .npz and a plain PNG.

    The .npz carries every channel under its channel name plus a ``sample``
    key with the full stacked array, so the user can inspect each input the
    model saw (before any basemap overlay). The PNG is a 2x5 grid of the
    interpretable static channels, the wind U/V quiver, and foliar moisture.
    """
    import numpy as _np
    from pathlib import Path

    sample = _np.asarray(sample, dtype=_np.float32)
    out = Path(run_dir)
    npz = out / npz_name
    data = {name: sample[i] for i, name in enumerate(CHANNEL_NAMES)}
    data["sample"] = sample
    _np.savez(npz, **data)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    channel_info = [
        (2, "Elevation", "terrain", "m"), (3, "Slope", "YlOrBr", "\u00b0"),
        (4, "Aspect", "hsv", "\u00b0"), (5, "Fuel (FBFM40)", "Set3", ""),
        (6, "Canopy Cover", "Greens", "%"), (7, "Stand Height", "Greens", "m"),
        (8, "Canopy Base Ht", "Greens", "m"),
        (9, "Canopy Bulk Dens", "Greens", "kg/m\u00b3"),
    ]
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    for i, (ch, name, cmap, unit) in enumerate(channel_info):
        row, col = divmod(i, 5)
        ax = axes[row, col]
        im = ax.imshow(sample[ch], cmap=cmap, aspect="equal")
        ax.set_title(name, fontsize=10)
        ax.axis("off")
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if unit:
            cbar.set_label(unit, fontsize=8)

    # wind quiver (channels 10/11)
    ax_wind = axes[1, 3]
    u, v = sample[10], sample[11]
    speed = _np.sqrt(u ** 2 + v ** 2)
    mean_speed = float(speed.mean())
    mean_dir = (_np.degrees(_np.arctan2(-float(u.mean()), -float(v.mean())))
                + 360) % 360
    step = 25
    X, Y = _np.meshgrid(_np.arange(0, 500, step), _np.arange(0, 500, step))
    ax_wind.quiver(X, Y, u[::step, ::step], v[::step, ::step],
                   scale=max(mean_speed, 1e-6) * 15, color="navy", alpha=0.7)
    ax_wind.set_xlim(0, 500)
    ax_wind.set_ylim(500, 0)
    ax_wind.set_aspect("equal")
    ax_wind.set_title(f"Wind U/V\n{mean_speed:.0f} avg @ {mean_dir:.0f}\u00b0",
                      fontsize=10)
    ax_wind.axis("off")

    ax_fm = axes[1, 4]
    ax_fm.text(0.5, 0.5, f"Foliar\nMoisture\n\n{float(sample[12, 0, 0]):.0f}%",
               ha="center", va="center", fontsize=14, transform=ax_fm.transAxes)
    ax_fm.axis("off")

    fig.suptitle("Inference Input Channels (raw)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out / png_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  input channels: {npz} (+{out / png_name})")
    return npz


def save_arrival_map(fat, run_dir, npy_name="arrival_map.npy",
                     png_name="arrival_time.png"):
    """Save the raw final fire-arrival-time map as .npy and a plain PNG."""
    import numpy as _np
    from pathlib import Path

    out = Path(run_dir)
    arr = _np.asarray(fat, dtype=_np.float32)
    np_npy = out / npy_name
    _np.save(np_npy, arr)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis = _np.where(arr <= 0, _np.nan, arr)
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(vis, cmap="YlOrRd", aspect="equal")
    ax.set_title("Fire arrival time (raw)")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out / png_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  arrival map (raw): {np_npy} (+{out / png_name})")
    return np_npy


# ---------------------------------------------------------------------------
# XYZ basemap tile math (mirrors compare_perimeter.py)
# ---------------------------------------------------------------------------
def _web_mercator_to_lonlat(x, y):
    lon = x / _MERCATOR_EXTENT * 360.0
    lat = math.degrees(2.0 * math.atan(
        math.exp(y * math.pi / (_MERCATOR_EXTENT / 2.0))) - math.pi / 2.0)
    return lon, lat


def _tiles_for(bounds, zoom):
    n = 1 << zoom
    minx, miny, maxx, maxy = bounds
    xs, ys = [], []
    for x, y in ((minx, maxy), (maxx, maxy), (maxx, miny), (minx, miny)):
        lon, lat = _web_mercator_to_lonlat(x, y)
        tx = int(math.floor((lon + 180.0) / 360.0 * n))
        ty = int(math.floor(
            (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n))
        xs.append(min(max(tx, 0), n - 1))
        ys.append(min(max(ty, 0), n - 1))
    return min(xs), max(xs), min(ys), max(ys)


def _choose_zoom(w_m, max_zoom):
    if w_m <= 0:
        return 1
    zoom = int(math.floor(math.log2(_MERCATOR_EXTENT * 2.0 / w_m)))
    zoom = max(0, min(zoom, max_zoom))
    return max(zoom, 1)


def fetch_basemap(providers, bounds3857, zoom, max_tiles=64):
    """First provider (in order) that fully mosaics, else None."""
    import requests
    from io import BytesIO
    from PIL import Image

    n = 1 << zoom
    for provider in providers:
        if provider not in _TILE_URLS:
            continue
        x0, x1, y0, y1 = _tiles_for(bounds3857, zoom)
        if (x1 - x0 + 1) * (y1 - y0 + 1) > max_tiles:
            continue
        url_t = _TILE_URLS[provider]
        try:
            cols = []
            for yt in range(y0, y1 + 1):
                row = [
                    np.array(
                        Image.open(BytesIO(requests.get(
                            url_t.format(z=zoom, x=xt, y=yt),
                            headers=_HEADERS, timeout=10).content)).convert("RGB")
                    )
                    for xt in range(x0, x1 + 1)
                ]
                cols.append(np.concatenate(row, axis=1))
            mosaic = np.concatenate(cols, axis=0)
        except Exception:
            continue
        scale = _MERCATOR_EXTENT / (n * _TILE_SIZE)
        x_min = -_MERCATOR_EXTENT / 2.0 + x0 * _TILE_SIZE * scale
        x_max = -_MERCATOR_EXTENT / 2.0 + (x1 + 1) * _TILE_SIZE * scale
        y_max = _MERCATOR_EXTENT / 2.0 - y0 * _TILE_SIZE * scale
        y_min = _MERCATOR_EXTENT / 2.0 - (y1 + 1) * _TILE_SIZE * scale
        return mosaic, (x_min, x_max, y_min, y_max), provider
    return None


def _reproject_3857(arr, src_transform, src_crs, dst_transform, dst_shape):
    """Reproject a single-band raster from ``src_crs`` to EPSG:3857 (nearest)."""
    from rasterio.crs import CRS
    from rasterio.warp import reproject, Resampling

    out = np.full(dst_shape, np.nan, dtype=np.float32)
    src_crs = CRS.from_user_input(src_crs)
    reproject(
        arr.astype(np.float32), out,
        src_transform=src_transform, src_crs=src_crs,
        dst_transform=dst_transform, dst_crs=CRS.from_epsg(3857),
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.nearest,
    )
    return out


def _mask_rgba(mask_3857, color, alpha):
    """RGBA overlay from a 0/1 mask, NaN nodata -> transparent."""
    rgba = np.zeros((*mask_3857.shape, 4), dtype=np.uint8)
    rgb = np.array(color, dtype=np.uint8)
    filled = (mask_3857 > 0.5) & (~np.isnan(mask_3857))
    rgba[filled] = np.array([*rgb, int(alpha * 255)], dtype=np.uint8)
    return rgba


def _signed_area(ring):
    s = 0.0
    n = len(ring)
    for i in range(n):
        x1, y1 = ring[i]
        x2, y2 = ring[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return 0.5 * s


def _path_for(geom, ox, oy):
    """PathPatch path: outer rings CCW, holes CW, shifted by (ox, oy)."""
    import numpy as _np
    from matplotlib.path import Path as MplPath

    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    verts, codes = [], []
    for poly in polys:
        ext = list(poly.exterior.coords)[:-1]
        rings = [(ext, True)] + [
            (list(h.coords)[:-1], False) for h in poly.interiors]
        for ring, outer in rings:
            if len(ring) < 3:
                continue
            if (_signed_area(ring) > 0) != outer:
                ring = ring[::-1]
            verts.extend((x - ox, y - oy) for x, y in ring)
            codes += [MplPath.MOVETO] + [MplPath.LINETO] * (len(ring) - 1)
    return MplPath(_np.asarray(verts, dtype=float), codes)


def _figure_frame(ax, basemap, ox, oy, ref_geom, ign_xy, caption, title,
                  extent):
    """Common decorations for one frame/plot: basemap, ref perimeter, marker."""
    if basemap is not None:
        mosaic, (x0, x1, y0, y1), _ = basemap
        ax.imshow(mosaic,
                  extent=(x0 - ox, x1 - ox, y0 - oy, y1 - oy),
                  interpolation="nearest", zorder=0)
    ax.add_patch(_path_patch(ref_geom, ox, oy, facecolor="#d9d9d9",
                             edgecolor="red", linewidth=1.2, zorder=1))
    if ign_xy is not None:
        ax.plot(ign_xy[0] - ox, ign_xy[1] - oy, marker="*", markersize=18,
                color="magenta", markeredgecolor="black", zorder=4)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")
    head = title if title else ""
    ax.set_title(f"{head}\n{caption}".strip())
    ax.tick_params(axis="both", which="both", bottom=False, labelbottom=False,
                   left=False, labelleft=False)
    return ax


def _path_patch(geom, ox, oy, **kw):
    from matplotlib.patches import PathPatch
    return PathPatch(_path_for(geom, ox, oy), fill=True, **kw)


def _transformer(src_crs, dst_epsg):
    from pyproj import CRS, Transformer
    src = CRS.from_user_input(src_crs)
    return Transformer.from_crs(src, f"EPSG:{dst_epsg}", always_xy=True)


def render(run_dir, history, times, crop_affine, landscape_crs,
           ref_geom, ref_crs, ignition_xy, start, end,
           dt, max_t, min_val_fat, max_val_fat,
           out_gif, out_png, providers=("imagery", "osm"),
           max_zoom=16, max_tiles=64):
    """Render the rollout to ``spread.gif`` + ``arrival_map.png``.

    history : list[ndarray (13,500,500) raw] - per-step rollout states; step i
        was produced at normalized loop time ``times[i]`` (fire extent at
        ``times[i] + dt``).
    times : list[float] - normalized loop times used at each step.
    crop_affine : rasterio Affine of the 500x500 crop window in ``landscape_crs``.
    ref_geom : shapely geometry of the real final perimeter in ``ref_crs``.
    ignition_xy : (x, y) ignition anchor in ``ref_crs``.
    start, end : python ``datetime`` (UTC) bounding the real-time window.
    dt, max_t : normalized rollout parameters.
    min_val_fat, max_val_fat : norm_stats channel-1 min/max.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from io import BytesIO
    from PIL import Image
    from rasterio.transform import array_bounds, from_bounds
    from shapely.ops import transform as sh_transform
    from pyproj import Transformer
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    if len(history) == 0:
        raise ValueError("render: empty history")

    to3857 = _transformer(landscape_crs, 3857)
    xmin, ymin, xmax, ymax = array_bounds(_CROP, _CROP, crop_affine)
    corners = [(xmin, ymin), (xmin, ymax), (xmax, ymin), (xmax, ymax)]
    pts = [to3857.transform(cx, cy) for cx, cy in corners]
    mminx = min(p[0] for p in pts)
    mmaxx = max(p[0] for p in pts)
    mminy = min(p[1] for p in pts)
    mmaxy = max(p[1] for p in pts)
    dst_shape = (_CROP, _CROP)
    dst_transform = from_bounds(mminx, mminy, mmaxx, mmaxy,
                                dst_shape[1], dst_shape[0])

    to3857_ref = _transformer(ref_crs, 3857)
    ref_3857 = sh_transform(to3857_ref.transform, ref_geom)
    ign_3857 = None if ignition_xy is None else to3857_ref.transform(
        ignition_xy[0], ignition_xy[1])

    rb = ref_3857.bounds
    bounds3857 = (min(mminx, rb[0]), min(mminy, rb[1]),
                  max(mmaxx, rb[2]), max(mmaxy, rb[3]))
    w_m = bounds3857[2] - bounds3857[0]
    zoom = _choose_zoom(w_m, max_zoom)
    basemap = fetch_basemap(providers, bounds3857, zoom, max_tiles)
    if basemap is None:
        print("  render: no basemap provider available; drawing on plain frame")

    ox, oy = bounds3857[0], bounds3857[1]
    pad = max(bounds3857[2] - bounds3857[0],
              bounds3857[3] - bounds3857[1], 1.0) * 0.05
    extent = (bounds3857[0] - ox - pad, bounds3857[2] - ox + pad,
              bounds3857[1] - oy - pad, bounds3857[3] - oy + pad)
    crop_extent = (mminx - ox, mmaxx - ox, mminy - oy, mmaxy - oy)

    if len(times) != len(history):
        raise ValueError("render: times and history lengths differ")
    total = (end - start).total_seconds()

    frames = []
    for i, state in enumerate(history):
        mask = np.asarray(state[0], dtype=np.float32)
        extent_norm = times[i] + dt
        real_t = start + timedelta_frac((extent_norm / max_t) * total)
        fig, ax = plt.subplots(figsize=(8.0, 8.0))
        # draw burned mask overlay
        burned = _reproject_3857(mask, crop_affine, landscape_crs,
                                 dst_transform, dst_shape)
        ax.imshow(_mask_rgba(burned, (0, 255, 255), 0.5),
                  extent=crop_extent, interpolation="nearest", zorder=2)
        _figure_frame(ax, basemap, ox, oy, ref_3857, ign_3857,
                      caption=f"{real_t:%Y-%m-%d %H:%M} UTC",
                      title=f"Spread step {i + 1}/{len(history)}",
                      extent=extent)
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGB"))
    if out_gif:
        frames[0].save(out_gif, save_all=True, append_images=frames[1:],
                       loop=0, duration=int(max(100, 1200 / len(frames))))
        print(f"  gif: {out_gif} ({len(frames)} frames)")

    # ---- arrival map -------------------------------------------------------
    final = np.asarray(history[-1], dtype=np.float32)
    mask = final[0]
    fat = final[1].copy()
    fat[mask <= 0.5] = np.nan
    fat_norm = (fat - min_val_fat) / max(max_val_fat - min_val_fat, 1e-8)
    fat_3857 = _reproject_3857(fat_norm, crop_affine, landscape_crs,
                               dst_transform, dst_shape)
    fig, ax = plt.subplots(figsize=(8.0, 8.0))
    cmap = plt.get_cmap("inferno")
    mappable = ScalarMappable(norm=Normalize(vmin=0.0, vmax=1.0), cmap=cmap)
    mappable.set_array([])
    colored = (cmap(np.clip(fat_3857, 0.0, 1.0)) if np.isfinite(fat_3857).any()
               else np.zeros((*dst_shape, 4), dtype=np.uint8))
    # never-burned cells -> transparent (alpha channel, element-wise)
    finite = np.isfinite(fat_3857)
    colored = np.asarray(colored, dtype=np.float64)
    colored[..., 3] = np.where(finite, colored[..., 3], 0.0)
    ax.imshow(colored, extent=crop_extent, interpolation="nearest", zorder=2)
    _figure_frame(ax, basemap, ox, oy, ref_3857, ign_3857,
                  caption=f"final fire: {end:%Y-%m-%d %H:%M} UTC",
                  title="Predicted fire arrival time",
                  extent=extent)
    cbar = fig.colorbar(mappable, ax=ax, fraction=0.046, pad=0.04)
    ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([
        _fmt_real(start + timedelta_frac((v / max_t) * total)) for v in ticks
    ])
    cbar.set_label("fire arrival (UTC)")
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"  arrival map: {out_png}")


def timedelta_frac(seconds):
    from datetime import timedelta
    return timedelta(seconds=float(seconds))


def _fmt_real(dt):
    """Format a computed real (UTC) instant for colorbar tick labels."""
    return dt.strftime("%Y-%m-%d %H:%M")
