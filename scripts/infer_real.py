#!/usr/bin/env python3
"""infer_real - end-to-end real-fire inference for the WildfireSurrogateModel.

Runs the surrogate on a real CAL FIRE fire using only the ignition point + a
UTC time range + a user foliar moisture, then autoregressively predicts spread
and validates against the real final perimeter. Inputs come from one TOML
config; all generated results land in ``<run_dir>/results/``: ``result.json``
(IoU + areas), ``spread.gif``, ``arrival_map.png``, the raw arrays
(``input_channels.npz``, ``arrival_map.npy``, ``rollout.npz``), the predicted
perimeter (``simulated_perimeter.shp``), the resampling note
(``resample.json``), and plain PNGs (``input_channels.png``, ``arrival_time.png``).

The farsite-docker ``tools/`` provide the CAL FIRE ingestion, LANDFIRE
landscape ingest, and HRRR->WindNinja wind grids; this driver wires them to the
surrogate. It does not run ``runfarsite``.

    python scripts/infer_real.py --config configs/infer.example.toml
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import sys
import tomllib
from datetime import datetime, timedelta
from pathlib import Path

# The repo's stray PROJ_LIB/PROJ_DATA (pointing at bundled Windows GDAL data)
# must be neutralized BEFORE any pyproj import, exactly as the farsite tools do.
for _proj_k in ("PROJ_LIB", "PROJ_DATA"):
    os.environ.pop(_proj_k, None)

import geopandas as gpd
import numpy as np
import rasterio
import torch
from rasterio.transform import from_origin, rowcol

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "farsite-docker" / "tools"))

from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.checkpoints import init_from_checkpoint
from wildfire_simulator import real_fire
from runroot_to_atm import scan_frames, read_asc
from hrrr_to_wxs import parse_utc, resolve_timezone

_ENV_BIN = Path.home() / "miniconda3/envs/wildfire_env_20261012/bin"
_DEFAULT_TOOLS_PYTHON = _ENV_BIN / "python"
_DEFAULT_WINDNINJA = _ENV_BIN / "WindNinja_cli"
_CROP = 500
_HALF = 250
_EPSG_5070 = "EPSG:5070"
_SQ_FT_PER_ACRE = 43560.0
_ACRE_M2 = 4046.8564224

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(2)


def _eval_fraction(s):
    parts = str(s).strip().split("/")
    if len(parts) == 2:
        return float(parts[0]) / float(parts[1])
    return float(s)


def _slug(name):
    return str(name).lower().replace(" ", "-")


def _fmt_dt(dt):
    return dt.strftime("%Y-%m-%dT%H:%M") + "Z"


def _scalar(v):
    """Render a config value the way WindNinja's config parser expects it."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return str(int(v)) if v.is_integer() else repr(v)
    die(f"windninja option must be scalar, got {type(v).__name__}")


def _chdir(cwd):
    os.chdir(cwd)


def run_subprocess(argv, cwd, env=None):
    r = subprocess.run(argv, cwd=cwd, env=env)
    if r.returncode != 0:
        die(f"command failed (rc={r.returncode}): {' '.join(argv)}")
    return r


def _cache_ok(marker_path, params, required_files, allow_orphan=False):
    """True if the cached outputs are reusable for ``params``.

    Requires every ``required_files`` to exist. A marker matching ``params``
    gives a strict hit; when ``allow_orphan`` is set and no marker exists, the
    existing outputs are reused (legacy runs that completed before the cache
    markers were introduced). On such an orphan first use the caller writes the
    marker so a later config change invalidates the cache. A changed config
    (different fire, bbox, cell size, source-landscape mtime) therefore
    invalidates the cache on subsequent runs.
    """
    marker_path = Path(marker_path)
    if not all(Path(p).is_file() for p in required_files):
        return False
    if not marker_path.is_file():
        return allow_orphan
    try:
        return json.loads(marker_path.read_text()) == params
    except Exception:
        return False


def _windninja_orphan_ok(wind_root, params):
    """Validate a no-marker (orphan) set of WindNinja pairs for reuse.

    Accepts them only if the grid cell size matches the requested ``cell_m``
    and every per-hour stamp falls within the simulation window. This lets a
    prior WindNinja run that predates the cache markers be reused instead of
    re-downloading PASTCAST and re-solving.
    """
    frames = scan_frames(wind_root)
    if not frames:
        return False
    try:
        vel, _ = next(iter(frames.values()))
        meta, _ = read_asc(vel)
        if abs(float(meta["cellsize"]) - params["cell_m"]) > 0.01:
            return False
    except Exception:
        return False
    try:
        ls = datetime.fromisoformat(params["local_start"])
        le = datetime.fromisoformat(params["local_end"])
    except Exception:
        return False
    for stamp in frames:
        y, mo, d, h = stamp
        try:
            local = datetime(y, mo, d, h)
        except Exception:
            return False
        if not (ls <= local <= le):
            return False
    return True


def _windninja_cached(wind_root, params):
    """True if WindNinja output is reusable (skip the PASTCAST re-download and
    the solver): either a marker matching ``params`` exists AND the pairs are
    present, or (orphan) the pairs are present and validate for ``params``."""
    if not Path(wind_root).is_dir() or not scan_frames(wind_root):
        return False
    marker = Path(wind_root) / "windninja.cache.json"
    if not marker.is_file():
        return _windninja_orphan_ok(wind_root, params)
    try:
        return json.loads(marker.read_text()) == params
    except Exception:
        return False


# ---------------------------------------------------------------------------
# config
# ---------------------------------------------------------------------------
def load_config(path):
    path = Path(path)
    try:
        with open(path, "rb") as f:
            cfg = tomllib.load(f)
    except FileNotFoundError:
        die(f"--config not found: {path}")
    except tomllib.TOMLDecodeError as e:
        die(f"cannot parse {path}: {e}")

    # paths defaults
    paths = cfg.setdefault("paths", {})
    paths.setdefault("tools_python", str(_DEFAULT_TOOLS_PYTHON))
    paths.setdefault(
        "farsite_repo", str(Path(__file__).parent.parent.parent / "farsite-docker"))
    paths.setdefault("run_dir", None)

    # model defaults
    model = cfg.setdefault("model", {})
    model.setdefault("cell_size_m", None)
    model.setdefault("margin", 0.10)

    # simulation defaults
    sim = cfg.setdefault("simulation", {})
    sim.setdefault("max_t", 1.0)
    sim.setdefault("row_timezone", None)
    sim.setdefault("dt", "1/48")

    # landscape defaults
    land = cfg.setdefault("landscape", {})
    land.setdefault("version", "2024")
    land.setdefault("fuel_model", "fbfm40")

    # wind defaults
    wind = cfg.setdefault("wind", {})
    if "windninja_command" not in wind:
        wind["windninja_command"] = str(_DEFAULT_WINDNINJA)
    wind.setdefault("threads", 8)
    wind.setdefault("windninja_options", {})

    return cfg


def _resolve_path(base_dir, p):
    if p is None:
        return None
    path = Path(p)
    return path if path.is_absolute() else base_dir / path


# ---------------------------------------------------------------------------
# step 1 - resolve the fire
# ---------------------------------------------------------------------------
def compute_run_dir(cfg, slug):
    run_dir = cfg["paths"].get("run_dir")
    if run_dir:
        return Path(run_dir)
    fire = cfg["fire"]
    year = fire.get("year") or "any"
    base = Path(cfg["paths"]["farsite_repo"]) / "FireBehaviorModels" / "SampleData"
    return base / f"{year}_{slug}"


def _fire_selection(fire):
    """The identifying params a cached fire record was resolved from."""
    return {
        "name": fire.get("name"), "year": fire.get("year"),
        "index": fire.get("index"), "inc": fire.get("inc"),
        "lat": fire.get("lat"), "lon": fire.get("lon"),
    }


def resolve_fire(cfg, run_dir):
    fire = cfg["fire"]
    marker = run_dir / "fire.cache.json"
    required = [run_dir / "fire.json", run_dir / "reference_perimeter.shp",
                run_dir / "ignition.shp"]
    selection = _fire_selection(fire)
    if _cache_ok(marker, selection, required, allow_orphan=True):
        with open(run_dir / "fire.json") as f:
            fj = json.load(f)
        # Legacy/orphan reuse: confirm the cached record is the selected fire.
        if (str(fj.get("name")).upper() == str(fire["name"]).upper()
                and str(fj.get("year")) == str(fire.get("year"))):
            if not marker.is_file():
                marker.write_text(json.dumps(selection, indent=2) + "\n")
            print(f"reusing cached fire under {run_dir}")
            return fj
        # else fall through: the existing record is for a different fire.

    tools_py = cfg["paths"]["tools_python"]
    farsite_repo = cfg["paths"]["farsite_repo"]
    argv = [tools_py, "tools/calfire_ignition.py",
            "--fire-name", str(fire["name"]), "--year", str(fire.get("year")),
            "--crs", _EPSG_5070, "--out-dir", str(run_dir)]
    if fire.get("index") is not None:
        argv += ["--index", str(fire["index"])]
    elif fire.get("inc"):
        argv += ["--inc", str(fire["inc"])]
    elif fire.get("lat") is not None and fire.get("lon") is not None:
        argv += ["--lat", str(fire["lat"]), "--lon", str(fire["lon"])]
    run_subprocess(argv, cwd=farsite_repo)
    marker.write_text(json.dumps(_fire_selection(fire), indent=2) + "\n")

    with open(run_dir / "fire.json") as f:
        fj = json.load(f)
    print(f"resolved fire: {fj['name']} ({fj['year']})  alarm={fj['alarm_date']} "
          f"cont={fj['cont_date']}  ignition=({fj['lat']:.6f}, {fj['lon']:.6f})")
    return fj


# ---------------------------------------------------------------------------
# step 2 - crop cell size
# ---------------------------------------------------------------------------
def _perimeter_vertices(geom):
    """Yield (x, y) vertices of a polygon/multipolygon geometry."""
    if geom is None or geom.is_empty:
        return
    polys = geom.geoms if geom.geom_type == "MultiPolygon" else [geom]
    for poly in polys:
        if poly.geom_type != "Polygon":
            continue
        for x, y in poly.exterior.coords:
            yield x, y


def compute_cell_m(cfg, fj, run_dir, results_dir):
    from pyproj import Transformer
    to5070 = Transformer.from_crs(4326, _EPSG_5070, always_xy=True)
    ig = to5070.transform(float(fj["lon"]), float(fj["lat"]))
    perim = gpd.read_file(run_dir / "reference_perimeter.shp")
    perim = perim.to_crs(_EPSG_5070)
    ig_x, ig_y = ig

    max_radius = 0.0
    for geom in perim.geometry:
        for x, y in _perimeter_vertices(geom):
            max_radius = max(max_radius, math.hypot(x - ig_x, y - ig_y))

    margin = float(cfg["model"].get("margin", 0.10))
    cell = cfg["model"].get("cell_size_m")
    if cell is not None:
        cell_m = float(cell)
        source = "config model.cell_size_m"
    else:
        cell_m = max(30.0,
                     math.ceil(2.0 * max_radius * (1.0 + margin) / _CROP))
        source = "computed"
        if cell_m > 30:
            print(f"\n*** RESAMPLING: LANDFIRE 30 m -> {cell_m:.0f} m "
                  f"(500-cell crop spans {cell_m * _CROP / 1000.0:.1f} km) ***")
    crop_span_km = cell_m * _CROP / 1000.0
    if not math.isfinite(crop_span_km) or crop_span_km >= 200.0:
        die(f"crop span {crop_span_km:.1f} km too large (> ~200 km); fire too "
            f"large or ignition far from the perimeter (max radius {max_radius:.0f} m)")
    payload = {
        "native_cell_m": 30,
        "cell_m": float(cell_m),
        "crop_span_km": float(crop_span_km),
        "max_radius_m": float(max_radius),
        "source": source,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "resample.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"cell size: {cell_m:.1f} m ({source}); "
          f"crop span {crop_span_km:.1f} km")
    return cell_m, ig_x, ig_y, max_radius


# ---------------------------------------------------------------------------
# step 3 - ingest + resample the landscape
# ---------------------------------------------------------------------------
def _bbox_around(cfg, fj, cell_m, margin=0.10):
    from pyproj import Transformer
    to5070 = Transformer.from_crs(4326, _EPSG_5070, always_xy=True)
    to4326 = Transformer.from_crs(_EPSG_5070, 4326, always_xy=True)
    ig = to5070.transform(float(fj["lon"]), float(fj["lat"]))
    R = (_CROP / 2.0) * cell_m * (1.0 + margin)
    box5070 = (ig[0] - R, ig[1] - R, ig[0] + R, ig[1] + R)
    corners = [
        to4326.transform(box5070[0], box5070[1]),
        to4326.transform(box5070[0], box5070[3]),
        to4326.transform(box5070[2], box5070[1]),
        to4326.transform(box5070[2], box5070[3]),
    ]
    lons = [c[0] for c in corners]
    lats = [c[1] for c in corners]
    return f"{min(lons):.8f} {min(lats):.8f} {max(lons):.8f} {max(lats):.8f}"


def ingest_landscape(cfg, run_dir, bbox):
    land = cfg["landscape"]
    marker = run_dir / "landscape.cache.json"
    params = {
        "bbox": bbox, "email": land.get("email"),
        "version": land.get("version", "2024"),
        "fuel_model": land.get("fuel_model", "fbfm40"),
    }
    required = [run_dir / "landscape.tif", run_dir / "dem-30m.tif"]
    if _cache_ok(marker, params, required, allow_orphan=True):
        if not marker.is_file():
            marker.write_text(json.dumps(params, indent=2) + "\n")
        print(f"reusing cached landscape: {run_dir / 'landscape.tif'}")
        return

    tools_py = cfg["paths"]["tools_python"]
    farsite_repo = cfg["paths"]["farsite_repo"]
    argv = [tools_py, "tools/ingest_landscape.py",
            "--bbox", bbox,
            "--email", str(land["email"]),
            "--version", str(land.get("version", "2024")),
            "--fuel-model", str(land.get("fuel_model", "fbfm40")),
            "--resolution", "30",
            "--out", str(run_dir / "landscape"),
            "--dem-out", str(run_dir / "dem-30m.tif")]
    run_subprocess(argv, cwd=farsite_repo)
    marker.write_text(json.dumps(params, indent=2) + "\n")


def resample_landscape(cfg, run_dir, cell_m):
    from rasterio.warp import reproject, Resampling
    src_path = run_dir / "landscape.tif"
    out = run_dir / "landscape_resampled.tif"
    dem_out = run_dir / "dem-resampled.tif"
    marker = run_dir / "resample.cache.json"
    params = {"cell_m": float(cell_m),
              "source_mtime": src_path.stat().st_mtime}
    if _cache_ok(marker, params, [out, dem_out], allow_orphan=True):
        with rasterio.open(src_path) as src:
            src_crs = src.crs
        with rasterio.open(out) as src:
            # Validate the cached file really is at the requested cell size
            # (an orphan cache could be from a different cell size).
            if abs(src.res[0] - cell_m) < 0.01:
                if not marker.is_file():
                    marker.write_text(json.dumps(params, indent=2) + "\n")
                print(f"reusing cached resampled landscape: {out.name}")
                return out, dem_out, src_crs, src.transform, (src.height, src.width)

    with rasterio.open(src_path) as src:
        src_data = src.read()          # (8, H, W)
        src_transform = src.transform
        src_crs = src.crs
        bounds = src.bounds
        ncols = int(round((bounds.right - bounds.left) / cell_m))
        nrows = int(round((bounds.top - bounds.bottom) / cell_m))
        dst_transform = from_origin(bounds.left, bounds.top, cell_m, cell_m)
        dst_data = np.zeros((8, nrows, ncols), dtype=src_data.dtype)
        for b in range(8):
            reproject(src_data[b], dst_data[b],
                      src_transform=src_transform, src_crs=src_crs,
                      dst_transform=dst_transform, dst_crs=src_crs,
                      resampling=Resampling.nearest,
                      src_nodata=-9999, dst_nodata=-9999)
        profile = src.profile.copy()
        profile.update(height=nrows, width=ncols, transform=dst_transform,
                       count=8)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(dst_data)
    # preserve band descriptions from the source
    with rasterio.open(src_path) as src:
        names = [src.descriptions[i] or f"band{i+1}" for i in range(src.count)]
    with rasterio.open(out, "r+") as dst:
        for i, nm in enumerate(names, 1):
            dst.set_band_description(i, nm)
    dem = dst_data[0].astype(np.int16)
    with rasterio.open(out) as src:
        prof = src.profile.copy()
    prof.update(count=1, dtype="int16")
    with rasterio.open(dem_out, "w", **prof) as dst:
        dst.write(dem, 1)
        dst.set_band_description(1, "elev")
    marker.write_text(json.dumps(params, indent=2) + "\n")
    print(f"resampled landscape: {out.name} ({ncols}x{nrows} @ {cell_m:.1f} m)")
    return out, dem_out, src_crs, dst_transform, (nrows, ncols)


def open_landscape(path):
    """Read (8,H,W) float32 with -9999 nodata zeroed (matches dataloader)."""
    with rasterio.open(path) as src:
        data = src.read().astype(np.float32)
        data[data == -9999] = 0.0
        crs = src.crs
        transform = src.transform
    return data, crs, transform


def _preflight_crop(cy, cx, shape, half=_HALF):
    H, W = shape
    if (cy - half < 0 or cx - half < 0 or
            cy + half > H or cx + half > W):
        die(f"ignition pixel ({cy}, {cx}) too close to the landscape edge "
            f"({H}x{W}) for a 500-cell crop; increase landscape.bbox / replan "
            f"the crop")


# ---------------------------------------------------------------------------
# step 4 - time-varying wind via WindNinja
# ---------------------------------------------------------------------------
def _serializable_opts(opts):
    """WindNinja options are scalars, but drop anything non-JSON-safe."""
    return {k: v for k, v in opts.items()
            if isinstance(v, (str, int, float, bool, type(None)))}


def stage_windninja(cfg, run_dir, cell_m, dem_resampled, slug,
                    local_start, local_end, tz_name):
    wind = cfg["wind"]
    opts = dict(wind.get("windninja_options") or {})
    wind_root = run_dir / "windroot"
    wind_root.mkdir(parents=True, exist_ok=True)

    # WindNinja cache: if per-hour pairs already exist for these exact inputs,
    # reuse them (skip PASTCAST re-download and the solver).
    params = {
        "cell_m": float(cell_m),
        "local_start": local_start.isoformat(),
        "local_end": local_end.isoformat(),
        "tz_name": tz_name,
        "options": _serializable_opts(opts),
        "dem_mtime": dem_resampled.stat().st_mtime if dem_resampled.exists() else None,
    }
    if _windninja_cached(wind_root, params):
        # Orphan (no marker yet) -> write one so future runs are strict.
        if not (wind_root / "windninja.cache.json").exists():
            (wind_root / "windninja.cache.json").write_text(
                json.dumps(params, indent=2) + "\n")
        n = len(scan_frames(wind_root))
        print(f"reusing cached WindNinja: {n} hour(s) under {wind_root.resolve()}")
        return wind_root

    im = opts.get("initialization_method")
    if not isinstance(im, str) or not im:
        die("wind.windninja_options.initialization_method is required "
            "(e.g. initialization_method = \"wxModelInitialization\")")

    lines = [f"{k} = {_scalar(v)}" for k, v in opts.items()]
    lines += [
        f"elevation_file = {dem_resampled.resolve()}",
        f"output_path = {wind_root.resolve()}",
        f"time_zone = {tz_name}",
        f"num_threads = {int(wind.get('threads', 8))}",
        "write_ascii_output = true",
        "write_farsite_atm = true",
        f"mesh_resolution = {cell_m:g}",
        "units_mesh_resolution = m",
    ]
    wx_type = str(opts.get("wx_model_type", ""))
    if wx_type.startswith("PASTCAST-") and not any(
            k in opts for k in ("start_year", "start_month", "start_day",
                                "start_hour", "start_minute")):
        ls, le = local_start, local_end
        lines += [
            f"start_year = {ls.year}", f"start_month = {ls.month}",
            f"start_day = {ls.day}", f"start_hour = {ls.hour}",
            f"start_minute = {ls.minute}",
            f"stop_year = {le.year}", f"stop_month = {le.month}",
            f"stop_day = {le.day}", f"stop_hour = {le.hour}",
            f"stop_minute = {le.minute}",
        ]
    cfg_path = wind_root / f"{slug}.cfg"
    cfg_path.write_text("\n".join(lines) + "\n")

    command = str(wind.get("windninja_command") or "").strip()
    if not command:
        print(f"WindNinja config written: {cfg_path.resolve()}")
        print(f"run manually, then re-run this driver (it resumes):")
        print(f"  WindNinja_cli {cfg_path.resolve()}")
        print("WindNinja not invoked (wind.windninja_command empty).")
        return None
    env = dict(os.environ)
    # The PastCast HRRR archive on Google Cloud is a PUBLIC bucket. GDAL's GCS
    # VSI (/vsigs/) normally demands credentials; disabling request signing lets
    # it read the public objects anonymously (no GCS account/keys needed).
    # WindNinja's own code also checks for the key pair before calling GDAL, so
    # provide placeholder values. A user who wants signed access can set
    # GS_NO_SIGN_REQUEST=NO and real GCS creds in their shell (setdefault keeps
    # any values already present in the environment).
    env.setdefault("GS_NO_SIGN_REQUEST", "YES")
    env.setdefault("GS_ACCESS_KEY_ID", "dummy")
    env.setdefault("GS_SECRET_ACCESS_KEY", "dummy")
    # The repo's bundled Windows PROJ/GDAL/WindNinja data vars (from
    # orchestrate._native_env) must NOT be used here (they point at wine-era
    # Windows data). The conda binary needs its own runtime data: without
    # PROJ_LIB/GDAL_DATA the native PROJ/GDAL can't find proj.db / the grib2
    # tables, producing the "proj_create_from_database ... failed" and the
    # "EPSG:5070 ... not the same as the EPSG registry" warnings.
    env_prefix = Path(sys.executable).resolve().parent.parent
    for var, sub in (("PROJ_LIB", "share/proj"), ("PROJ_DATA", "share/proj"),
                     ("GDAL_DATA", "share/gdal"), ("WINDNINJA_DATA", "share/windninja")):
        data_dir = env_prefix / sub
        if data_dir.is_dir():
            env[var] = str(data_dir)
    argv = shlex.split(command) + [str(cfg_path)]
    run_subprocess(argv, cwd=str(run_dir), env=env)
    (wind_root / "windninja.cache.json").write_text(
        json.dumps(params, indent=2) + "\n")
    return wind_root


def _align_meta(meta, transform, shape):
    """True if an ASCII-grid header aligns with the landscape grid."""
    ncols, nrows = int(meta["ncols"]), int(meta["nrows"])
    cellsize = meta["cellsize"]
    aff = from_origin(meta["xllcorner"],
                      meta["yllcorner"] + nrows * cellsize,
                      cellsize, cellsize)
    H, W = shape
    return (ncols == W and nrows == H and
            abs(aff.a - transform.a) < 1e-6 and
            abs(aff.c - transform.c) < 1e-6 and
            abs(aff.e - transform.e) < 1e-6 and
            abs(aff.f - transform.f) < 1e-6)


def _grid_crs(asc_path, default_crs):
    """CRS for a wind ASCII grid: its .prj sidecar if present, else default."""
    prj = Path(asc_path).with_suffix(".prj")
    if prj.exists():
        try:
            return rasterio.crs.CRS.from_wkt(prj.read_text())
        except Exception:
            return default_crs
    return default_crs


def _regrid_wind(data, meta, src_crs, dst_transform, dst_shape):
    from rasterio.warp import reproject, Resampling
    ncols, nrows = int(meta["ncols"]), int(meta["nrows"])
    cellsize = meta["cellsize"]
    aff = from_origin(meta["xllcorner"],
                      meta["yllcorner"] + nrows * cellsize,
                      cellsize, cellsize)
    out = np.zeros(dst_shape, dtype=np.float32)
    reproject(data.astype(np.float32), out,
              src_transform=aff, src_crs=src_crs,
              dst_transform=dst_transform, dst_crs=src_crs,
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=Resampling.nearest)
    return out


def load_wind_frames(wind_root, src_crs, landscape_transform, landscape_shape,
                     std_off):
    frames = scan_frames(wind_root)
    if not frames:
        die(f"no *_vel.asc/_ang.asc pairs found under {wind_root}")
    result = []
    for stamp, (vel, ang) in frames.items():
        yyyy, mm, dd, hour = stamp
        local = datetime(yyyy, mm, dd, hour)
        utc = local - std_off
        meta_v, speed = read_asc(vel)
        meta_a, direction = read_asc(ang)
        dir_rad = np.radians(direction)
        u = -speed * np.sin(dir_rad)
        v = -speed * np.cos(dir_rad)
        H, W = landscape_shape
        u_crs = _grid_crs(vel, src_crs)
        v_crs = _grid_crs(ang, src_crs)
        if _align_meta(meta_v, landscape_transform, landscape_shape):
            u_grid = u.astype(np.float32) if (u.shape == (H, W)) else \
                _regrid_wind(u, meta_v, u_crs, landscape_transform,
                             landscape_shape)
        else:
            u_grid = _regrid_wind(u, meta_v, u_crs, landscape_transform,
                                  landscape_shape)
        if _align_meta(meta_a, landscape_transform, landscape_shape):
            v_grid = v.astype(np.float32) if (v.shape == (H, W)) else \
                _regrid_wind(v, meta_a, v_crs, landscape_transform,
                             landscape_shape)
        else:
            v_grid = _regrid_wind(v, meta_a, v_crs, landscape_transform,
                                  landscape_shape)
        result.append((utc, u_grid, v_grid))
    result.sort(key=lambda x: x[0])
    print(f"wind frames: {len(result)} per-hour grids under {wind_root}")
    return result


def nearest_wind(frame_list, real_t):
    """Pick the per-hour (utc, u_grid, v_grid) closest to ``real_t`` (UTC)."""
    if not frame_list:
        return None
    best = min(frame_list, key=lambda f: abs((f[0] - real_t).total_seconds()))
    return best


# ---------------------------------------------------------------------------
# step 6 - autoregressive rollout
# ---------------------------------------------------------------------------
def rollout(cfg, model, sample_norm, norm, wind_frames, transform, start, end,
            dt, max_t, device, cy, cx):
    times = np.arange(dt, max_t, dt)
    total = (end - start).total_seconds()
    min_w = torch.as_tensor(norm.min_val[10:12], dtype=torch.float32).to(device)
    range_w = torch.clamp(
        torch.as_tensor(norm.max_val[10:12], dtype=torch.float32).to(device)
        - min_w, min=1e-8)
    cy0, cy1 = cy - _HALF, cy + _HALF
    cx0, cx1 = cx - _HALF, cx + _HALF

    def _norm_wind(u, v):
        u = torch.as_tensor(u, dtype=torch.float32).to(device)
        v = torch.as_tensor(v, dtype=torch.float32).to(device)
        return (u - min_w[0]) / range_w[0], (v - min_w[1]) / range_w[1]

    # seed: apply the burn at t0 = dt (only the ignition cell remains lit)
    state = sample_norm.clone()
    not_burnt = state[:, 1:2] > dt
    state[:, 0:1][not_burnt] = 0.0
    state[:, 1:2][not_burnt] = 0.0

    history = []
    for t in times:
        real_t = start + timedelta(seconds=(t / max_t) * total)
        wf = nearest_wind(wind_frames, real_t)
        if wf is None:
            die("no wind grid available for the simulation window")
        _, u_grid, v_grid = wf
        # crop the wind grids to the sample window
        u_win = u_grid[cy0:cy1, cx0:cx1]
        v_win = v_grid[cy0:cy1, cx0:cx1]
        u_n, v_n = _norm_wind(u_win, v_win)
        state = real_fire.fire_burn_step_wind(
            float(t), model, state, dt, u_n, v_n)
        history.append(transform.inverse(state).squeeze(0).cpu().numpy())
    return history, times


# ---------------------------------------------------------------------------
# step 7 - georeference the final mask
# ---------------------------------------------------------------------------
def georeference_mask(history, times, cy, cx, landscape_transform,
                      landscape_crs, landscape_shape, results_dir):
    from rasterio.features import shapes
    from shapely.geometry import shape
    import shapely
    full = np.zeros(landscape_shape, dtype=np.bool_)
    cy0, cy1 = cy - _HALF, cy + _HALF
    cx0, cx1 = cx - _HALF, cx + _HALF
    mask = (history[-1][0] > 0.5)
    full[cy0:cy1, cx0:cx1] = mask
    polys = []
    for geom, val in shapes(full.astype(np.uint8), mask=full,
                            transform=landscape_transform):
        if val:
            polys.append(shape(geom))
    if not polys:
        die("simulated fire did not burn any cell (empty final mask)")
    union = shapely.union_all(polys)
    union = union.buffer(0) if not union.is_valid else union
    gdf = gpd.GeoDataFrame(geometry=[union], crs=landscape_crs)
    results_dir.mkdir(parents=True, exist_ok=True)
    out = results_dir / "simulated_perimeter.shp"
    gdf.to_file(out)
    # rollout npz
    masks = np.stack([h[0] for h in history])
    fats = np.stack([h[1] for h in history])
    np.savez(results_dir / "rollout.npz", masks=masks, fats=fats,
             times_norm=np.asarray(times, dtype=np.float64))
    return out, union


# ---------------------------------------------------------------------------
# step 8 - IoU metric
# ---------------------------------------------------------------------------
def compute_iou(sim_path, ref_path):
    sim = gpd.read_file(sim_path).to_crs(_EPSG_5070)
    ref = gpd.read_file(ref_path).to_crs(_EPSG_5070)
    a = sim.union_all().buffer(0)
    b = ref.union_all().buffer(0)
    iou = a.intersection(b).area / a.union(b).area
    return {
        "iou": float(iou),
        "sim_area_m2": float(a.area),
        "ref_area_m2": float(b.area),
        "sim_area_acres": float(a.area / _ACRE_M2),
        "ref_area_acres": float(b.area / _ACRE_M2),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv=None):
    p = argparse.ArgumentParser(
        prog="infer_real.py",
        description="Run the WildfireSurrogateModel on a real CAL FIRE fire.",
    )
    p.add_argument("--config", required=True, help="inference TOML config")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    fire = cfg["fire"]
    sim = cfg["simulation"]
    model_cfg = cfg["model"]
    if not fire.get("name"):
        die("[fire] name is required")
    if fire.get("year") is None:
        die("[fire] year is required")
    if not model_cfg.get("checkpoint"):
        die("[model] checkpoint is required")
    if not model_cfg.get("norm_stats"):
        die("[model] norm_stats is required")
    if sim.get("foliar_moisture") is None:
        die("[simulation] foliar_moisture is required")
    if not cfg["landscape"].get("email"):
        die("[landscape] email is required (LFPS requester email)")
    if sim.get("start") is None and sim.get("end") is None:
        die("[simulation] start and end (UTC) are required; set "
            "[simulation] start/end = <UTC whole hour YYYY-MM-DDTHH:MM>")

    slug = _slug(fire["name"])
    run_dir = compute_run_dir(cfg, slug)
    run_dir.mkdir(parents=True, exist_ok=True)
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ---- step 1: resolve fire ----
    fj = resolve_fire(cfg, run_dir)

    # ---- step 2: cell size ----
    cell_m, ig_x, ig_y, max_radius = compute_cell_m(cfg, fj, run_dir, results_dir)

    # ---- step 3: landscape ----
    bbox = _bbox_around(cfg, fj, cell_m, margin=float(cfg["model"].get("margin", 0.10)))
    ingest_landscape(cfg, run_dir, bbox)
    land_path, dem_path, src_crs, land_transform, land_shape = \
        resample_landscape(cfg, run_dir, cell_m)
    land_data, land_crs, land_transform = open_landscape(land_path)
    H, W = land_data.shape[1], land_data.shape[2]
    # ignition pixel (scalar x/y -> scalar row/col; arrays would be [0]-indexed)
    ig_r, ig_c = rowcol(land_transform, ig_x, ig_y)
    cy, cx = int(ig_r), int(ig_c)
    _preflight_crop(cy, cx, (H, W))

    # ---- step 4: wind ----
    lat, lon = float(fj["lat"]), float(fj["lon"])
    zi, std_off = resolve_timezone(sim.get("row_timezone"), lat, lon)
    tz_name = sim.get("row_timezone") or zi.key
    start = parse_utc(sim["start"]) if sim.get("start") else None
    start = start or _parse_or(fj.get("alarm_date"))
    end = parse_utc(sim["end"]) if sim.get("end") else None
    end = end or _parse_or(fj.get("cont_date"))
    if end is None or start is None or end < start:
        die("need [simulation] start/end (UTC) or a fire with alarm/cont dates")
    local_start = start + std_off
    local_end = end + std_off

    wind_root = stage_windninja(cfg, run_dir, cell_m, dem_path, slug,
                                local_start, local_end, tz_name)
    if wind_root is None:
        print("stopped: WindNinja .cfg written; run WindNinja manually then "
              "re-run this driver to resume")
        return 0
    wind_frames = load_wind_frames(wind_root, land_crs, land_transform,
                                   (H, W), std_off)

    # ---- step 5: build sample ----
    norm = _load_norm(cfg["model"]["norm_stats"])
    first_wf = nearest_wind(wind_frames, start)
    if first_wf is None:
        die("no wind frame near the simulation start")
    _, u0, v0 = first_wf
    sample = real_fire.build_sample(
        land_data, u0, v0, cy, cx,
        foliar_moisture=float(sim["foliar_moisture"]),
        dt=_eval_fraction(sim["dt"]) if isinstance(sim.get("dt"), str)
        else float(sim.get("dt", 0.125)),
        max_val_fat=float(norm.max_val[1]),
    )
    real_fire.save_input_channels(sample, results_dir)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MK_UNet_Regression(in_channels=14, out_channels=1,
                               channels=[16, 32, 64, 96, 160],
                               final_activation="sigmoid")
    init_from_checkpoint(model, cfg["model"]["checkpoint"])
    model.to(device).eval()
    print(f"model: {cfg['model']['checkpoint']} on {device}")

    # ---- step 6: rollout ----
    transform = MinMaxPerChannel(norm.min_val, norm.max_val)
    sample_norm = transform(torch.as_tensor(sample).unsqueeze(0).to(device))
    dt = _eval_fraction(sim["dt"]) if isinstance(sim.get("dt"), str) \
        else float(sim.get("dt", 0.125))
    max_t = float(sim.get("max_t", 1.0))
    history, times = rollout(cfg, model, sample_norm, norm, wind_frames,
                             transform, start, end, dt, max_t, device, cy, cx)
    print(f"rollout: {len(history)} steps")
    real_fire.save_arrival_map(history[-1][1], results_dir)

    # ---- step 7: georeference ----
    sim_perim, sim_union = georeference_mask(history, times, cy, cx,
                                             land_transform, land_crs,
                                             (H, W), results_dir)

    # ---- step 8: IoU ----
    metric = compute_iou(sim_perim, run_dir / "reference_perimeter.shp")
    print(f"IoU: {metric['iou']:.4f}")

    # ---- step 9: render ----
    crop_affine = land_transform * rasterio.Affine.translation(cx - _HALF, cy - _HALF)
    ref_gdf = gpd.read_file(run_dir / "reference_perimeter.shp").to_crs(_EPSG_5070)
    ref_geom = ref_gdf.union_all().buffer(0)
    real_fire.render(
        run_dir, history, times, crop_affine, land_crs,
        ref_geom, _EPSG_5070, (ig_x, ig_y), start, end,
        dt, max_t, float(norm.min_val[1]), float(norm.max_val[1]),
        out_gif=str(results_dir / "spread.gif"),
        out_png=str(results_dir / "arrival_map.png"),
    )

    # ---- write result.json ----
    result = {
        "fire": fj.get("name"), "year": fj.get("year"),
        "start": _fmt_dt(start), "end": _fmt_dt(end),
        "cell_m": cell_m, "crop_span_km": float(cell_m * _CROP / 1000.0),
        "iou": metric["iou"],
        "sim_area_acres": metric["sim_area_acres"],
        "ref_area_acres": metric["ref_area_acres"],
        "checkpoint": cfg["model"]["checkpoint"],
        "norm_stats": cfg["model"]["norm_stats"],
        "foliar_moisture": float(sim["foliar_moisture"]),
    }
    (results_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n")
    print(f"result: {results_dir / 'result.json'}")
    return 0


def _parse_or(s):
    """Parse a fire.json ISO datetime (may carry seconds/offset) to a naive UTC."""
    if not s:
        return None
    text = str(s).strip()
    if text.endswith(("Z", "z")):
        text = text[:-1]
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        die(f"cannot parse {s!r} as an ISO datetime")


def _load_norm(path):
    z = np.load(path)
    return _Norm(min_val=z["min_val"], max_val=z["max_val"])


class _Norm:
    def __init__(self, min_val, max_val):
        self.min_val = np.asarray(min_val, dtype=np.float32)
        self.max_val = np.asarray(max_val, dtype=np.float32)


if __name__ == "__main__":
    sys.exit(main())
