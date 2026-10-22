#!/usr/bin/env python3
"""compare_real_spread - compare inferred per-step spread to real fire products.

Drives ``wildfire_simulator.real_spread`` over a completed ``infer_real.py`` run
and a real-data folder (NASA FEDS perimeters on VIIRS). It produces, in an
output dir (default ``<run_dir>/results/real_compare``):

  * ``metrics.json``    - per-step aligned IoU, avg step IoU, final footprint IoU
  * ``spread_compare.gif`` - per-step side-by-side real vs model spread
  * ``arrival_compare.png`` - side-by-side first-burn (arrival) time maps

The model's raw per-step masks are used (never the cleaned largest-component
perimeter). The comparison is modeled-vs-real only; the CAL FIRE reference
perimeter is never read or drawn here.

    python scripts/compare_real_spread.py \
      --run-dir  farsite-docker/FireBehaviorModels/SampleData/2025_palisades \
      --real-data /mnt/wildfire/real-data/v5_wildfires_2026-10-08
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Neutralize stray PROJ_LIB/PROJ_DATA (bundled Windows GDAL data) before pyproj.
for _proj_key in ("PROJ_LIB", "PROJ_DATA"):
    os.environ.pop(_proj_key, None)

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from wildfire_simulator import real_spread as rs  # noqa: E402

_EPSG_5070 = "EPSG:5070"


def _load_json(path):
    with open(path) as f:
        return json.load(f)


def _compute_window(rows, start, end, start_step, steps):
    """Resolve the real-data chain window ``(start_step, steps)``.

    Defaults: start at the first sample whose ``t_label`` is at/after the
    simulation start; include every later sample whose ``t_label`` is at/before
    the simulation end (at least one).
    """
    labels = [rs._parse_utc(v) for v in rows["t_label"]]
    if start_step is None:
        start_step = next((i for i, tl in enumerate(labels) if tl >= start), 0)
    if steps is None:
        steps = sum(1 for tl in labels[start_step:] if tl <= end)
        steps = max(1, steps)
    return int(start_step), int(steps)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="compare inferred spread to real per-step fire products")
    p.add_argument("--run-dir", required=True,
                   help="a completed infer_real run_dir")
    p.add_argument("--real-data", required=True,
                   help="real-data root (fires.csv, samples.csv, samples/)")
    p.add_argument("--name", help="fire name (default: run fire.json name)")
    p.add_argument("--year", type=int, help="fire year (default: run fire.json year)")
    p.add_argument("--fire-id", help="real-data fire_id (bypasses name matching)")
    p.add_argument("--inc", help="incident-number trailing digits (disambiguator)")
    p.add_argument("--state", help="state (disambiguator)")
    p.add_argument("--county", help="county (disambiguator)")
    p.add_argument("--start-step", type=int, help="first real-data sample index")
    p.add_argument("--steps", type=int, help="number of real-data steps to report")
    p.add_argument("--out", help="output dir (default <run_dir>/results/real_compare)")
    args = p.parse_args(argv)

    run_dir = Path(args.run_dir).resolve()
    real_data = Path(args.real_data).resolve()
    result = _load_json(run_dir / "results" / "result.json")
    fire = _load_json(run_dir / "fire.json")
    start = rs._parse_utc(result["start"])
    end = rs._parse_utc(result["end"])
    cell_m = float(result["cell_m"])
    crop_span_km = result.get("crop_span_km", cell_m * rs._CROP / 1000.0)
    name = args.name or fire["name"]
    year = args.year or int(fire["year"])

    z = np.load(run_dir / "results" / "rollout.npz")
    masks, times_norm = z["masks"], z["times_norm"]
    surr_times = rs.surrogate_extent_times(start, end, times_norm)

    # geo-reference the surrogate crop (reproduces infer_real's crop_affine)
    land_tif = run_dir / "landscape_resampled.tif"
    if land_tif.exists():
        import rasterio

        src = rasterio.open(land_tif)
        land_transform, landscape_crs = src.transform, src.crs
        crop_affine = rs._surrogate_crop_affine(
            land_transform, landscape_crs, float(fire["lon"]), float(fire["lat"]),
            cell_m)
    else:
        from rasterio.transform import from_origin
        from pyproj import Transformer

        crs = fire.get("crs", _EPSG_5070)
        trans = Transformer.from_crs(4326, crs, always_xy=True)
        x, y = trans.transform(float(fire["lon"]), float(fire["lat"]))
        crop_affine = from_origin(x - rs._HALF * cell_m, y + rs._HALF * cell_m,
                                  cell_m, cell_m)
        landscape_crs = crs
        print("  note: landscape_resampled.tif absent; crop affine reconstructed "
              "from fire.json+cell_m (<=1-cell registration tolerance)")

    # match the real fire
    disambiguator = None
    if args.inc:
        disambiguator = ("inc", args.inc)
    elif args.state:
        disambiguator = ("state", args.state)
    elif args.county:
        disambiguator = ("county", args.county)
    fires_csv = real_data / "fires.csv"
    row = rs.load_fire(fires_csv, name=name, year=year, disambiguator=disambiguator,
                       fire_id=args.fire_id, run_inc=fire.get("inc_num"))
    fire_id = row["fire_id"]
    print(f"real fire: {row['name']} ({fire_id}) {row['state']}/{row['county']}")

    # real-data chain window
    import pandas as pd

    samples_csv = real_data / "samples.csv"
    samples_npz = real_data / "samples" / f"{fire_id}.npz"
    rows = pd.read_csv(samples_csv)
    rows = rows[rows["fire_id"].astype(str) == str(fire_id)]
    rows = rows.sort_values("step").reset_index(drop=True)
    start_step, steps = _compute_window(rows, start, end, args.start_step, args.steps)
    chain = rs.load_chain(fire_id, samples_csv, samples_npz, start_step, steps)

    # resample surrogate masks onto the chain grid, and the coverage mask
    pred_on_grid = [rs.sample_surrogate_on_grid(m, crop_affine, landscape_crs,
                                                chain.grid) for m in masks]
    coverage = rs.surrogate_crop_coverage(crop_affine, landscape_crs, chain.grid)

    # align + score per step
    aligned = rs.align_truth_to_surrogate(chain.truth_times, surr_times)
    pairs = []
    for k, surr_idx in enumerate(aligned):
        valid = coverage & chain.scored[k]
        iv = rs.iou(pred_on_grid[surr_idx], chain.truths[k], valid)
        pairs.append((chain.truth_times[k], surr_times[surr_idx], iv,
                      pred_on_grid[surr_idx], chain.truths[k]))

    n_aligned = len(pairs)
    avg_step_iou = float(np.mean([iv for _, _, iv, _, _ in pairs])) if pairs else None

    # final footprint IoU: model extent nearest end vs real burned nearest end
    if pairs:
        surr_end = int(np.argmin([abs((t - end).total_seconds()) for t in surr_times]))
        truth_end = int(np.argmin([abs((t - end).total_seconds())
                                   for t in chain.truth_times]))
        valid_end = coverage & chain.scored[truth_end]
        final_iou = rs.iou(pred_on_grid[surr_end], chain.truths[truth_end],
                           valid_end)
    else:
        final_iou = None

    # metrics
    steps_meta = [{
        "truth_time": tt.strftime("%Y-%m-%dT%H:%M:%S"),
        "surr_time": st.strftime("%Y-%m-%dT%H:%M:%S"),
        "iou": iv,
    } for tt, st, iv, _, _ in pairs]
    out = Path(args.out) if args.out else (run_dir / "results" / "real_compare")
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "fire": row["name"], "year": int(year), "fire_id": fire_id,
        "start": start.strftime("%Y-%m-%dT%H:%M:%S"),
        "end": end.strftime("%Y-%m-%dT%H:%M:%S"),
        "cell_m": cell_m, "crop_span_km": float(crop_span_km),
        "chain_start_step": int(start_step), "n_aligned": n_aligned,
        "avg_step_iou": avg_step_iou, "final_iou": final_iou,
        "steps": steps_meta,
    }
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # visuals
    rs.render_side_by_side_gif(pairs, chain.grid, out / "spread_compare.gif")
    chain_start_time = chain.truth_times[0]
    real_arr = rs.real_arrival_map(chain.truths, chain.truth_times, chain_start_time)
    surr_arr = rs.surrogate_arrival_map(pred_on_grid, surr_times,
                                        chain_start_time)
    rs.render_arrival_compare(real_arr, surr_arr, chain.grid,
                              out / "arrival_compare.png")
    rs.render_combined_fire_map(
        run_dir, chain.grid, real_arr, surr_arr,
        (float(fire["lon"]), float(fire["lat"])),
        out / "combined_fat_map.png",
        title=f"{row['name']} ({fire_id})")

    print(f"  chain: start_step={start_step} steps={steps}")
    print(f"  aligned pairs: {n_aligned}   avg_step_iou="
          f"{'None' if avg_step_iou is None else f'{avg_step_iou:.3f}'}"
          f"   final_iou={final_iou if final_iou is None else f'{final_iou:.3f}'}")
    for k, (tt, st, iv, _, _) in enumerate(pairs):
        print(f"    step {k}: real {tt:%Y-%m-%d %H:%M} -> model {st:%Y-%m-%d %H:%M}"
              f"  IoU={iv:.3f}")
    print(f"  wrote {out / 'metrics.json'}")
    print(f"  wrote {out / 'spread_compare.gif'}")
    print(f"  wrote {out / 'arrival_compare.png'}")
    print(f"  wrote {out / 'combined_fat_map.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
