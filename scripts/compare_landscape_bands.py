"""
compare_landscape_bands.py — Verify that multiple landscape GeoTIFFs share the
same band structure and (as far as detectable) the same band ordering/meaning.

A GeoTIFF does not generally store per-band semantics, so this compares in
three layers, from most to least authoritative:

  1. Structural: band count, per-band dtype, nodata, CRS, resolution.
  2. Named bands: GDAL band descriptions/tags, if present.
  3. Heuristic: per-band value ranges (min/mean/max) and integer-ness, which
     act as a fingerprint of each band's physical meaning (elevation in m,
     slope 0-90, aspect 0-360, fuel model as small integer codes, etc.).

The first file is treated as the reference; every other file is diffed against
it. Exit status is non-zero if any structural mismatch is found.

Usage:
    python scripts/compare_landscape_bands.py <ref.tif> <other.tif> [<other2.tif> ...]
"""

import argparse
import sys

import numpy as np
import rasterio


EXPECTED_BAND_NAMES = [
    "elevation", "slope", "aspect", "fuel_model",
    "canopy_cover", "stand_height", "canopy_base_height", "canopy_bulk_density",
]


def band_fingerprint(ds, b, nodata_extra=(-9999,)):
    """Return a dict of structural + statistical properties for band `b` (1-based)."""
    dtype = ds.dtypes[b - 1]
    nodata = ds.nodatavals[b - 1]
    desc = ds.descriptions[b - 1]

    arr = ds.read(b).astype(np.float64)
    mask = np.ones(arr.shape, dtype=bool)
    if nodata is not None:
        mask &= arr != nodata
    for nd in nodata_extra:
        mask &= arr != nd
    finite = arr[mask & np.isfinite(arr)]

    if finite.size:
        vmin, vmax, vmean = float(finite.min()), float(finite.max()), float(finite.mean())
        n_unique = int(np.unique(finite).size)
        # integer-valued? (fuel models are small integer codes)
        is_int = bool(np.all(np.equal(np.mod(finite, 1), 0)))
    else:
        vmin = vmax = vmean = float("nan")
        n_unique = 0
        is_int = False

    return {
        "dtype": dtype,
        "nodata": nodata,
        "desc": desc,
        "min": vmin,
        "max": vmax,
        "mean": vmean,
        "n_unique": n_unique,
        "is_int": is_int,
    }


def summarize(path):
    with rasterio.open(path) as ds:
        info = {
            "path": path,
            "count": ds.count,
            "width": ds.width,
            "height": ds.height,
            "crs": str(ds.crs),
            "res": tuple(round(r, 6) for r in ds.res),
            "bands": [band_fingerprint(ds, b) for b in range(1, ds.count + 1)],
        }
    return info


def fmt_band(i, fp):
    name = EXPECTED_BAND_NAMES[i] if i < len(EXPECTED_BAND_NAMES) else f"band{i}"
    desc = fp["desc"] if fp["desc"] else "-"
    return (f"  [{i}] {name:20s} dtype={fp['dtype']:8s} desc={desc:12s} "
            f"range=[{fp['min']:.2f}, {fp['max']:.2f}] mean={fp['mean']:.2f} "
            f"uniq={fp['n_unique']:6d} int={fp['is_int']}")


def compare(ref, other, range_tol=0.25):
    """Return list of human-readable difference strings between two summaries."""
    diffs = []
    if ref["count"] != other["count"]:
        diffs.append(f"BAND COUNT differs: ref={ref['count']} vs {other['count']}")
    if ref["crs"] != other["crs"]:
        diffs.append(f"CRS differs:\n    ref={ref['crs']}\n    oth={other['crs']}")
    if ref["res"] != other["res"]:
        diffs.append(f"RESOLUTION differs: ref={ref['res']} vs {other['res']}")

    n = min(ref["count"], other["count"])
    for i in range(n):
        rb, ob = ref["bands"][i], other["bands"][i]
        name = EXPECTED_BAND_NAMES[i] if i < len(EXPECTED_BAND_NAMES) else f"band{i}"

        if rb["dtype"] != ob["dtype"]:
            diffs.append(f"band[{i}] {name}: dtype ref={rb['dtype']} vs {ob['dtype']}")
        if (rb["desc"] or None) != (ob["desc"] or None):
            diffs.append(f"band[{i}] {name}: description ref={rb['desc']!r} vs {ob['desc']!r}")
        if rb["is_int"] != ob["is_int"]:
            diffs.append(f"band[{i}] {name}: integer-valued ref={rb['is_int']} vs {ob['is_int']} "
                         f"(possible meaning/order mismatch)")

        # Heuristic range check: flag if ranges are wildly different in scale.
        # Different terrains legitimately differ in elevation etc., so this is a
        # soft signal — large *relative* divergence hints at a reordering.
        for key in ("min", "max"):
            r, o = rb[key], ob[key]
            if np.isfinite(r) and np.isfinite(o):
                denom = max(abs(r), abs(o), 1.0)
                if abs(r - o) / denom > (1.0 / range_tol):  # >4x scale difference
                    diffs.append(f"band[{i}] {name}: {key} scale differs a lot "
                                 f"ref={r:.2f} vs {o:.2f} (check band meaning/order)")
    return diffs


def main():
    parser = argparse.ArgumentParser(description="Compare band structure/order across landscape GeoTIFFs")
    parser.add_argument("tifs", nargs="+", help="landscape GeoTIFFs; first is the reference")
    args = parser.parse_args()

    if len(args.tifs) < 2:
        parser.error("provide at least two GeoTIFFs to compare")

    summaries = [summarize(p) for p in args.tifs]

    ref = summaries[0]
    print("=" * 72)
    print(f"REFERENCE: {ref['path']}")
    print(f"  count={ref['count']} size={ref['width']}x{ref['height']} "
          f"crs={ref['crs'][:40]}... res={ref['res']}")
    for i, fp in enumerate(ref["bands"]):
        print(fmt_band(i, fp))
    print("=" * 72)

    any_diff = False
    for other in summaries[1:]:
        print(f"\nCOMPARE vs: {other['path']}")
        print(f"  count={other['count']} size={other['width']}x{other['height']} "
              f"crs={other['crs'][:40]}... res={other['res']}")
        for i, fp in enumerate(other["bands"]):
            print(fmt_band(i, fp))

        diffs = compare(ref, other)
        if not diffs:
            print("  ✓ structurally aligned with reference (no differences flagged)")
        else:
            any_diff = True
            print("  ✗ DIFFERENCES:")
            for d in diffs:
                print("    - " + d)

    print("\n" + "=" * 72)
    if any_diff:
        print("RESULT: differences found — review before treating as the same band layout.")
        sys.exit(1)
    else:
        print("RESULT: all files structurally aligned with the reference.")
        sys.exit(0)


if __name__ == "__main__":
    main()
