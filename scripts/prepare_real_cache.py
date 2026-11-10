#!/usr/bin/env python3
"""prepare_real_cache - decompress the real-data samples once into a mirror.

Why: the product stores each fire's samples as a single DEFLATE-compressed
``.npy`` member inside its ``.npz``, so *any* access to a fire costs a full
inflate of that fire (measured ~150 MB/s per core). Training re-inflates the
whole training split every epoch — ~104 GB for the v5 train split at 8 workers,
i.e. more wall time than the model itself. This converts each fire once into
plain ``.npy`` sidecars that training memory-maps instead:

    <out>/<fire_id>.X.npy   (n, 135, 135, 55) float32|float16  == npz['X']
    <out>/<fire_id>.y.npy   (n, 135, 135) uint8               == npz['y']
    <out>/manifest.json     source root, dtype, shape, fire ids

**All 55 channels are kept**, so the cache is a drop-in for the npz: the channel
subset stays a training-time choice and no downstream indexing changes.

Sizes for v5 (38,147 samples, 1.9x compressed on disk): float32 ~150 GB in
total (train ~102 GB, val ~24 GB, test ~23 GB), float16 ~75 GB. Reading is
sequential and memory-mapped, so the per-epoch input cost drops to the raw read
(~1-2 GB/s per NVMe) with no inflate at all.

float16 halves the size and the read traffic at the cost of ~1 m quantization on
elevation-scale channels (fp16 has an 11-bit mantissa); use float32 for a run
that must be numerically identical to reading the npz.

Usage:
    python scripts/prepare_real_cache.py --real-data /mnt/wildfire/real-data/v5_wildfires_2026-10-08 \
        --out /mnt/wildfire/real-data/v5_wildfires_2026-10-08-uncompressed \
        --splits train,val,test --workers 12

Then train against it (the run's config keeps the path, so evaluation picks it
up too):

    python scripts/train_real.py --output-dir <run> --data-cache <out>

The conversion is resumable: an existing output file is skipped unless
``--overwrite`` is given.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from tqdm import tqdm

from finetune.dataset import RealDataStore, RealDataError

MANIFEST = "manifest.json"


def convert_fire(args):
    """Decompress one fire into its two sidecars.

    Returns ``(fire_id, wrote, bytes, source_size, source_mtime_ns)``; the source
    fingerprint travels into the manifest so a later rebuild of the product
    cannot be silently trained against a stale mirror.
    """
    root, out, fire_id, dtype, overwrite = args
    x_out = out / f"{fire_id}.X.npy"
    y_out = out / f"{fire_id}.y.npy"
    src = root / "samples" / f"{fire_id}.npz"
    if not src.is_file():
        raise RealDataError(f"missing sample file {src}")
    src_stat = src.stat()
    if not overwrite and x_out.is_file() and y_out.is_file():
        return (fire_id, False, 0, src_stat.st_size, src_stat.st_mtime_ns)
    with np.load(src) as z:
        X, y = z["X"], z["y"]
    np.save(x_out, X.astype(dtype, copy=False))
    np.save(y_out, y)
    return (fire_id, True, int(x_out.stat().st_size + y_out.stat().st_size),
            src_stat.st_size, src_stat.st_mtime_ns)


def main(argv=None):
    p = argparse.ArgumentParser(
        description="decompress the real-data samples into a .npy mirror",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--real-data", required=True, help="the v5 real-data root")
    p.add_argument("--out", required=True, help="cache directory to create")
    p.add_argument("--splits", default="train,val,test",
                   help="comma-separated frozen split keys, or 'all'")
    p.add_argument("--dtype", default="float32", choices=["float32", "float16"],
                   help="stored dtype (float16 halves size and read traffic)")
    p.add_argument("--workers", type=int, default=8,
                   help="parallel conversions (each holds one fire in RAM)")
    p.add_argument("--overwrite", action="store_true",
                   help="reconvert fires whose sidecars already exist")
    p.add_argument("--limit-fires", type=int, default=None)
    args = p.parse_args(argv)

    root = Path(args.real_data).resolve()
    out = Path(args.out).resolve()
    if out == root / "samples":
        sys.exit("refusing to convert into the data root's own samples/")
    out.mkdir(parents=True, exist_ok=True)

    # Table-only: decompressing data must not require training stats.
    store = RealDataStore(root, load_norm_stats=False)
    if args.splits == "all":
        fire_ids = sorted(store.fires["fire_id"].astype(str))
    else:
        fire_ids = []
        for key in args.splits.split(","):
            fire_ids.extend(store.split_fires(key.strip()))
        seen, unique = set(), []
        for f in fire_ids:               # a fire can appear in one split only,
            if f not in seen:            # but keep this robust anyway
                seen.add(f)
                unique.append(f)
        fire_ids = unique
    if args.limit_fires is not None:
        fire_ids = fire_ids[:args.limit_fires]
    if not fire_ids:
        sys.exit("no fires selected")

    dtype = np.dtype(args.dtype)
    n_samples = sum(store.n_samples(f) for f in fire_ids)
    per_sample = 135 * 135 * 55 * dtype.itemsize + 135 * 135
    print(f"{len(fire_ids)} fires, {n_samples} samples -> {out}")
    print(f"stored as {dtype}: about {n_samples * per_sample / 1e9:.0f} GB "
          f"({args.workers} workers, one fire per worker in RAM)")

    t0 = time.perf_counter()
    wrote = 0
    total_bytes = 0
    jobs = [(root, out, f, dtype, args.overwrite) for f in fire_ids]
    sources = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fire_id, did_write, nbytes, size, mtime in tqdm(
                pool.map(convert_fire, jobs), total=len(jobs), unit="fire"):
            wrote += int(did_write)
            total_bytes += nbytes
            sources[fire_id] = [int(size), int(mtime)]
    dt = time.perf_counter() - t0
    print(f"converted {wrote} fires ({total_bytes/1e9:.0f} GB) in "
          f"{dt/60:.1f} min")

    manifest_path = out / MANIFEST
    previous = {}
    if manifest_path.is_file():
        with open(manifest_path) as f:
            previous = json.load(f)
    if previous and previous.get("dtype") != dtype.name:
        sys.exit(f"{manifest_path} is a {previous.get('dtype')} cache; refusing "
                 f"to mix dtypes (pass a different --out or delete it)")
    # Converting another split must not drop the fingerprints of the fires
    # already in the mirror (that would silently disable the stale-mirror check
    # for them).
    sources_all = dict(previous.get("source_sizes") or {})
    sources_all.update(sources)
    fires_done = sorted(p.stem.split(".")[0] for p in out.glob("*.X.npy"))
    manifest = {
        "source_root": str(root),
        "dtype": dtype.name,
        "layout": "X: (n, 135, 135, 55), y: (n, 135, 135), same order as the npz",
        "channels": store.channel_names_all,
        "fires": fires_done,
        "source_sizes": sources_all,
        "updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {manifest_path} ({len(fires_done)} fires)")
    print(f"\ntrain against it with:  --data-cache {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
