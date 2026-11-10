import json
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from wildfire_simulator.dataloader import WildfireDataLoader, TrialFileLoader, TrialCollection
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"

# --- real-data (v5 wildfires) fixture --------------------------------------
# A tiny synthetic root with the same layout as the real product: two fires, a
# per-sample window origin, growing perimeters, a pre-fire FEDS placeholder on
# one fire and a large window drift on the other.
WIN = 135
REAL_CHANNELS = [
    ("perimeter", "feds_perimeter", "at_input"),
    ("frp", "viirs_frp", "at_input"),
    ("n_detections", "viirs_frp", "at_input"),
    ("new_fire_pixels", "viirs_frp", "at_input"),
    ("u10", "hrrr", "window_mean"),
    ("v10", "hrrr", "window_mean"),
    ("hours_since_start", "time", "at_input"),
    ("elev", "landfire", "static"),
    ("slope_mean", "landfire", "static"),
    ("fbfm_mode", "landfire", "static"),
    ("wn_u10", "windninja", "window_mean"),
    ("wn_v10", "windninja", "window_mean"),
]
# (fire_id, name, state, county, [(row0, col0), ...])
REAL_FIRES = [
    ("V5_2020_2020_00001", "Alpha", "CA", "Test", [(10, 20), (11, 21), (12, 21), (12, 22)]),
    # third sample's window jumps 60 rows north: part of the start window is
    # then uncovered (NaN inputs, unscored cells)
    ("V5_2020_2020_00002", "Beta", "OR", "Test", [(10, 20), (10, 20), (70, 20)]),
]


def _arrival_grid(size=200):
    """Fire-grid first-burn step: band of 18 rows per step, step 0 = seed."""
    rows, cols = np.mgrid[0:size, 0:size]
    arr = np.full((size, size), -1, dtype=np.int16)
    inside = (rows >= 60) & (rows < 60 + 90) & (cols >= 60) & (cols < 60 + 70)
    arr[inside] = ((rows[inside] - 60) // 18)
    return arr


def _window(arr, row0, col0, fill=-1):
    """Slice a fire-grid array into a 135x135 sample window."""
    out = np.full((WIN, WIN), fill, dtype=arr.dtype)
    r0, r1 = row0, row0 + WIN
    c0, c1 = col0, col0 + WIN
    src = arr[max(r0, 0):min(r1, arr.shape[0]), max(c0, 0):min(c1, arr.shape[1])]
    out[:src.shape[0], :src.shape[1]] = src
    return out


@pytest.fixture(scope="session")
def real_root(tmp_path_factory):
    """A synthetic real-data root (fires.csv, samples.csv, channels.json, npz)."""
    root = tmp_path_factory.mktemp("real_data")
    (root / "samples").mkdir()
    arrival = _arrival_grid()

    with open(root / "channels.json", "w") as f:
        json.dump([{"index": i, "name": n, "product": p, "cadence": c}
                   for i, (n, p, c) in enumerate(REAL_CHANNELS)], f)
    stats = {n: {"mean": 0.0, "std": 1.0, "log": n in ("frp", "n_detections", "new_fire_pixels")}
             for n, _, _ in REAL_CHANNELS if n != "fbfm_mode"}
    (root / "norm_cache").mkdir()
    with open(root / "norm_cache" / "stats.json", "w") as f:
        json.dump(stats, f)

    rows = []
    for fire_id, name, state, county, origins in REAL_FIRES:
        n = len(origins)
        X = np.zeros((n, WIN, WIN, len(REAL_CHANNELS)), dtype=np.float32)
        y = np.zeros((n, WIN, WIN), dtype=np.uint8)
        for j, (row0, col0) in enumerate(origins):
            perim = (_window(arrival, row0, col0) >= 0) & \
                    (_window(arrival, row0, col0) <= j)
            label = (_window(arrival, row0, col0) >= 0) & \
                    (_window(arrival, row0, col0) <= j + 1)
            if fire_id.endswith("00001") and j == 0:
                # FEDS pre-fire placeholder: 5 cells, not the real ignition
                perim = np.zeros((WIN, WIN), dtype=bool)
                perim[61 - row0, 61 - col0:66 - col0] = True
            X[j, :, :, 0] = perim
            y[j] = label
            X[j, :, :, 1] = 3.0       # frp (satellite observation at input time)
            X[j, :, :, 2] = 2.0
            X[j, :, :, 3] = 1.0
            X[j, :, :, 4] = 4.0 + j   # u10
            X[j, :, :, 5] = -2.0      # v10
            X[j, :, :, 6] = 12.0 * j  # hours_since_start
            X[j, :, :, 7] = 100.0 + 10.0 * j   # elev (static in reality: varies
                                               # here so the test can prove the
                                               # dataset reads it from the start)
            X[j, :, :, 8] = 5.0 + j
            X[j, :, :, 9] = 3.0       # fbfm_mode (no norm stats)
            X[j, :, :, 10] = 5.0 + j  # wn_u10
            X[j, :, :, 11] = -1.0 - j  # wn_v10
        np.savez(root / "samples" / f"{fire_id}.npz", X=X, y=y)
        for j, (row0, col0) in enumerate(origins):
            t_in = datetime(2020, 8, 1) + timedelta(hours=12 * j)
            rows.append({
                "sample_index": len(rows), "fire_id": fire_id, "step": j,
                "t_input": t_in.strftime("%Y-%m-%d %H:%M:%S"),
                "t_label": (t_in + timedelta(hours=12)).strftime("%Y-%m-%d %H:%M:%S"),
                "growth_px": 0 if j == 0 else 10, "category": "steady",
                "row0": row0, "col0": col0, "lat0": 40.0 + row0 / 1000,
                "lon0": -120.0 + col0 / 1000,
            })
    pd.DataFrame(rows).to_csv(root / "samples.csv", index=False)
    pd.DataFrame([
        {"fire_id": fid, "name": name, "state": state, "county": county,
         "n_samples": len(origins), "incident_number": f"2020-TEST-{i:06d}",
         "centroid_lat": 40.0, "centroid_lon": -120.0}
        for i, (fid, name, state, county, origins) in enumerate(REAL_FIRES)
    ]).to_csv(root / "fires.csv", index=False)
    with open(root / "split_frozen_test.json", "w") as f:
        json.dump({"train": [REAL_FIRES[0][0]], "val": [REAL_FIRES[1][0]],
                   "test": []}, f)
    return root


@pytest.fixture(scope="session")
def real_store(real_root):
    from finetune.dataset import RealDataStore

    return RealDataStore(real_root, norm_stats=real_root / "norm_cache" / "stats.json")


@pytest.fixture(scope="session")
def dataloader():
    trials = TrialCollection(
        TrialFileLoader(),
        trials_dir=str(DATA_DIR / "Trials"),
    )
    return WildfireDataLoader(
        trials,
        landscape_path=str(DATA_DIR / "palisades.tif"),
        ignitions_dir=str(DATA_DIR / "Ignitions"),
    )
