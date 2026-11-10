"""Tests for the real-data chain layer (finetune.dataset)."""

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from finetune.dataset import (
    ARTIFACT_THRESHOLD, RealChainDataset, RealDataError, RealDataStore,
    StreamingChainDataset, build_full_life_chain, build_val_chains,
    collate_chains, fire_chain_cuts,
)

NORM = "norm_cache/stats.json"
FIRE_A = "V5_2020_2020_00001"   # 4 samples, step 0 is a 5-cell FEDS placeholder
FIRE_B = "V5_2020_2020_00002"   # 3 samples, 3rd window jumps 60 rows


def _store(root, **kw):
    return RealDataStore(root, norm_stats=root / NORM, **kw)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
def test_channels_default_excludes_unstattable_channel(real_store):
    # fbfm_mode has no normalization statistics and is redundant with the
    # fbfm_* one-hot channels, so it is not in the default input set.
    assert "fbfm_mode" not in real_store.channels
    assert "perimeter" in real_store.channels
    assert len(real_store.channels) == 11
    assert real_store.mean.shape == (11,)
    assert bool(real_store.static[real_store.channels.index("elev")])
    assert not bool(real_store.static[real_store.channels.index("u10")])
    assert "real data" in real_store.describe()


def test_requested_channel_without_stats_fails_loudly(real_root):
    with pytest.raises(RealDataError, match="fbfm_mode"):
        _store(real_root, channels=["perimeter", "fbfm_mode"])


def test_unknown_channel_fails_loudly(real_root):
    with pytest.raises(RealDataError, match="unknown channels"):
        _store(real_root, channels=["perimeter", "not_a_channel"])


def test_missing_norm_stats_fails_loudly(real_root):
    with pytest.raises(RealDataError, match="normalization stats not found"):
        RealDataStore(real_root)          # default cache name does not exist


def test_split_and_fire_resolution(real_store):
    assert real_store.split_fires("val") == [FIRE_B]
    assert real_store.split_fires("train") == [FIRE_A]
    with pytest.raises(RealDataError, match="split"):
        real_store.split_fires("nope")
    assert real_store.resolve_fire(name="alpha", year=2020) == FIRE_A
    assert real_store.resolve_fire(fire_id=FIRE_B) == FIRE_B
    with pytest.raises(RealDataError, match="no real fire"):
        real_store.resolve_fire(name="alpha", year=1999)
    with pytest.raises(RealDataError, match="unknown fire_id"):
        real_store.resolve_fire(fire_id="V5_0000_0000_00000")


def test_first_fire_step_skips_the_placeholder(real_store):
    assert real_store.first_fire_step(FIRE_A) == 1
    assert real_store.first_fire_step(FIRE_B) == 0
    assert real_store.n_samples(FIRE_A) == 4
    assert real_store.n_samples(FIRE_B) == 3


def test_first_fire_step_matches_the_brute_force_scan(real_store):
    """The row-at-a-time scan must agree with the full-array convention."""
    for fire_id in (FIRE_A, FIRE_B):
        X = real_store._arrays(fire_id)[0]
        counts = (X[:, :, :, 0] > 0.5).sum(axis=(1, 2))
        idx = np.flatnonzero(counts > ARTIFACT_THRESHOLD)
        expected = int(idx[0]) if idx.size else 0
        assert real_store.first_fire_step(fire_id) == expected


# ---------------------------------------------------------------------------
# chains
# ---------------------------------------------------------------------------
def test_chain_state_target_and_scoring(real_store):
    chain = build_full_life_chain(real_store, FIRE_B)
    assert chain.length == 3
    assert chain.start == 0

    # channel 0 is the state *before* the step: the seed, then the previous
    # step's cumulative truth ("burned land does not unburn").
    assert np.array_equal(chain.x[0, 0] > 0.5, chain.perim0)
    assert np.array_equal(chain.x[1, 0] > 0.5, chain.y[0])
    assert np.array_equal(chain.x[2, 0] > 0.5, chain.y[1])

    # truth is monotone and grows by the label
    for k in range(1, chain.length):
        assert (chain.y[k] & ~chain.y[k - 1]).any()                 # it grew
        assert not (chain.y[k - 1] & ~chain.y[k]).any()             # never shrank
    assert chain.y[-1].sum() > chain.perim0.sum()

    # scored always covers what is burned
    for k in range(chain.length):
        assert not (chain.y[k] & ~chain.scored[k]).any()

    # the last label of the fire is the fire's final footprint, so the chain's
    # final truth equals the seed joined with every label, shifted into the
    # start window (the last sample's own window is 60 rows north of it)
    from wildfire_simulator.real_spread import shift_window

    rows = real_store.steps(FIRE_B)
    z = np.load(real_store.root / "samples" / f"{FIRE_B}.npz")
    frame = (int(rows.row0[0]), int(rows.col0[0]))
    burned = chain.perim0.copy()
    for j in range(chain.length):
        origin = (int(rows.row0[j]), int(rows.col0[j]))
        burned = burned | shift_window(z["y"][j] > 0, origin, frame, False)
    assert np.array_equal(chain.y[-1], burned)
    assert chain.y[-1].sum() > np.count_nonzero(z["y"][-1])


def test_chain_truth_matches_a_reference_reimplementation(real_store):
    """Independent re-derivation of ROLLOUT.md's chain truth and scoring.

    Every sample is cut from its own window, so both the cumulative burned
    truth and the scored cells have to be re-built here from the tables and the
    raw labels and compared against what the store produced.
    """
    from wildfire_simulator.real_spread import shift_window

    for fire_id, start in ((FIRE_A, 0), (FIRE_B, 0)):
        rows = real_store.steps(fire_id)
        z = np.load(real_store.root / "samples" / f"{fire_id}.npz")
        y = z["y"]
        origin = [(int(rows.row0[i]), int(rows.col0[i])) for i in range(len(rows))]
        frame = origin[start]
        chain = real_store.build_chain(fire_id, start, chain_len=None)

        burned = chain.perim0.copy()
        for k in range(chain.length):
            j = start + k
            inside = shift_window(np.ones((135, 135), bool), origin[j], frame, False)
            burned = burned | shift_window(y[j] > 0, origin[j], frame, False)
            assert np.array_equal(chain.y[k], burned), f"{fire_id} truth step {k}"
            assert np.array_equal(chain.scored[k], inside | burned), \
                f"{fire_id} scored step {k}"


def test_uncovered_cells_are_nan_inputs_and_unscored(real_store):
    """Fire B's third window jumps 60 rows; the row band it leaves behind is
    NaN in the inputs and excluded from scoring (unless already burned)."""
    from wildfire_simulator.real_spread import shift_window

    chain = build_full_life_chain(real_store, FIRE_B)
    rows = real_store.steps(FIRE_B)
    row0, row2 = int(rows.row0[0]), int(rows.row0[2])
    drift = row2 - row0
    assert drift == 60
    frame = (row0, int(rows.col0[0]))
    uncovered = ~shift_window(np.ones((135, 135), bool), (row2, int(rows.col0[2])),
                              frame, False)
    assert uncovered[:drift].all()          # the band the drift left behind

    elevation = real_store.channels.index("elev")
    assert not np.isnan(chain.x[:, elevation]).any()   # static: start window

    u10 = real_store.channels.index("u10")
    assert np.isnan(chain.x[2, u10, :drift]).all()
    assert not np.isnan(chain.x[2, u10, drift + 1:]).any()

    # scoring: never uncovered-and-unburned, always covers the truth
    assert not (chain.scored[2] & uncovered & ~chain.y[2]).any()
    assert chain.scored[2][chain.y[2]].all()
    first_burned = int(np.flatnonzero(chain.y[2].any(axis=1))[0])
    assert not chain.scored[2, :first_burned].any()    # nothing burned out there
    # burned cells *inside* the uncovered band are still scored (ROLLOUT.md)
    assert uncovered[first_burned:drift].all()
    assert chain.scored[2, first_burned:drift][
        chain.y[2, first_burned:drift]].all()


def test_static_channels_come_from_the_start_window(real_store):
    chain = real_store.build_chain(FIRE_A, 0, chain_len=None)
    elev = real_store.channels.index("elev")
    slope = real_store.channels.index("slope_mean")
    X, _ = real_store._arrays(FIRE_A)
    assert np.allclose(chain.x[:, elev], X[0, :, :, 7])
    assert np.allclose(chain.x[:, slope], X[0, :, :, 8])
    # the fixture varies elev per sample, so this is not a tautology
    assert not np.allclose(X[0, :, :, 7], X[1, :, :, 7])


def test_satellite_channels_dropped_after_the_first_step(real_store):
    """frp / n_detections / new_fire_pixels are observations at the input time:
    a forecast only has them at the rollout's first step."""
    chain = real_store.build_chain(FIRE_A, 1, chain_len=None)
    for name in ("frp", "n_detections", "new_fire_pixels"):
        i = real_store.channels.index(name)
        assert (chain.x[0, i] > 0).any()
        assert (chain.x[1:, i] == 0).all()


def test_step_times_follow_the_sample_cadence(real_store):
    chain = build_full_life_chain(real_store, FIRE_A)   # starts at the real fire
    assert chain.start == 1
    assert np.allclose(chain.t_hours, [0.0, 12.0, 24.0])
    assert np.allclose(chain.extent_hours, [12.0, 24.0, 36.0])
    from_zero = real_store.build_chain(FIRE_A, 0, chain_len=None)
    assert np.allclose(from_zero.t_hours, [0.0, 12.0, 24.0, 36.0])


def test_build_chain_clamps_to_the_fire_end(real_store):
    chain = real_store.build_chain(FIRE_A, 2, chain_len=12)
    assert chain.length == 2
    assert real_store.build_chain(FIRE_A, 3, chain_len=12).length == 1
    with pytest.raises(RealDataError, match="outside fire"):
        real_store.build_chain(FIRE_A, 4)


def test_window_size_is_the_products_window(real_store):
    chain = build_full_life_chain(real_store, FIRE_B)
    assert chain.x.shape[-2:] == (135, 135)
    assert chain.x.shape[1] == len(real_store.channels)


# ---------------------------------------------------------------------------
# dataset / collate / sampling
# ---------------------------------------------------------------------------
def test_dataset_normalizes_and_fills_nan(real_store):
    chains = [(FIRE_B, 0, None)]
    ds = RealChainDataset(real_store, chains)
    item = ds[0]
    x = item["x"]
    assert x.shape == (3, 11, 135, 135)
    assert x.dtype == torch.float32
    assert not torch.isnan(x).any()                      # NaN -> channel mean
    # perimeter channel normalizes to the raw 0/1 mask (mean 0, std 1 here)
    assert set(np.unique(x[0, 0].numpy()).tolist()) <= {0.0, 1.0}
    # the row band fire B's third window leaves behind is mean-filled (0)
    u10 = real_store.channels.index("u10")
    drift = int(real_store.steps(FIRE_B).row0[2]) - int(real_store.steps(FIRE_B).row0[0])
    assert (x[2, u10, :drift] == 0).all()
    assert item["y"].dtype == torch.float32
    assert item["scored"].dtype == torch.bool
    assert torch.allclose(item["t_hours"], torch.tensor([0.0, 12.0, 24.0]))
    assert item["fire_id"] == FIRE_B and item["start"] == 0


def test_collate_pads_shorter_chains(real_store):
    ds = RealChainDataset(real_store, [(FIRE_A, 0, 4), (FIRE_B, 0, 3)])
    batch = collate_chains([ds[0], ds[1]])
    assert batch["x"].shape == (2, 4, 11, 135, 135)
    assert batch["valid_step"].tolist() == [[True] * 4, [True] * 3 + [False]]
    assert batch["perim0"].shape == (2, 135, 135)
    assert batch["fire_id"] == [FIRE_A, FIRE_B]
    # the padded step is all zeros and unscored
    assert (batch["x"][1, 3] == 0).all()
    assert not batch["valid_step"][1, 3]


def test_fire_chain_cuts_tile_the_fire_and_move_per_epoch(real_store):
    """Back-to-back cuts cover every real sample exactly once, and the cut
    starts move when the rng does."""
    for fire_id, start0, n in ((FIRE_A, 1, 4), (FIRE_B, 0, 3)):
        cuts = fire_chain_cuts(n, start0, 2, random.Random("x"),
                               extra_chains=False)
        covered = [s for start, length in cuts for s in range(start, start + length)]
        # ROLLOUT.md: back-to-back chains from a random start to the fire's end,
        # so the samples before that start are simply not cut this epoch.
        assert cuts[0][0] >= start0 and cuts[0][0] < n
        assert sorted(covered) == list(range(cuts[0][0], n))
        assert all(length <= 2 for _, length in cuts)
        assert all(start + length <= n for start, length in cuts)
        moved = {tuple(fire_chain_cuts(n, start0, 2, random.Random(str(e)),
                                       extra_chains=False))
                 for e in range(20)}
        assert len(moved) > 1                              # cut points move
        assert fire_chain_cuts(n, start0, 2, random.Random("x"),
                               extra_chains=False) == cuts

    # with the extras on, the back-to-back tail is still tiled and every cut
    # starts at the fire's first real sample, the random start, or a growing step
    active = [n - 1]
    for epoch in range(6):
        cuts = fire_chain_cuts(n, start0, 2, random.Random(str(epoch)),
                               active_steps=active)
        covered = {s for start, length in cuts for s in range(start, start + length)}
        assert covered >= set(range(cuts[0][0], n))
        assert all(start >= start0 for start, _ in cuts)
        assert all(start in {start0, *active}
                   or start >= cuts[0][0] for start, _ in cuts)

    # a fire with nothing but placeholder samples yields no chain
    assert fire_chain_cuts(3, 3, 2, random.Random(0)) == []


def test_fire_chain_extras_respect_the_first_fire_sample(real_store):
    """The extra "first sample" chain must start at the real fire, not at the
    placeholder."""
    found = False
    for epoch in range(30):
        cuts = fire_chain_cuts(real_store.n_samples(FIRE_A),
                               real_store.first_fire_step(FIRE_A), 12,
                               random.Random(epoch), active_steps=[1, 2])
        for start, _ in cuts:
            assert start >= real_store.first_fire_step(FIRE_A)
            found = found or start == real_store.first_fire_step(FIRE_A)
    assert found


def test_streaming_dataset_yields_exactly_the_planned_cuts(real_store):
    """The streamed chains are the epoch's cuts (nothing dropped or extra), and
    the back-to-back part of them tiles every sample from its random start to
    the end of the fire."""
    ds = StreamingChainDataset(real_store, [FIRE_A, FIRE_B], chain_len=2, seed=5,
                               epoch=0, transform=real_store.transform())
    items = list(ds)                                   # single worker
    assert items
    assert all(torch.isfinite(it["x"]).all() for it in items)

    # replay the dataset's own rng stream (single worker -> wid 0)
    rng = random.Random(f"{ds.seed}:{ds.epoch}:0")
    expected, tiling = {}, {}
    for fire_id in ds.plan[0]:
        cuts = ds.cuts_for(fire_id, rng)      # back-to-back cuts, extras last
        tiling[fire_id] = list(cuts)
        rng.shuffle(cuts)                     # within-fire order, as streamed
        expected[fire_id] = cuts

    got = {}
    for item in items:
        got.setdefault(item["fire_id"], []).append(
            (item["start"], item["x"].shape[0]))
    assert got == expected

    for fire_id, cuts in tiling.items():
        n = real_store.n_samples(fire_id)
        start0 = real_store.first_fire_step(fire_id)
        assert all(s >= start0 for s, _ in cuts)
        assert all(s + length <= n for s, length in cuts)
        covered = {s for s, length in cuts for s in range(s, s + length)}
        assert covered == set(range(cuts[0][0], n))   # tiled, no gap, no overlap
        assert set(range(cuts[0][0], n)) <= {
            s for it in items if it["fire_id"] == fire_id
            for s in range(it["start"], it["start"] + it["x"].shape[0])}


def test_streaming_plan_is_balanced_and_deterministic(real_store):
    fires = [FIRE_A, FIRE_B]
    a = StreamingChainDataset(real_store, fires, 2, seed=1, epoch=0, num_workers=2)
    b = StreamingChainDataset(real_store, fires, 2, seed=1, epoch=0, num_workers=2)
    assert a.plan == b.plan
    assert sorted(f for b in a.plan for f in b) == sorted(fires)   # disjoint cover
    # the heavy fire (B) lands in the lighter bin: A has 4 samples, B has 3
    assert any(FIRE_B in bin_ for bin_ in a.plan)


def test_streaming_epochs_move_their_cuts(real_store):
    """A new epoch re-cuts the fires: the cut starts must not be frozen."""
    cuts = set()
    for epoch in range(6):
        ds = StreamingChainDataset(real_store, [FIRE_A, FIRE_B], 2, seed=3,
                                   epoch=epoch)
        cuts.add(tuple((it["fire_id"], it["start"]) for it in ds))
    assert len(cuts) > 1


def test_streaming_dataset_rejects_a_worker_count_mismatch(real_store):
    ds = StreamingChainDataset(real_store, [FIRE_A], 2, seed=1, num_workers=4)
    with pytest.raises(RuntimeError, match="planned for 4 workers"):
        list(ds)                       # iterated outside a worker of that size


def test_streaming_dataloader_round_trip(real_store):
    from torch.utils.data import DataLoader

    ds = StreamingChainDataset(real_store, [FIRE_A, FIRE_B], chain_len=2, seed=11,
                               epoch=0, transform=real_store.transform())
    loader = DataLoader(ds, batch_size=3, collate_fn=collate_chains)
    batches = list(loader)
    assert batches
    batch = batches[0]
    assert batch["x"].ndim == 5
    assert batch["x"].shape[0] <= 3
    assert torch.isfinite(batch["x"]).all()
    total = sum(b["x"].shape[0] for b in batches)
    assert total == len(list(ds))                      # nothing dropped


def test_val_chains_are_deterministic_and_capped(real_store):
    chains = build_val_chains(real_store, [FIRE_A, FIRE_B], chain_len=2)
    assert chains == [(FIRE_A, 1, 2), (FIRE_B, 0, 2)]
    assert build_val_chains(real_store, [FIRE_A], chain_len=99) == [(FIRE_A, 1, 3)]


# ---------------------------------------------------------------------------
# decompressed mirror (scripts/prepare_real_cache.py)
# ---------------------------------------------------------------------------
def _build_cache(real_root, out, dtype="float32", fires=None, overwrite=False):
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import prepare_real_cache as prc

    out.mkdir(parents=True, exist_ok=True)
    fires = list(real_root.glob("samples/*.npz")) if fires is None else fires
    jobs = [(real_root, out, p.stem, np.dtype(dtype), overwrite) for p in fires]
    for job in jobs:
        prc.convert_fire(job)
    manifest = {
        "source_root": str(real_root),
        "dtype": np.dtype(dtype).name,
        "fires": sorted(p.stem for p in fires),
    }
    (out / "manifest.json").write_text(json.dumps(manifest))
    return out


def test_decompressed_cache_is_a_drop_in_for_the_npz(real_root, tmp_path):
    """Chains from the mirror must be identical to chains from the npz."""
    cache = _build_cache(real_root, tmp_path / "cache")
    cached = RealDataStore(real_root, norm_stats=real_root / NORM,
                           arrays_cache=cache)
    plain = RealDataStore(real_root, norm_stats=real_root / NORM)
    assert "decompressed cache" in cached.describe()
    assert cached.channels == plain.channels

    for fire_id, start, length in ((FIRE_A, 0, None), (FIRE_A, 1, 2),
                                   (FIRE_B, 0, 3)):
        a = cached.build_chain(fire_id, start, chain_len=length)
        b = plain.build_chain(fire_id, start, chain_len=length)
        assert np.array_equal(a.x, b.x, equal_nan=True)
        assert np.array_equal(a.y, b.y)
        assert np.array_equal(a.scored, b.scored)
        assert np.array_equal(a.perim0, b.perim0)
        assert np.array_equal(a.t_hours, b.t_hours)
    assert cached.first_fire_step(FIRE_A) == plain.first_fire_step(FIRE_A) == 1

    # and the dataset built on the mirror itemizes identically
    ds_cached = RealChainDataset(cached, [(FIRE_B, 0, None)],
                                 transform=cached.transform())
    ds_plain = RealChainDataset(plain, [(FIRE_B, 0, None)],
                                transform=plain.transform())
    for key in ("x", "y", "scored", "perim0", "t_hours"):
        assert torch.equal(ds_cached[0][key], ds_plain[0][key]), key


def test_decompressed_cache_fails_loudly_on_a_foreign_or_mixed_cache(real_root,
                                                                    tmp_path):
    cache = _build_cache(real_root, tmp_path / "cache")
    # a manifest pointing at another dataset
    (cache / "manifest.json").write_text(json.dumps(
        {"source_root": "/somewhere/else", "dtype": "float32"}))
    with pytest.raises(RealDataError, match="was built from"):
        RealDataStore(real_root, norm_stats=real_root / NORM, arrays_cache=cache)

    # a fire whose dtype disagrees with the manifest
    _build_cache(real_root, cache, dtype="float32", overwrite=True)
    bad = np.load(cache / f"{FIRE_A}.X.npy")
    np.save(cache / f"{FIRE_A}.X.npy", bad.astype(np.float16))
    store = RealDataStore(real_root, norm_stats=real_root / NORM,
                          arrays_cache=cache)
    with pytest.raises(RealDataError, match="mixed cache"):
        store._arrays(FIRE_A)

    # a directory that is not a cache at all
    with pytest.raises(RealDataError, match="no manifest"):
        RealDataStore(real_root, norm_stats=real_root / NORM,
                      arrays_cache=tmp_path / "empty")


def test_decompressed_cache_detects_a_stale_mirror(real_root, tmp_path):
    """If the product is rebuilt in place, the mirror must refuse to be used."""
    import shutil

    root = tmp_path / "root"
    shutil.copytree(real_root, root)
    cache = _build_cache(root, tmp_path / "cache")
    manifest = json.loads((cache / "manifest.json").read_text())
    manifest["source_sizes"] = {
        p.stem: [int((root / "samples" / f"{p.stem}.npz").stat().st_size),
                 int((root / "samples" / f"{p.stem}.npz").stat().st_mtime_ns)]
        for p in (root / "samples").glob("*.npz")
    }
    (cache / "manifest.json").write_text(json.dumps(manifest))

    store = RealDataStore(root, norm_stats=root / NORM, arrays_cache=cache)
    assert store.build_chain(FIRE_A, 1, chain_len=2) is not None   # fresh: fine

    # rebuild the source npz in place (same content, new fingerprint)
    src = root / "samples" / f"{FIRE_A}.npz"
    with np.load(src) as z:
        np.savez(src, X=z["X"], y=z["y"])
    store = RealDataStore(root, norm_stats=root / NORM, arrays_cache=cache)
    with pytest.raises(RealDataError, match="stale mirror"):
        store.build_chain(FIRE_A, 1, chain_len=2)


def test_incremental_conversion_keeps_other_fires_fingerprints(real_root, tmp_path):
    """Converting a second split must not drop the first split's fingerprints."""
    import sys as _sys
    _sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
    import prepare_real_cache as prc

    out = tmp_path / "cache"
    out.mkdir()
    for fire_id in (FIRE_A, FIRE_B):
        prc.convert_fire((real_root, out, fire_id, np.dtype("float32"), False))
    first = {
        p.name.split(".")[0]: [int(p.stat().st_size), int(p.stat().st_mtime_ns)]
        for p in (real_root / "samples").glob("*.npz")
    }
    (out / "manifest.json").write_text(json.dumps(
        {"source_root": str(real_root), "dtype": "float32",
         "source_sizes": first, "fires": [FIRE_A, FIRE_B]}))

    # a second run over only FIRE_B (as converting another split would be)
    rc = prc.main(["--real-data", str(real_root), "--out", str(out),
                   "--splits", "val"])            # val == [FIRE_B]
    assert rc == 0
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["fires"] == [FIRE_A, FIRE_B]
    assert set(manifest["source_sizes"]) == {FIRE_A, FIRE_B}


def test_store_without_norm_stats_is_table_only(real_root):
    """Conversions must not require training statistics; a transform still must."""
    store = RealDataStore(real_root, load_norm_stats=False)
    assert store.n_samples(FIRE_A) == 4
    assert store.first_fire_step(FIRE_A) == 1
    assert "(none)" in store.describe()
    with pytest.raises(RealDataError, match="no normalization statistics"):
        store.transform()


def test_decompressed_cache_reports_an_unconverted_split(real_root, tmp_path):
    cache = _build_cache(real_root, tmp_path / "cache",
                         fires=[real_root / "samples" / f"{FIRE_A}.npz"])
    store = RealDataStore(real_root, norm_stats=real_root / NORM,
                          arrays_cache=cache)
    assert store.n_samples(FIRE_A) == 4          # tables come from the root
    with pytest.raises(RealDataError, match="not converted"):
        store._arrays(FIRE_B)
