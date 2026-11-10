"""Real-data (v5 wildfires) chain access for the surrogate baseline.

This is the real-data counterpart of ``dataloader``/``datasets``: instead of
cropping a single MTT trial around an ignition and slicing its arrival-time
raster, it reads the 12-hour chain product documented in
``<root>/README.md`` and ``<root>/ROLLOUT.md``.

Data layout
-----------
One ``samples/<fire_id>.npz`` per fire with

* ``X`` float32 ``(n, 135, 135, C)`` — the inputs at each 12 h step,
* ``y`` uint8 ``(n, 135, 135)`` — the perimeter 12 h later (``y[k]`` is the
  *whole* perimeter at step ``k + 1``, not just the new burn),

plus ``samples.csv`` (one row per sample: ``step``, ``t_input``, ``t_label``,
``row0``, ``col0``, ``lat0``, ``lon0``, ``growth_px``, ``category``),
``fires.csv`` (one row per fire), ``channels.json`` (channel order) and a frozen
split (``split_frozen_*.json``, keys ``train``/``val``/``test``).

Chains
------
A chain is a contiguous run of samples laid out in its **start** sample's
135x135 window. Later samples are cut from their own window, so their inputs and
labels are shifted into the start window (``row0``/``col0`` per sample make the
shift exact). Cells a later window does not cover become NaN inputs; the chain
also carries the label-coverage mask used to exclude them from scoring. This is
verbatim the convention in ``ROLLOUT.md`` (whose ``shift_window`` this module
reuses, so there is a single implementation of the shift).

Per step ``k`` (sample ``j = start + k``) the chain stores

* ``x[k, 0]`` — the fire state *before* the step, i.e. the cumulative burned
  mask at ``t_input(j)`` (``perim0`` for ``k = 0``). The trainer overwrites this
  channel with the model's own mask when scheduled sampling draws
  "self-driven".
* ``x[k, 1:]`` — that step's own meteorology/terrain/time (this is what makes
  the future meteorology available to every rollout step). Channels 1..3
  (frp, n_detections, new_fire_pixels) are satellite observations *at the input
  time*, so they are zeroed for ``k >= 1`` (a forecast would not have them).
* ``y[k]`` — cumulative burned truth at ``t_label(j)`` (burned land does not
  unburn: ``perim0`` joined with every label up to ``k``).
* ``scored[k]`` — cells the real data can score: step ``j``'s own window,
  joined with everything already burned.
* ``t_hours[k]`` — hours from the chain's start input time to step ``k``'s
  input time (the model's time channel is ``t_hours / 24``; 24 h is the
  surrogate's time unit, see ``finetune.trainer.HOURS_PER_T_UNIT``).

Static channels (``cadence == "static"`` in ``channels.json``) are identical in
every window of a fire — one landscape per fire — so they are always read from
the start window.
"""

from __future__ import annotations

import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from wildfire_simulator.real_spread import shift_window
from finetune.transforms import ZScorePerChannel

WIN = 135                  # window edge (cells)
CELL_DEG = 0.003          # cell size (degrees)
CELL_M = 330.0            # nominal cell size (m), README: "about 330 m"
# Pre-fire samples carry a small constant placeholder perimeter (~15 cells);
# real detections are far larger. Same cutoff as real_spread.first_fire_step.
ARTIFACT_THRESHOLD = 20
# The z-score stats the dataset was built with (mean/std/log per channel).
DEFAULT_NORM_CACHE = "norm_cache/04c547cd280a3e4d.json"
_STATIC_CADENCE = "static"
# Satellite observations at the input time; unavailable to a forecast after
# the first step (channels 1..3 of channels.json: frp, n_detections,
# new_fire_pixels). Kept by name so a channel reorder cannot silently shift it.
_SATELLITE_OBS_CHANNELS = ("frp", "n_detections", "new_fire_pixels")
# The one channel the product ships without normalization statistics. It is also
# a *nominal* FBFM40 fuel-model code (32 distinct codes, 91-189, in a 6-fire
# sample), so z-scoring it would invent a metric distance between classes. The
# seven `fbfm_*` composition channels carry its group (the code's group matches
# argmax of those fractions in 96.7 % of 12.1 M sampled cells) but 32 codes
# collapse into 7 groups, so the within-group detail is not in the default input.
# Dropped for those two reasons; pass it explicitly via `channels=` with
# statistics (or encode it properly, e.g. one-hot) if that detail is wanted.
_DEFAULT_EXCLUDED = ("fbfm_mode",)

_DATA_COLUMNS = (
    "fire_id", "step", "t_input", "t_label", "growth_px", "category",
    "row0", "col0", "lat0", "lon0",
)


class RealDataError(ValueError):
    """Raised for malformed real-data roots or unresolvable fire references."""


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _parse_time(value) -> datetime:
    """Parse a real-data timestamp (fire *local* wall clock) into a datetime.

    Differences between these timestamps are what the chains need, and the
    dataset's cadence is exact in local time, so no timezone conversion is
    applied anywhere in the real-data baseline.
    """
    if isinstance(value, datetime):
        return value
    s = str(value).strip()
    return datetime.fromisoformat(s.replace(" ", "T", 1))


def _fire_year(fire_id) -> int | None:
    """Year of a v5 fire id (``V5_<year>_...`` / ``V5NRT_<year>_...``)."""
    parts = str(fire_id).split("_")
    try:
        return int(parts[1]) if len(parts) > 1 else None
    except ValueError:
        return None


def read_channels(root) -> list[dict]:
    """Parse ``channels.json`` (ordered channel name/product/cadence records)."""
    with open(Path(root) / "channels.json") as f:
        channels = json.load(f)
    if not channels:
        raise RealDataError(f"{root}/channels.json is empty")
    return channels


# ---------------------------------------------------------------------------
# chain containers
# ---------------------------------------------------------------------------
@dataclass
class RealChain:
    """One contiguous run of real-data samples, laid out in its start window.

    ``x`` is raw (unnormalized) and NaN where a later window does not cover a
    cell; ``RealChainDataset`` normalizes and fills it.
    """

    fire_id: str
    start: int
    x: np.ndarray            # (L, C, WIN, WIN) float32, raw, NaN where uncovered
    y: np.ndarray            # (L, WIN, WIN) bool, cumulative burned truth
    scored: np.ndarray       # (L, WIN, WIN) bool, cells the real data can score
    perim0: np.ndarray       # (WIN, WIN) bool, observed perimeter at the start
    t_hours: np.ndarray      # (L,) float, hours from start input to step input
    extent_hours: np.ndarray  # (L,) float, hours from start input to step label
    input_times: list        # (L,) datetime, local wall clock
    label_times: list        # (L,) datetime, local wall clock
    frame_row0: int
    frame_col0: int

    @property
    def length(self) -> int:
        return self.x.shape[0]


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------
class RealDataStore:
    """Read-only, cached accessor for one v5 real-data root.

    Args:
        root: real-data root (``fires.csv``, ``samples.csv``, ``samples/``,
            ``channels.json``, a frozen split, ``norm_cache/``).
        channels: channel *names* to load, in model-input order. ``None`` uses
            every channel in ``channels.json`` except ``_DEFAULT_EXCLUDED``.
        norm_stats: path to a z-score stats JSON (``{name: {mean, std, log}}``).
            ``None`` uses ``<root>/norm_cache/04c547cd280a3e4d.json``.
        static_from_start: read ``cadence == "static"`` channels from the
            chain's start window instead of shifting them (identical values;
            avoids mean-filling terrain a later window does not cover).
        cache_bytes: byte budget for the decoded per-fire ``X``/``y`` arrays
            shared by all chains of a fire (a 12-step chain only needs 12 of a
            fire's samples, so caching pays off across an epoch). Unused when
            ``arrays_cache`` is given.
        split_path: explicit frozen-split JSON; ``None`` globs
            ``split_frozen_*.json`` in the root.
        load_norm_stats: load the z-score statistics eagerly (default). Set
            ``False`` for table-only users that never build a transform — the
            conversion script, for one, must not require training stats to
            decompress data; ``transform()`` then raises if it is called.
        arrays_cache: directory holding a decompressed ``.npy`` mirror of
            ``samples/`` (``scripts/prepare_real_cache.py``). The product's npz
            members are DEFLATE-compressed, so every access to a fire normally
            costs a full inflate of it; the mirror is memory-mapped instead.
            Its manifest pins the source root, dtype and per-fire source
            fingerprints, so a foreign, mixed or stale mirror fails loudly.
    """

    def __init__(self, root, channels=None, norm_stats=None,
                 static_from_start=True, cache_bytes=600 << 20,
                 split_path=None, arrays_cache=None, load_norm_stats=True):
        self.root = Path(root)
        if not self.root.is_dir():
            raise RealDataError(f"real-data root not found: {self.root}")
        # Optional decompressed mirror of samples/ (scripts/prepare_real_cache.py).
        # The product's npz members are DEFLATE-compressed, so every fire costs
        # a full inflate per epoch; the mirror is a plain .npy per fire that is
        # memory-mapped instead.
        self.arrays_cache = Path(arrays_cache) if arrays_cache else None
        if self.arrays_cache is not None:
            self._load_arrays_manifest()

        self.channels_json = read_channels(self.root)
        self.channel_names_all = [c["name"] for c in self.channels_json]
        self.channel_cadence = {c["name"]: c.get("cadence")
                                for c in self.channels_json}

        if channels is None:
            channels = [n for n in self.channel_names_all
                        if n not in _DEFAULT_EXCLUDED]
        unknown = [n for n in channels if n not in self.channel_names_all]
        if unknown:
            raise RealDataError(
                f"unknown channels {unknown}; available: {self.channel_names_all}")
        self.channels = list(channels)
        self.chan_idx = [self.channel_names_all.index(n) for n in self.channels]
        self.static = np.array(
            [self.channel_cadence[n] == _STATIC_CADENCE for n in self.channels],
            dtype=bool,
        )
        self.sat_obs = np.array(
            [n in _SATELLITE_OBS_CHANNELS for n in self.channels], dtype=bool)
        self.static_from_start = bool(static_from_start)

        self.mean = self.std = self.log = None
        self.norm_stats_path = None
        if load_norm_stats:
            self._load_norm_stats(norm_stats)
        self._load_tables(split_path)
        self._cache = OrderedDict()
        self._cache_bytes = int(cache_bytes)
        self._cache_used = 0
        self._first_fire_step = {}

    # -- setup -------------------------------------------------------------
    def _load_norm_stats(self, norm_stats):
        path = Path(norm_stats) if norm_stats else self.root / DEFAULT_NORM_CACHE
        if not path.is_file():
            raise RealDataError(
                f"normalization stats not found: {path}. Pass --norm-stats "
                f"pointing at a JSON with {{channel: {{mean, std, log}}}}."
            )
        with open(path) as f:
            stats = json.load(f)
        missing = [n for n in self.channels if n not in stats]
        if missing:
            raise RealDataError(
                f"no normalization stats for channels {missing} in {path}. "
                f"Drop them from the channel list or supply stats."
            )
        self.norm_stats_path = str(path)
        self.mean = np.array([stats[n]["mean"] for n in self.channels], np.float32)
        self.std = np.array([stats[n]["std"] for n in self.channels], np.float32)
        self.log = np.array([bool(stats[n].get("log", False))
                             for n in self.channels], dtype=bool)

    def _load_arrays_manifest(self):
        """Validate the decompressed mirror against this store's expectations.

        The mirror keeps every channel of the product, so a channel subset stays
        a training-time choice and the array layout is identical to the npz.
        What must match is the source root (a mirror of a different dataset
        would silently train on the wrong fires) and the array dtype.
        """
        manifest_path = self.arrays_cache / "manifest.json"
        if not manifest_path.is_file():
            raise RealDataError(
                f"{self.arrays_cache} is not a decompressed cache: no "
                f"manifest.json (build it with scripts/prepare_real_cache.py)"
            )
        with open(manifest_path) as f:
            manifest = json.load(f)
        if Path(manifest.get("source_root", "")) != self.root:
            raise RealDataError(
                f"cache {self.arrays_cache} was built from "
                f"{manifest.get('source_root')!r}, not {str(self.root)!r}"
            )
        self.arrays_manifest = manifest

    def _cached_arrays(self, fire_id):
        """``(X, y)`` from the decompressed mirror, memory-mapped."""
        x_path = self.arrays_cache / f"{fire_id}.X.npy"
        y_path = self.arrays_cache / f"{fire_id}.y.npy"
        for path in (x_path, y_path):
            if not path.is_file():
                raise RealDataError(
                    f"missing {path}: this split was not converted (rerun "
                    f"scripts/prepare_real_cache.py for it)"
                )
        X, y = (np.load(x_path, mmap_mode="r"), np.load(y_path, mmap_mode="r"))
        expected = self.arrays_manifest.get("dtype")
        if expected is not None and X.dtype.name != expected:
            raise RealDataError(
                f"{x_path} is {X.dtype.name} but the cache manifest says "
                f"{expected}; a mixed cache is not supported"
            )
        self._check_source_unchanged(fire_id)
        return X, y

    def _check_source_unchanged(self, fire_id):
        """Fail loudly if the product changed under a mirror built from it."""
        recorded = (self.arrays_manifest.get("source_sizes") or {}).get(fire_id)
        if not recorded:
            return
        src = self.root / "samples" / f"{fire_id}.npz"
        if not src.is_file():
            raise RealDataError(
                f"{src} is gone but {self.arrays_cache} was built from it; "
                f"re-run scripts/prepare_real_cache.py"
            )
        stat = src.stat()
        if [int(stat.st_size), int(stat.st_mtime_ns)] != list(recorded):
            raise RealDataError(
                f"{src} changed since {self.arrays_cache} was built (stale "
                f"mirror), so training would use old samples. Rebuild with "
                f"scripts/prepare_real_cache.py --overwrite"
            )

    def _load_tables(self, split_path):
        fires_path = self.root / "fires.csv"
        samples_path = self.root / "samples.csv"
        for p in (fires_path, samples_path):
            if not p.is_file():
                raise RealDataError(f"missing {p}")
        self.fires = pd.read_csv(fires_path)
        samples = pd.read_csv(samples_path, usecols=list(_DATA_COLUMNS))
        samples = samples.sort_values(["fire_id", "step"])
        # fire_id -> step-ordered rows, so a window lookup is a plain iloc.
        self._samples = {fid: g.reset_index(drop=True)
                         for fid, g in samples.groupby("fire_id", sort=False)}
        # Plain dicts, not pandas namedtuples: the store is pickled to DataLoader
        # workers (forkserver/spawn), and itertuples' dynamic class is not
        # picklable.
        self._fire_row = {
            str(r["fire_id"]): dict(r) for _, r in self.fires.iterrows()
        }

        if split_path is None:
            candidates = sorted(self.root.glob("split_frozen_*.json"))
            split_path = candidates[0] if candidates else None
        self.split_path = str(split_path) if split_path else None
        self.split = {}
        if split_path is not None:
            with open(split_path) as f:
                self.split = {k: [str(v) for v in vs]
                              for k, vs in json.load(f).items()}

    # -- queries -----------------------------------------------------------
    def split_fires(self, key) -> list:
        """Fire ids of a frozen split (``train``/``val``/``test``)."""
        if key not in self.split:
            raise RealDataError(
                f"split {key!r} not present in {self.split_path}; "
                f"have {sorted(self.split)}"
            )
        return list(self.split[key])

    def fire_row(self, fire_id):
        """Metadata of one fire as a plain dict (``fires.csv`` columns)."""
        fire_id = str(fire_id)
        if fire_id not in self._fire_row:
            raise RealDataError(f"unknown fire_id {fire_id!r}")
        return self._fire_row[fire_id]

    def __getstate__(self):
        """Drop the decoded-sample cache when pickling to DataLoader workers.

        The cache can hold hundreds of MB of a fire's samples; the per-fire
        ``first_fire_step`` scan is kept because it is small and expensive to
        redo (it reads the sample files).
        """
        state = dict(self.__dict__)
        state["_cache"] = OrderedDict()
        state["_cache_used"] = 0
        return state

    def resolve_fire(self, name=None, year=None, fire_id=None) -> str:
        """Resolve a fire reference to a ``fire_id``.

        ``fire_id`` short-circuits matching. Otherwise ``name`` (matched
        case-insensitively) and optionally ``year`` (from the fire id) select a
        fire; an ambiguous match lists the candidates instead of guessing.
        """
        if fire_id is not None:
            self.fire_row(fire_id)  # raises when unknown
            return str(fire_id)
        if name is None:
            raise RealDataError("resolve_fire needs a fire_id or a name")
        want = str(name).strip().upper()
        match = self.fires[
            self.fires["name"].astype(str).str.strip().str.upper() == want
        ]
        if year is not None:
            match = match[match["fire_id"].map(_fire_year) == int(year)]
        if len(match) == 0:
            raise RealDataError(f"no real fire named {name!r} year {year}")
        if len(match) > 1:
            cands = match[["fire_id", "name", "state", "county"]].to_string(index=False)
            raise RealDataError(
                f"ambiguous fire {name!r} year {year}; pass --fire-id\n{cands}")
        return str(match.iloc[0]["fire_id"])

    def steps(self, fire_id) -> pd.DataFrame:
        """Step-ordered ``samples.csv`` rows for one fire."""
        fire_id = str(fire_id)
        if fire_id not in self._samples:
            raise RealDataError(f"no samples.csv rows for fire {fire_id!r}")
        return self._samples[fire_id]

    def n_samples(self, fire_id) -> int:
        """Number of 12 h samples for a fire (from ``samples.csv``, no I/O).

        ``build_chain`` re-checks this against the ``.npz`` so a stale table
        fails loudly instead of silently misaligning chains.
        """
        return int(len(self.steps(fire_id)))

    def first_fire_step(self, fire_id) -> int:
        """First sample whose observed perimeter exceeds the pre-fire placeholder.

        Sample 0 of some fires carries a small constant FEDS placeholder
        (~15 cells) that is not real fire; a chain seeded there would train the
        model on a fake ignition.
        """
        fire_id = str(fire_id)
        if fire_id not in self._first_fire_step:
            self._first_fire_step[fire_id] = self._scan_first_fire_step(fire_id)
        return self._first_fire_step[fire_id]

    def _scan_first_fire_step(self, fire_id) -> int:
        # The product's arrays are DEFLATE-compressed, so this costs one
        # decompression. It uses the same cache ``build_chain`` does, so the
        # fire stays in memory for the chains that follow (which is why the
        # streaming loader can afford to call this once per fire per epoch).
        X = self._arrays(fire_id)[0]
        counts = (X[:, :, :, 0] > 0.5).sum(axis=(1, 2))
        idx = np.flatnonzero(counts > ARTIFACT_THRESHOLD)
        return int(idx[0]) if idx.size else 0

    # -- arrays ------------------------------------------------------------
    def _arrays(self, fire_id):
        """Decoded ``(X, y)`` for a fire.

        From the decompressed mirror (memory-mapped, so nothing is held in
        RAM) when one is configured, else a byte-bounded LRU of the npz
        content, which costs a full inflate per fire.
        """
        fire_id = str(fire_id)
        if self.arrays_cache is not None:
            return self._cached_arrays(fire_id)
        if fire_id in self._cache:
            self._cache.move_to_end(fire_id)
            return self._cache[fire_id]
        path = self.root / "samples" / f"{fire_id}.npz"
        if not path.is_file():
            raise RealDataError(f"missing sample file {path}")
        with np.load(path) as z:
            X, y = z["X"], z["y"]
        arrays = (X, y)
        nbytes = int(X.nbytes + y.nbytes)
        self._cache[fire_id] = arrays
        self._cache_used += nbytes
        while self._cache_used > self._cache_bytes and len(self._cache) > 1:
            _, (old_X, old_y) = self._cache.popitem(last=False)
            self._cache_used -= int(old_X.nbytes + old_y.nbytes)
            del old_X, old_y
        return arrays

    # -- chains ------------------------------------------------------------
    def build_chain(self, fire_id, start, chain_len=None) -> RealChain:
        """Build a chain starting at sample ``start``.

        ``chain_len`` counts *steps*; ``None`` runs to the fire's last sample.
        """
        fire_id = str(fire_id)
        X, y = self._arrays(fire_id)
        rows = self.steps(fire_id)
        n = int(X.shape[0])
        if not 0 <= start < n:
            raise RealDataError(
                f"start {start} outside fire {fire_id!r} ({n} samples)")
        L = n - start if chain_len is None else min(int(chain_len), n - start)
        if L <= 0:
            raise RealDataError(f"empty chain for {fire_id!r} at start {start}")

        idx = np.asarray(self.chan_idx)
        if len(rows) != n:
            raise RealDataError(
                f"samples.csv has {len(rows)} rows for {fire_id!r} but "
                f"{Path(self.root, 'samples', fire_id + '.npz')} holds {n} "
                f"samples; the tables and the sample files disagree."
            )
        static_local = (np.flatnonzero(self.static) if self.static_from_start
                        else np.zeros(0, dtype=int))
        dynamic_local = np.setdiff1d(np.arange(len(idx)), static_local)
        frame = (int(rows.row0[start]), int(rows.col0[start]))
        origin = [(int(rows.row0[i]), int(rows.col0[i])) for i in range(n)]

        x = np.full((L, len(idx), WIN, WIN), np.nan, dtype=np.float32)
        y_out = np.empty((L, WIN, WIN), dtype=bool)
        scored = np.empty((L, WIN, WIN), dtype=bool)

        # Static channels: one landscape per fire -> the start window covers
        # the whole frame, so no shift (and no mean-filled terrain) is needed.
        start_pack = X[start][:, :, idx]                      # (135, 135, C)
        if static_local.size:
            x[:, static_local] = np.moveaxis(
                start_pack[:, :, static_local], -1, 0)

        perim0 = X[start, :, :, 0] > 0.5
        burned = perim0.copy()
        ones = np.ones((WIN, WIN), dtype=bool)
        for k in range(L):
            j = start + k
            if dynamic_local.size:
                pack = X[j][:, :, idx[dynamic_local]]          # (135, 135, D)
                shifted = shift_window(
                    np.moveaxis(pack, -1, 0), origin[j], frame, np.nan)
                if k > 0 and self.sat_obs.any():
                    # satellite obs at the input time: not a forecast input
                    shifted[self.sat_obs[dynamic_local]] = 0.0
                x[k, dynamic_local] = shifted
            label = shift_window(y[j] > 0, origin[j], frame, False)
            burned = burned | label
            y_out[k] = burned
            scored[k] = shift_window(ones, origin[j], frame, False) | burned

        # Channel 0 is the fire state before the step: perim0 at k = 0, then
        # the cumulative truth of the previous step.
        x[0, 0] = perim0
        if L > 1:
            x[1:, 0] = y_out[:-1]

        t_input = [_parse_time(v) for v in rows.t_input.iloc[start:start + L]]
        t_label = [_parse_time(v) for v in rows.t_label.iloc[start:start + L]]
        t0 = t_input[0]
        t_hours = np.array([(t - t0).total_seconds() / 3600.0 for t in t_input],
                           dtype=np.float32)
        extent_hours = np.array(
            [(t - t0).total_seconds() / 3600.0 for t in t_label], dtype=np.float32)

        return RealChain(
            fire_id=fire_id, start=int(start), x=x, y=y_out, scored=scored,
            perim0=perim0, t_hours=t_hours, extent_hours=extent_hours,
            input_times=t_input, label_times=t_label,
            frame_row0=frame[0], frame_col0=frame[1],
        )

    def transform(self) -> ZScorePerChannel:
        """Normalizer built from this store's channel stats."""
        if self.mean is None:
            raise RealDataError(
                "this store was opened with load_norm_stats=False, so it has no "
                "normalization statistics to build a transform from"
            )
        return ZScorePerChannel(self.mean, self.std, self.log)

    def describe(self) -> str:
        """One-line summary for logs."""
        where = (f"decompressed cache {self.arrays_cache.name}"
                 if self.arrays_cache is not None else "npz (inflate per fire)")
        stats = (Path(self.norm_stats_path).name if self.norm_stats_path
                 else "(none)")
        return (f"real data {self.root.name}: {len(self.channels)} channels "
                f"({int(self.static.sum())} static), norm stats {stats}, "
                f"arrays: {where}")


# ---------------------------------------------------------------------------
# torch dataset
# ---------------------------------------------------------------------------
def _chain_item(store, chain, transform, fill_value):
    """One chain as training tensors (shared by both dataset flavours)."""
    x = torch.from_numpy(chain.x)                     # (L, C, H, W)
    if transform is not None:
        x = transform(x)
    # NaN -> fill value only after normalizing, so the fill is the channel mean
    # (0) rather than a raw-scale zero.
    x = torch.nan_to_num(x, nan=fill_value)
    return {
        "x": x,
        "y": torch.from_numpy(chain.y.astype(np.float32)),
        "scored": torch.from_numpy(chain.scored),
        "perim0": torch.from_numpy(chain.perim0.astype(np.float32)),
        "t_hours": torch.from_numpy(chain.t_hours),
        "extent_hours": torch.from_numpy(chain.extent_hours),
        "fire_id": chain.fire_id,
        "start": chain.start,
    }


def fire_chain_cuts(n_samples, first_fire_step, chain_len, rng,
                    extra_chains=True, active_steps=()) -> list:
    """One fire's chain cuts for one epoch (``ROLLOUT.md``'s recipe).

    Back-to-back ``chain_len``-step chains from a *random* start, so the cut
    points move from epoch to epoch; together they tile every sample from the
    fire's first real sample to its last, once. With ``extra_chains`` the fire
    also has a 50 % chance of one extra chain from its first real sample and a
    50 % chance of one from a step where the satellites saw growth.

    Args:
        n_samples: samples in the fire.
        first_fire_step: first sample that is real fire (not the FEDS placeholder).
        chain_len: steps per chain.
        rng: ``random.Random``; ``None`` means no random start and no extras.
        extra_chains: draw the two extra chains.
        active_steps: sample indices where the fire grew.

    Returns:
        List of ``(start_step, length)``.
    """
    if n_samples - first_fire_step < 1:
        return []
    cuts = []
    s = first_fire_step if rng is None else rng.randrange(first_fire_step, n_samples)
    while s < n_samples:
        cuts.append((int(s), min(chain_len, n_samples - s)))
        s += chain_len
    if rng is not None and extra_chains:
        if rng.random() < 0.5:
            cuts.append((int(first_fire_step),
                         min(chain_len, n_samples - first_fire_step)))
        active = list(active_steps)
        if active and rng.random() < 0.5:
            s = int(rng.choice(active))
            cuts.append((s, min(chain_len, n_samples - s)))
    return cuts


class StreamingChainDataset(IterableDataset):
    """Chains streamed fire by fire: one decompression per fire per worker.

    Why not the map-style :class:`RealChainDataset` for training: the product
    stores each fire's samples as a single DEFLATE-compressed ``.npy`` member
    inside its ``.npz``, so *any* access to a fire costs a full decompression of
    that fire. Indexing chains independently pays that cost again for every
    chain that lands in another worker or after cache eviction — several times
    a fire's size per epoch. Streaming assigns each fire to exactly one worker,
    decompresses it once, and yields all of that epoch's chains for it from
    RAM, so the per-epoch read volume stays at one pass over the data.

    What that changes for training: the cut recipe is identical
    (``fire_chain_cuts``: back-to-back chains from a fresh random start each
    epoch, covering every sample from that start to the fire's end, plus the
    ROLLOUT.md extra chains, with the chains of a fire shuffled), but the
    *batch composition* is not a global shuffle — a batch can hold several
    chains of the same fire.

    Args:
        store: :class:`RealDataStore`.
        fire_ids: fires to stream (the training split).
        chain_len: steps per chain.
        seed / epoch: together select the plan, the fire order, the cut starts
            and the within-fire chain order; the same pair always gives the
            same epoch.
        num_workers: worker count the plan is built for (must match the
            DataLoader's).
        transform: normalization; defaults to the store's.
    """

    def __init__(self, store, fire_ids, chain_len, seed, epoch=0, transform=None,
                 fill_value=0.0, extra_chains=True, num_workers=1):
        self.store = store
        self.fire_ids = [str(f) for f in fire_ids]
        self.chain_len = int(chain_len)
        self.seed = seed
        self.epoch = int(epoch)
        self.transform = transform if transform is not None else store.transform()
        self.fill_value = float(fill_value)
        self.extra_chains = bool(extra_chains)
        self.num_workers = max(1, int(num_workers))
        self.plan = self._build_plan()

    # -- plan --------------------------------------------------------------
    def _build_plan(self):
        """Balanced fire -> worker assignment, identical in every worker."""
        rng = random.Random(f"{self.seed}:{self.epoch}:plan")
        fires = [f for f in self.fire_ids if self.store.n_samples(f) >= 1]
        loads = {f: self.store.n_samples(f) for f in fires}
        rng.shuffle(fires)                      # tie-break / order diversity
        order = sorted(fires, key=lambda f: -loads[f])
        bins = [[] for _ in range(self.num_workers)]
        totals = [0] * self.num_workers
        for fire_id in order:                   # greedy longest-first packing
            i = totals.index(min(totals))
            bins[i].append(fire_id)
            totals[i] += loads[fire_id]
        return bins

    def cuts_for(self, fire_id, rng) -> list:
        """This epoch's cuts for one fire (also used by tests)."""
        return fire_chain_cuts(
            self.store.n_samples(fire_id), self.store.first_fire_step(fire_id),
            self.chain_len, rng, extra_chains=self.extra_chains,
            active_steps=_active_steps(self.store, fire_id))

    # -- iteration ---------------------------------------------------------
    def __iter__(self):
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info is not None else (0, 1)
        if nw != self.num_workers:
            raise RuntimeError(
                f"StreamingChainDataset was planned for {self.num_workers} "
                f"workers but the DataLoader is running {nw}; build the dataset "
                f"with num_workers matching the loader."
            )
        rng = random.Random(f"{self.seed}:{self.epoch}:{wid}")
        for fire_id in self.plan[wid]:
            cuts = self.cuts_for(fire_id, rng)
            rng.shuffle(cuts)                    # within-fire order
            for start, length in cuts:
                chain = self.store.build_chain(fire_id, start, chain_len=length)
                yield _chain_item(self.store, chain, self.transform,
                                  self.fill_value)


class RealChainDataset(Dataset):
    """Chains as training tensors.

    ``__getitem__`` returns raw, normalized, NaN-filled tensors:

    * ``x`` (L, C, WIN, WIN) float32 — NaN (cells no window covered) filled with
      0 *after* normalization, i.e. the channel mean,
    * ``y`` (L, WIN, WIN) float32 — cumulative burned target,
    * ``scored`` (L, WIN, WIN) bool,
    * ``perim0`` (WIN, WIN) float32,
    * ``t_hours`` (L,) float32,
    * ``extent_hours`` (L,) float32 — hours from the chain start to each step's
      *label* time (what an arrival-time map is drawn against),
    * ``fire_id`` str, ``start`` int.
    """

    def __init__(self, store: RealDataStore, chains, transform=None, fill_value=0.0):
        """``chains`` is a list of ``(fire_id, start_step, length)`` where
        ``length`` is the number of steps (``None`` = to the fire's end)."""
        self.store = store
        self.chains = list(chains)
        self.transform = transform if transform is not None else store.transform()
        self.fill_value = float(fill_value)

    def __len__(self):
        return len(self.chains)

    def __getitem__(self, idx):
        fire_id, start, length = self.chains[idx]
        chain = self.store.build_chain(fire_id, start, chain_len=length)
        return _chain_item(self.store, chain, self.transform, self.fill_value)


def collate_chains(batch):
    """Pad a list of chains into batch tensors.

    Chains of one batch rarely have the same length (a fire's last chain is
    short), so every tensor is padded to the longest chain and ``valid_step``
    marks the real steps. Padded steps hold zeros and are excluded from the loss
    and from the reported metrics by the trainer.
    """
    L = max(item['x'].shape[0] for item in batch)
    B = len(batch)
    C, H, W = batch[0]['x'].shape[1:]
    x = torch.zeros((B, L, C, H, W), dtype=torch.float32)
    y = torch.zeros((B, L, H, W), dtype=torch.float32)
    scored = torch.zeros((B, L, H, W), dtype=torch.bool)
    t_hours = torch.zeros((B, L), dtype=torch.float32)
    extent_hours = torch.zeros((B, L), dtype=torch.float32)
    valid_step = torch.zeros((B, L), dtype=torch.bool)
    for i, item in enumerate(batch):
        n = item['x'].shape[0]
        x[i, :n] = item['x']
        y[i, :n] = item['y']
        scored[i, :n] = item['scored']
        t_hours[i, :n] = item['t_hours']
        extent_hours[i, :n] = item['extent_hours']
        valid_step[i, :n] = True
    return {
        'x': x, 'y': y, 'scored': scored, 't_hours': t_hours,
        'extent_hours': extent_hours, 'valid_step': valid_step,
        'perim0': torch.stack([item['perim0'] for item in batch]),
        'fire_id': [item['fire_id'] for item in batch],
        'start': torch.tensor([item['start'] for item in batch]),
    }


# ---------------------------------------------------------------------------
# chain sampling
# ---------------------------------------------------------------------------
def _active_steps(store, fire_id):
    """Sample indices where the fire actually grew (category != no_growth)."""
    rows = store.steps(fire_id)
    return rows.loc[rows["growth_px"].to_numpy() > 0, "step"].to_numpy()


def build_val_chains(store, fire_ids, chain_len) -> list:
    """Deterministic per-fire validation chains (first real sample, capped).

    Returns ``(fire_id, start_step, length)`` triples, one per fire that has at
    least two steps available.
    """
    chains = []
    for fire_id in fire_ids:
        fire_id = str(fire_id)
        n = store.n_samples(fire_id)
        start0 = store.first_fire_step(fire_id)
        if n - start0 < 2:
            continue
        chains.append((fire_id, int(start0), min(chain_len, n - start0)))
    return chains


def full_life_chain_ref(fire_id, store) -> tuple:
    """``(fire_id, start_step, None)`` chain list entry for a full-life rollout."""
    start0 = store.first_fire_step(fire_id)
    return (str(fire_id), int(start0), None)


def build_full_life_chain(store, fire_id) -> RealChain:
    """One chain from a fire's first real sample to its last sample."""
    return store.build_chain(fire_id, *full_life_chain_ref(fire_id, store)[1:])
