import numpy as np
import torch
from torch.utils.data import Dataset

class WildfireDataset(Dataset):
    def __init__(self, dataloader):
        self.loader = dataloader

        # Compute channel-wise min and max across all trials/frames for normalization
        n_channels = 13
        min_val = np.full(n_channels,  np.inf)
        max_val = np.full(n_channels, -np.inf)
        for i in range(self.__len__()):
            arr = self._load_raw_channels(i)
            frame_min = np.min(arr, axis=(1, 2))
            frame_max = np.max(arr, axis=(1, 2))
            min_val = np.minimum(min_val, frame_min)
            max_val = np.maximum(max_val, frame_max)
        self.min_val = min_val
        self.max_val = max_val

    def _load_raw_channels(self, idx):
        trial = self.loader.trials[idx]
        ig_idx = trial["ignition"]
        cy, cx = self.loader.ignitions[ig_idx]
        half = 250

        # Guard: the fixed 500x500 crop must fit within the landscape raster.
        # If an ignition sits within `half` pixels of an edge, numpy slicing
        # would silently return an undersized array and misalign the channel
        # stack. Fail loudly instead.
        H, W = self.loader.elevation.shape
        if cy - half < 0 or cx - half < 0 or cy + half > H or cx + half > W:
            raise ValueError(
                f"Ignition {ig_idx} at (row={cy}, col={cx}) is too close to the "
                f"landscape edge for a {2*half}x{2*half} crop "
                f"(landscape is {H}x{W}). Trial: {trial.get('filename', idx)}."
            )

        # 8 landscape layers in order
        land_layers = [
            self.loader.elevation,
            self.loader.slope,
            self.loader.aspect,
            self.loader.fuel,
            self.loader.canopy_cover,
            self.loader.stand_height,
            self.loader.canopy_base_height,
            self.loader.canopy_bulk_density
        ]
        crops = [
            np.asarray(arr[cy-half:cy+half, cx-half:cx+half], dtype=np.float32)
            for arr in land_layers
        ]

        # 2 fire channels (mask and arrival time)
        fire_mask = np.asarray(
            trial["fire"][0][cy-half:cy+half, cx-half:cx+half],
            dtype=np.float32,
        )
        fire_arr = np.asarray(
            trial["fire"][1][cy-half:cy+half, cx-half:cx+half],
            dtype=np.float32,
        )

        # Wind U/V channels: use per-cell grids if available, else fall back to uniform
        # Convert wind speed + direction to U/V cartesian components
        # Meteorological convention: direction is where wind comes FROM, clockwise from north
        # U = east-west component (positive = wind blowing east)
        # V = north-south component (positive = wind blowing north)
        if trial["wind_speed_grid"] is not None and trial["wind_dir_grid"] is not None:
            # Per-cell terrain-aware wind from WindNinja grids
            speed_grid = trial["wind_speed_grid"][cy-half:cy+half, cx-half:cx+half]
            dir_grid = trial["wind_dir_grid"][cy-half:cy+half, cx-half:cx+half]
            dir_rad = np.radians(dir_grid)
            wu = (-speed_grid * np.sin(dir_rad)).astype(np.float32)
            wv = (-speed_grid * np.cos(dir_rad)).astype(np.float32)
        else:
            # Fallback: uniform wind from scalar metadata (v1 data)
            wind_speed = trial["windspeed"]
            wind_dir_rad = np.radians(trial["winddir"])
            wind_u = -wind_speed * np.sin(wind_dir_rad)
            wind_v = -wind_speed * np.cos(wind_dir_rad)
            wu = np.full((500, 500), wind_u, dtype=np.float32)
            wv = np.full((500, 500), wind_v, dtype=np.float32)

        fm = np.full((500, 500), trial["foliar_moisture"], dtype=np.float32)
        # stack all 13 channels
        stacked = np.stack(
            [fire_mask, fire_arr, *crops, wu, wv, fm], axis=0
        )

        return stacked

    def __len__(self):
        return len(self.loader.trials)

    def __getitem__(self, idx):
        channels = self._load_raw_channels(idx)
        return torch.from_numpy(channels).to(torch.float32)


class MultiSceneDataset(Dataset):
    """Concatenates several single-scene ``WildfireDataset`` instances into one
    flat, indexable dataset with a single shared normalization.

    Each scene is a ``WildfireDataset`` bound to its own landscape / trials /
    ignitions. A "scene" therefore owns its own terrain crops and its own
    ignition-index namespace, so ignition indices never collide across scenes.

    This class is pure routing plus statistic aggregation: all cropping, wind
    conversion, and channel assembly stays in ``WildfireDataset``. Downstream
    code (``random_split``, ``TransformedDataset``, ``DataLoader``) sees the
    same interface as a single ``WildfireDataset``: ``__len__``, ``__getitem__``
    returning a ``(13, 500, 500)`` tensor, and ``min_val`` / ``max_val``.
    """

    def __init__(self, scene_datasets, scene_names=None):
        if not scene_datasets:
            raise ValueError("MultiSceneDataset requires at least one scene.")
        self.scenes = list(scene_datasets)

        # Stable, human-readable name per scene for metric labels. Defaults to
        # scene_0, scene_1, ... when not provided.
        if scene_names is None:
            scene_names = [f"scene_{i}" for i in range(len(self.scenes))]
        if len(scene_names) != len(self.scenes):
            raise ValueError(
                f"scene_names length ({len(scene_names)}) must match number of "
                f"scenes ({len(self.scenes)})."
            )
        self.scene_names = list(scene_names)

        # Flat global index -> (scene_id, local_idx). Iteration order is stable
        # (scene 0 first, then scene 1, ...), which the leave-scene-out helper
        # relies on to produce contiguous per-scene index ranges.
        self.index_map = [
            (scene_id, local_idx)
            for scene_id, ds in enumerate(self.scenes)
            for local_idx in range(len(ds))
        ]

        # Shared normalization: combine per-scene stats that each WildfireDataset
        # already computed in its own __init__. No re-scan of the data needed.
        self.min_val = np.minimum.reduce([ds.min_val for ds in self.scenes])
        self.max_val = np.maximum.reduce([ds.max_val for ds in self.scenes])

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        scene_id, local_idx = self.index_map[idx]
        return self.scenes[scene_id][local_idx]

    def scene_indices(self, scene_id):
        """Return the list of global indices belonging to ``scene_id``.

        Useful for a leave-scene-out validation split: hold out one scene's
        indices for validation and use the rest for training.
        """
        return [
            global_idx
            for global_idx, (s_id, _) in enumerate(self.index_map)
            if s_id == scene_id
        ]

    def num_scenes(self):
        return len(self.scenes)

    def scene_name(self, scene_id):
        return self.scene_names[scene_id]

    def stratified_split(self, val_frac=0.2, seed=42):
        """Split *within each scene* so every scene is represented in both
        train and validation in known proportion.

        Returns:
            train_indices: list[int] of global indices for training.
            val_indices:   list[int] of global indices for validation
                           (concatenation of all per-scene val indices).
            per_scene_val: dict[str, list[int]] mapping scene name -> the global
                           validation indices belonging to that scene, so
                           validation performance can be measured per scene.

        Deterministic given ``seed``. Uses a torch generator so shuffling
        matches the rest of the pipeline's RNG conventions.
        """
        import torch

        train_indices = []
        val_indices = []
        per_scene_val = {}

        for scene_id in range(len(self.scenes)):
            g_idx = self.scene_indices(scene_id)
            n = len(g_idx)
            # Deterministic per-scene permutation (offset seed by scene so
            # scenes don't share the same ordering).
            gen = torch.Generator().manual_seed(seed + scene_id)
            perm = torch.randperm(n, generator=gen).tolist()
            n_val = int(round(val_frac * n))
            val_local = perm[:n_val]
            train_local = perm[n_val:]

            scene_val = [g_idx[i] for i in val_local]
            train_indices.extend(g_idx[i] for i in train_local)
            val_indices.extend(scene_val)
            per_scene_val[self.scene_names[scene_id]] = scene_val

        return train_indices, val_indices, per_scene_val


class TransformedDataset:
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.transform(self.dataset[idx])


def build_multiscene_dataset(config):
    """Build a ``MultiSceneDataset`` from ``config['scenes']``.

    Shared by all scripts (run, evaluate, ablation, preflight, examine_data) so
    scene assembly, scene naming, and shared normalization live in one place.

    Scene names are derived from each landscape's parent directory, which
    disambiguates scenes whose inner .tif shares a filename (e.g. two different
    ``palisades.tif`` under ``classic-palisades/`` and ``palisades/``).

    Args:
        config: parsed config dict; must contain a ``scenes`` list of
            ``{landscape, trials, ignitions}`` entries.

    Returns:
        A ``MultiSceneDataset`` (exposes ``min_val``/``max_val`` for building a
        ``MinMaxPerChannel`` transform, and ``stratified_split`` for splitting).
    """
    # Imported here to avoid a circular import at module load time.
    from wildfire_simulator.dataloader import (
        TrialCollection, TrialFileLoader, WildfireDataLoader,
    )
    from pathlib import Path

    scenes = config.get('scenes') or []
    if not scenes:
        raise ValueError(
            "config['scenes'] is empty; at least one scene "
            "{landscape, trials, ignitions} is required."
        )

    scene_datasets = []
    scene_names = []
    for scene in scenes:
        loader = WildfireDataLoader(
            TrialCollection(
                TrialFileLoader(),
                trials_dir=scene['trials'],
                limit=scene.get('limit'),
                seed=scene.get('seed', 42),
            ),
            landscape_path=scene['landscape'],
            ignitions_dir=scene['ignitions'],
        )
        scene_datasets.append(WildfireDataset(loader))
        scene_names.append(Path(scene['landscape']).parent.name)

    return MultiSceneDataset(scene_datasets, scene_names=scene_names)

