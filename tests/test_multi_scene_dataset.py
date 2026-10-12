import numpy as np
import torch

from wildfire_simulator.datasets import WildfireDataset, MultiSceneDataset


def test_multi_scene_length_and_routing(dataloader):
    # Two identical single-scene datasets stand in for two scenes. Using the
    # same underlying loader is sufficient to exercise routing and aggregation.
    scene_a = WildfireDataset(dataloader)
    scene_b = WildfireDataset(dataloader)
    multi = MultiSceneDataset([scene_a, scene_b])

    # Flat length is the sum of per-scene lengths.
    assert len(multi) == len(scene_a) + len(scene_b)
    assert multi.num_scenes() == 2

    # Global index 0 routes to scene 0 local 0; the first index past scene A
    # routes to scene 1 local 0.
    first_of_b = len(scene_a)
    assert torch.equal(multi[0], scene_a[0])
    assert torch.equal(multi[first_of_b], scene_b[0])


def test_multi_scene_shared_normalization(dataloader):
    scene_a = WildfireDataset(dataloader)
    scene_b = WildfireDataset(dataloader)
    multi = MultiSceneDataset([scene_a, scene_b])

    # Shared stats are the element-wise reduction over per-scene stats.
    expected_min = np.minimum(scene_a.min_val, scene_b.min_val)
    expected_max = np.maximum(scene_a.max_val, scene_b.max_val)
    assert np.array_equal(multi.min_val, expected_min)
    assert np.array_equal(multi.max_val, expected_max)


def test_multi_scene_scene_indices_are_contiguous(dataloader):
    scene_a = WildfireDataset(dataloader)
    scene_b = WildfireDataset(dataloader)
    multi = MultiSceneDataset([scene_a, scene_b])

    idx_a = multi.scene_indices(0)
    idx_b = multi.scene_indices(1)

    # Every global index belongs to exactly one scene, and the two sets
    # partition the full index range.
    assert idx_a == list(range(len(scene_a)))
    assert idx_b == list(range(len(scene_a), len(scene_a) + len(scene_b)))
    assert sorted(idx_a + idx_b) == list(range(len(multi)))


def test_stratified_split_covers_every_scene(dataloader):
    scene_a = WildfireDataset(dataloader)
    scene_b = WildfireDataset(dataloader)
    multi = MultiSceneDataset([scene_a, scene_b], scene_names=["old", "new"])

    train_idx, val_idx, per_scene_val = multi.stratified_split(val_frac=0.5, seed=7)

    # Every scene is represented in the per-scene val mapping.
    assert set(per_scene_val.keys()) == {"old", "new"}
    # With 2 samples/scene and val_frac=0.5, each scene contributes 1 val index.
    assert len(per_scene_val["old"]) == 1
    assert len(per_scene_val["new"]) == 1

    # Combined val indices equal the union of per-scene val indices.
    assert sorted(val_idx) == sorted(
        i for idxs in per_scene_val.values() for i in idxs
    )
    # Train and val partition the full dataset with no overlap.
    assert set(train_idx).isdisjoint(set(val_idx))
    assert sorted(train_idx + val_idx) == list(range(len(multi)))

    # Each scene's val indices actually belong to that scene.
    assert set(per_scene_val["old"]).issubset(set(multi.scene_indices(0)))
    assert set(per_scene_val["new"]).issubset(set(multi.scene_indices(1)))


def test_stratified_split_is_deterministic(dataloader):
    scene_a = WildfireDataset(dataloader)
    scene_b = WildfireDataset(dataloader)
    multi = MultiSceneDataset([scene_a, scene_b], scene_names=["old", "new"])

    a = multi.stratified_split(val_frac=0.5, seed=13)
    b = multi.stratified_split(val_frac=0.5, seed=13)
    assert a == b


def test_scene_names_default_and_custom(dataloader):
    scene = WildfireDataset(dataloader)
    default = MultiSceneDataset([scene])
    assert default.scene_name(0) == "scene_0"

    named = MultiSceneDataset([scene], scene_names=["palisades"])
    assert named.scene_name(0) == "palisades"
