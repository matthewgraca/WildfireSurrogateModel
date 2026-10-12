import numpy as np
import pickle

from wildfire_simulator.dataloader import TrialCollection

def test_landscape_layers(dataloader):
    elevation = dataloader.elevation 
    elevation_expected = np.load('tests/baseline/elevation.npy')
    assert (elevation == elevation_expected).all()

    slope = dataloader.slope
    slope_expected = np.load('tests/baseline/slope.npy')
    assert (slope == slope_expected).all()

    aspect = dataloader.aspect
    aspect_expected = np.load('tests/baseline/aspect.npy')
    assert (aspect == aspect_expected).all()

    fuel = dataloader.fuel
    fuel_expected = np.load('tests/baseline/fuel.npy')
    assert (fuel == fuel_expected).all()

    canopy_cover = dataloader.canopy_cover
    canopy_cover_expected = np.load('tests/baseline/canopy_cover.npy')
    assert (canopy_cover == canopy_cover_expected).all()

    stand_height = dataloader.stand_height
    stand_height_expected = np.load('tests/baseline/stand_height.npy')
    assert (stand_height == stand_height_expected).all()

    canopy_base_height = dataloader.canopy_base_height
    canopy_base_height_expected = np.load('tests/baseline/canopy_base_height.npy')
    assert (canopy_base_height == canopy_base_height_expected).all()

    canopy_bulk_density = dataloader.canopy_bulk_density
    canopy_bulk_density_expected = np.load('tests/baseline/canopy_bulk_density.npy')
    assert (canopy_bulk_density == canopy_bulk_density_expected).all()

def test_ignitions(dataloader):
    ignitions = dataloader.ignitions
    with open("tests/baseline/ignitions.pkl", "rb") as file:
        ignitions_expected = pickle.load(file)
    assert ignitions == ignitions_expected

def test_trials(dataloader):
    trials = list(dataloader.trials)
    with open("tests/baseline/trials.pkl", "rb") as file:
        trials_expected = pickle.load(file)

    for idx, trial in enumerate(trials):
        # Iterate over the keys captured in the baseline. Newer keys added to
        # the trial dict later (e.g. per-cell wind grids) are not present in the
        # baseline and are intentionally not asserted here.
        for k, v in trials_expected[idx].items():
            if k == "fire":
                assert (trial[k] == v).all()
            else:
                assert trial[k] == v

def test_trial_collection_laziness():
    from pathlib import Path

    class FakeTrialFileLoader:
        def load(self, file_path):
            return self.data

    loader = FakeTrialFileLoader()
    # Point at the test Trials dir so at least one file path is discovered;
    # the fake loader ignores the path and returns self.data, which is what
    # this test exercises (lazy load on each __getitem__).
    trials_dir = str(Path(__file__).parent / "data" / "Trials")
    trials = TrialCollection(loader, trials_dir=trials_dir)

    loader.data = "example data"
    assert trials[0] == "example data"

    loader.data = "some other example data"
    assert trials[0] == "some other example data"

