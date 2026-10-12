import pytest
from wildfire_simulator.dataloader import WildfireDataLoader, TrialFileLoader, TrialCollection
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"


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
