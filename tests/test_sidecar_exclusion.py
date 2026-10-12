from wildfire_simulator.dataloader import TrialCollection


class _RecordingLoader:
    """Loader stub that returns the path it was asked to load."""
    def load(self, file_path):
        return file_path


def test_sidecars_excluded_from_trial_enumeration(tmp_path):
    # A terrain-aware trial is 3 files: the base arrival raster plus two
    # sidecar wind grids. Only the base should be enumerated as a trial.
    (tmp_path / "trial_I0_WS10_WD90_M80.tif").write_bytes(b"")
    (tmp_path / "trial_I0_WS10_WD90_M80_windspeed.tif").write_bytes(b"")
    (tmp_path / "trial_I0_WS10_WD90_M80_winddir.tif").write_bytes(b"")
    # A second, uniform-wind trial with no sidecars.
    (tmp_path / "trial_I1_WS20_WD45_M75.tif").write_bytes(b"")
    # A non-tif file that must be ignored entirely.
    (tmp_path / "notes.txt").write_bytes(b"")

    tc = TrialCollection(_RecordingLoader(), trials_dir=str(tmp_path))

    # Exactly the two base trials, no sidecars.
    assert len(tc) == 2
    names = sorted(p.rsplit("/", 1)[-1] for p in tc.files)
    assert names == [
        "trial_I0_WS10_WD90_M80.tif",
        "trial_I1_WS20_WD45_M75.tif",
    ]
    # No enumerated path is a wind sidecar.
    assert not any(
        f.endswith("_windspeed.tif") or f.endswith("_winddir.tif") for f in tc.files
    )
