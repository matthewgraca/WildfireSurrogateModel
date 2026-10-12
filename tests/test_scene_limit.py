from wildfire_simulator.dataloader import TrialCollection


class _RecordingLoader:
    def load(self, file_path):
        return file_path


def _make_trials(tmp_path, n):
    for i in range(n):
        (tmp_path / f"trial_I{i}_WS10_WD90_M80.tif").write_bytes(b"")


def test_limit_caps_count(tmp_path):
    _make_trials(tmp_path, 10)
    tc = TrialCollection(_RecordingLoader(), trials_dir=str(tmp_path), limit=4)
    assert len(tc) == 4


def test_limit_none_keeps_all(tmp_path):
    _make_trials(tmp_path, 10)
    tc = TrialCollection(_RecordingLoader(), trials_dir=str(tmp_path), limit=None)
    assert len(tc) == 10


def test_limit_larger_than_available_is_noop(tmp_path):
    _make_trials(tmp_path, 5)
    tc = TrialCollection(_RecordingLoader(), trials_dir=str(tmp_path), limit=100)
    assert len(tc) == 5


def test_same_seed_same_subset(tmp_path):
    _make_trials(tmp_path, 20)
    a = TrialCollection(_RecordingLoader(), trials_dir=str(tmp_path), limit=8, seed=7)
    b = TrialCollection(_RecordingLoader(), trials_dir=str(tmp_path), limit=8, seed=7)
    assert a.files == b.files  # reproducible
    assert len(a) == 8


def test_different_seed_different_subset(tmp_path):
    _make_trials(tmp_path, 20)
    a = TrialCollection(_RecordingLoader(), trials_dir=str(tmp_path), limit=8, seed=1)
    b = TrialCollection(_RecordingLoader(), trials_dir=str(tmp_path), limit=8, seed=2)
    # Same size, but (overwhelmingly likely) a different subset.
    assert len(a) == len(b) == 8
    assert a.files != b.files


def test_subset_files_are_sorted(tmp_path):
    _make_trials(tmp_path, 20)
    tc = TrialCollection(_RecordingLoader(), trials_dir=str(tmp_path), limit=8, seed=3)
    assert tc.files == sorted(tc.files)
