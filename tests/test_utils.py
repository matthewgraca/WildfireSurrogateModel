import numpy as np
import torch
from pathlib import Path
import shutil
from PIL import Image

from wildfire_simulator.utils import save_frame, save_comparison, ScalarRNG

# matplotlib's rasterization is not pixel-stable across versions (anti-aliasing,
# tight-bbox rounding, font metrics), which would make an exact-pixel comparison
# against a checked-in baseline fail on any matplotlib bump. Validate the render
# structurally (same mode/layout) plus a tolerance on an average-pooled view, so a
# broken/blank/mis-laid-out image still fails while a version-rasterized one passes.
_BASELINE = Path(__file__).parent / "baseline"
_BLUR = 8          # average-pool factor: smooths sub-pixel anti-aliasing changes
_SHAPE_TOL = 8     # tight bbox can shift a few pixels between matplotlib versions
_MEAN_TOL = 15.0   # max allowed mean |diff| on the pooled comparison


def _assert_render_matches(got_path, baseline_name):
    got = Image.open(got_path)
    base = Image.open(_BASELINE / baseline_name)
    assert got.mode == base.mode

    a = np.asarray(got).astype(np.float64)
    b = np.asarray(base).astype(np.float64)

    # Same grid layout -> close overall dimensions (allow tight-bbox rounding).
    assert abs(a.shape[0] - b.shape[0]) <= _SHAPE_TOL, (a.shape, b.shape)
    assert abs(a.shape[1] - b.shape[1]) <= _SHAPE_TOL, (a.shape, b.shape)

    def pool(x):
        H = x.shape[0] // _BLUR * _BLUR
        W = x.shape[1] // _BLUR * _BLUR
        return x[:H, :W].reshape(H // _BLUR, _BLUR, W // _BLUR, _BLUR, -1).mean(axis=(1, 3))

    h, w = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
    mean_diff = np.abs(pool(a[:h, :w]) - pool(b[:h, :w])).mean()
    assert mean_diff < _MEAN_TOL, f"render deviates from baseline: mean abs diff {mean_diff:.1f} >= {_MEAN_TOL}"


def test_save_frame():
    rng = np.random.default_rng(seed=42)
    data = rng.random((13, 500, 500))

    shutil.rmtree('./tmp', ignore_errors=True)

    save_frame(data, './tmp/random_frame.png')

    file_path = Path('./tmp/random_frame.png')
    assert file_path.is_file()

    _assert_render_matches(file_path, "random_frame.png")


def test_save_comparison():
    rng = np.random.default_rng(seed=42)

    # 10 different fires where each image is an arrival map
    trues = rng.random((10, 500, 500))
    preds = rng.random((10, 500, 500))

    shutil.rmtree('./tmp', ignore_errors=True)

    save_comparison(preds, trues, './tmp/random_comparison.png')

    file_path = Path('./tmp/random_comparison.png')
    assert file_path.is_file()

    _assert_render_matches(file_path, "random_comparison.png")


def test_scalar_rng():
    rng1 = ScalarRNG()
    rng1.seed(42)
    value1 = rng1.rand()
    
    assert isinstance(value1, torch.Tensor)
    assert value1.dim() == 0
    assert value1.dtype == torch.float32
    assert 0.0 <= value1.item() < 1.0

    rng2 = ScalarRNG()
    rng2.seed(42)
    value2 = rng2.rand()
    assert torch.equal(value1, value2)

    rng3 = ScalarRNG()
    rng3.seed(84)
    value3 = rng3.rand()
    assert not torch.equal(value1, value3)
