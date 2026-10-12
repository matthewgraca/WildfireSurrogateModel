import numpy as np
import torch

from wildfire_simulator.callbacks import TensorBoardCallback


class _FakeWriter:
    def __init__(self):
        self.images = {}
        self.scalars = {}

    def add_image(self, tag, img, step, dataformats=None):
        # The renderers emit uint8 (H, W, 3) arrays; HWC must be declared
        # or torch's CHW default permutes the image into garbage.
        assert dataformats == "HWC"
        self.images.setdefault(tag, []).append((step, img))

    def add_scalar(self, tag, value, step):
        self.scalars.setdefault(tag, []).append((step, value))


def _payload(idx):
    pred_hist = [torch.zeros(2, 4, 4) for _ in range(2)]
    pred_hist[0][0, 0, 0] = 1.0
    gt_hist = [torch.zeros(2, 4, 4) for _ in range(2)]
    gt_hist[1][0, 0, 0] = 1.0
    return {
        "idx": idx,
        "pred_history": pred_hist,
        "gt_history": gt_hist,
    }


def test_tensorboard_callback_logs_viz_images_and_iou():
    train_w, val_w = _FakeWriter(), _FakeWriter()
    cb = TensorBoardCallback(train_w, val_w)

    metrics = {
        "train_loss": 0.5,
        "val_loss": 0.4,
        "val_iou": 0.7,
        "val_loss/sceneA": 0.3,
        "val_iou/sceneA": 0.8,
        "viz": {"sceneA": [_payload(3), _payload(17)]},
    }
    cb.on_validation_end(7, metrics, None, None)

    assert val_w.scalars["IOU"] == [(7, 0.7)]
    assert val_w.scalars["IOU/scene/sceneA"] == [(7, 0.8)]
    assert val_w.scalars["Loss/scene/sceneA"] == [(7, 0.3)]

    expected_tags = {"viz/sceneA/fat_montage"}
    assert set(val_w.images) == expected_tags
    for entries in val_w.images.values():
        assert [step for step, _ in entries] == [7]
        for _, img in entries:
            assert img.dtype == np.uint8
            assert img.ndim == 3 and img.shape[2] == 3


def test_tensorboard_callback_no_viz_no_images():
    train_w, val_w = _FakeWriter(), _FakeWriter()
    cb = TensorBoardCallback(train_w, val_w)
    cb.on_validation_end(
        0, {"train_loss": 0.5, "val_loss": 0.4, "val_iou": 0.1}, None, None
    )
    assert val_w.images == {}
    assert val_w.scalars["IOU"] == [(0, 0.1)]
