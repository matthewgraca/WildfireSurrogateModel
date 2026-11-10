"""Tests for real-data TensorBoard logging (finetune.tb + tracked recording)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from finetune.dataset import RealChainDataset, collate_chains
from finetune.tb import RealTensorBoardCallback, chain_iou
from finetune.trainer import RealChainTrainer

FIRE_A = "V5_2020_2020_00001"
FIRE_B = "V5_2020_2020_00002"


class _FakeWriter:
    def __init__(self):
        self.images = {}
        self.scalars = {}

    def add_image(self, tag, img, step, dataformats=None):
        assert dataformats == "HWC"
        self.images.setdefault(tag, []).append((step, img))

    def add_scalar(self, tag, value, step):
        self.scalars.setdefault(tag, []).append((step, value))


def _loader(store, chains, batch_size=4):
    ds = RealChainDataset(store, chains, transform=store.transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_chains)


def _payload(idx, fire_id, n_steps=3):
    """A tracked payload shaped like the trainer's recording."""
    seed = torch.zeros(135, 135)
    seed[10:40, 10:40] = 1.0
    pred, gt = [], []
    for k in range(n_steps):
        mask = seed.clone()
        mask[10 + k:40 + k, 10:40] = 1.0
        hours = (k + 1) * 12.0 / 144.0
        pred.append(torch.stack([mask, torch.full(mask.shape, hours)]))
        gt.append(torch.stack([mask, torch.full(mask.shape, hours)]))
    return {
        'idx': idx, 'fire_id': fire_id, 'start': 0,
        'perim0': seed > 0.5, 'hours': [(k + 1) * 12.0 for k in range(n_steps)],
        'scored': [torch.ones(135, 135, dtype=torch.bool) for _ in range(n_steps)],
        'pred_history': pred, 'gt_history': gt,
    }


def test_callback_logs_iou_scalars_and_tracked_images():
    train_w, val_w = _FakeWriter(), _FakeWriter()
    cb = RealTensorBoardCallback(train_w, val_w)
    cb.on_validation_end(4, {
        "train_loss": 0.5, "val_loss": 0.4, "train_iou": 0.21, "val_iou": 0.3,
        "viz": {"val": [_payload(0, FIRE_A), _payload(1, FIRE_B)]},
    }, None, None)

    assert train_w.scalars["IOU"] == [(4, 0.21)]
    assert val_w.scalars["IOU"] == [(4, 0.3)]                  # from the base
    assert val_w.scalars[f"IOU/fire/{FIRE_A}"] == [(4, 1.0)]    # identical masks
    assert val_w.scalars[f"IOU/fire/{FIRE_B}"] == [(4, 1.0)]
    assert train_w.scalars["Loss"] == [(4, 0.5)]
    assert val_w.scalars["Loss"] == [(4, 0.4)]

    assert set(val_w.images) == {
        "viz/val/fat_montage",
        f"viz/fire/{FIRE_A}/direct",
        f"viz/fire/{FIRE_B}/direct",
    }
    for entries in val_w.images.values():
        assert [step for step, _ in entries] == [4]
        for _, img in entries:
            assert img.dtype == np.uint8
            assert img.ndim == 3 and img.shape[2] == 3


def test_callback_without_viz_only_logs_scalars():
    train_w, val_w = _FakeWriter(), _FakeWriter()
    cb = RealTensorBoardCallback(train_w, val_w)
    cb.on_validation_end(1, {"train_loss": 0.5, "val_loss": 0.4,
                             "train_iou": float('nan'), "val_iou": 0.3},
                         None, None)
    assert val_w.images == {}
    assert "IOU" not in train_w.scalars          # NaN is not plotted
    assert val_w.scalars["IOU"] == [(1, 0.3)]


def test_chain_iou_uses_the_scored_cells():
    payload = _payload(0, FIRE_A, n_steps=1)
    assert chain_iou(payload) == pytest.approx(1.0)
    # a prediction outside the scored region must not count
    payload['pred_history'][-1][0][0, 0] = 1.0
    payload['gt_history'][-1][0][0, 0] = 0.0
    payload['scored'][-1][0, 0] = False
    assert chain_iou(payload) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# trainer side: recording
# ---------------------------------------------------------------------------
def _trainer(store, model, sampler, **kw):
    epochs = kw.pop("epochs", 1)
    return RealChainTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=torch.nn.BCELoss(),
        train_loader_factory=lambda epoch: _loader(store, [(FIRE_A, 1, 2)]),
        val_loader=_loader(store, [(FIRE_A, 1, 2), (FIRE_B, 0, 3)]),
        sampler=sampler, epochs=epochs, device=torch.device("cpu"),
        progress=False, **kw)


class _Stub(torch.nn.Module):
    def __init__(self, value=0.9):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.value = value

    def forward(self, x):
        return [torch.full((x.shape[0], 1, x.shape[2], x.shape[3]),
                           self.value) + self.dummy * 0.0]


class _GrowingStub(torch.nn.Module):
    """Burns one 10x10 box per call (one call per rollout step)."""

    def __init__(self):
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, x):
        b, _, h, w = x.shape
        out = torch.zeros((b, 1, h, w), dtype=x.dtype)
        r = 10 * (self.calls % 4)
        out[:, :, r:r + 10, r:r + 10] = 1.0
        self.calls += 1
        return [out + self.dummy * 0.0]


class _ConstSampler:
    def __init__(self, prob):
        self.prob = prob

    def get_prob(self, epoch):
        return self.prob


def test_run_records_only_the_requested_positions(real_store):
    trainer = _trainer(real_store, _GrowingStub(), _ConstSampler(0.0))
    loader = trainer.val_loader
    _, _, records = trainer._run(loader, 0, "t", train=False,
                                 record_positions=[1])
    assert list(records) == [1]
    payload = records[1]
    assert payload['fire_id'] == FIRE_B
    assert payload['start'] == 0
    assert len(payload['pred_history']) == 3          # the chain's own length
    assert list(payload['hours']) == [12.0, 24.0, 36.0]
    assert len(payload['scored']) == 3
    assert payload['pred_history'][0].shape == (2, 135, 135)

    # Frames are [mask, arrival hours / fat_norm_hours] (the MTT convention),
    # so a cell keeps the hour it first burned: the seed at 0, then one box per
    # step at 12 h / 24 h / 36 h out of the 144 h full chain.
    frames = payload['pred_history']
    assert frames[0][0].max() == 1.0
    arrived = sorted({round(float(v) * 144, 1)
                      for f in frames for v in f[1].unique()})
    assert arrived == [0.0, 12.0, 24.0, 36.0]
    for frame in frames + payload['gt_history']:
        assert float(frame[1].min()) >= 0.0 and float(frame[1].max()) <= 1.0

    # no positions requested -> no records at all
    _, _, none = trainer._run(loader, 0, "t", train=False)
    assert none is None


def test_fit_publishes_viz_only_on_viz_epochs(real_store):
    seen = []

    class _Capture:
        def on_validation_end(self, epoch, metrics, model, optimizer):
            seen.append(('viz' in metrics, metrics.get('train_iou')))
            return False

    trainer = _trainer(real_store, _Stub(), _ConstSampler(0.0), epochs=3,
                       viz_every=2, viz_positions=[0], callbacks=[_Capture()])
    trainer.fit()
    assert [has_viz for has_viz, _ in seen] == [True, False, True]
    assert all(np.isfinite(iou) for _, iou in seen)


def test_viz_every_zero_records_nothing(real_store):
    trainer = _trainer(real_store, _Stub(), _ConstSampler(0.0),
                       viz_every=0, viz_positions=[0], epochs=1)

    class _Capture:
        def on_validation_end(self, epoch, metrics, model, optimizer):
            assert 'viz' not in metrics
            return False

    trainer.callbacks = [_Capture()]
    trainer.fit()


# ---------------------------------------------------------------------------
# end to end: a real SummaryWriter and event file
# ---------------------------------------------------------------------------
def test_event_file_contains_iou_and_tracked_images(real_store, tmp_path):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    from torch.utils.tensorboard import SummaryWriter

    val_dir = tmp_path / "val"
    writer = SummaryWriter(str(val_dir))
    cb = RealTensorBoardCallback(
        train_writer=SummaryWriter(str(tmp_path / "train")),
        val_writer=writer,
    )
    trainer = _trainer(real_store, _Stub(), _ConstSampler(0.0), epochs=2,
                       viz_every=1, viz_positions=[1], callbacks=[cb])
    trainer.fit()
    writer.close()

    acc = EventAccumulator(str(val_dir))
    acc.Reload()
    tags = acc.Tags()
    assert "IOU" in tags["scalars"]
    assert "Loss" in tags["scalars"]
    assert f"IOU/fire/{FIRE_B}" in tags["scalars"]
    assert set(tags["images"]) == {f"viz/fire/{FIRE_B}/direct",
                                   "viz/val/fat_montage"}

    steps = [e.step for e in acc.Scalars("IOU")]
    assert steps == [0, 1]                       # IoU per epoch

    events = acc.Images("viz/val/fat_montage")
    assert [e.step for e in events] == [0, 1]    # every viz epoch
    assert events[0].encoded_image_string.startswith(b"\x89PNG")
    assert events[0].width > 50 and events[0].height > 50
    direct = acc.Images(f"viz/fire/{FIRE_B}/direct")
    assert direct[0].encoded_image_string.startswith(b"\x89PNG")
    assert len(direct[0].encoded_image_string) > 2000     # palette + legend
