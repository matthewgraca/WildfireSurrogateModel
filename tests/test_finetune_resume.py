"""Resuming a real-data run: checkpoint state, epoch continuity, TensorBoard.

Reuses the tiny-model helpers from ``test_finetune_trainer`` (pytest puts the
tests directory on ``sys.path``).
"""

import numpy as np
import pytest
import torch

from test_finetune_trainer import (_loader, ConstSampler, FIRE_A, FIRE_B,
                                   RecordingModel, _trainer)

from finetune.trainer import RealChainTrainer
from wildfire_simulator.callbacks import EarlyStopping, ModelCheckpoint
from wildfire_simulator.losses import DiceLoss

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from train_real import check_resume_epochs, resolve_resume    # noqa: E402


# ---------------------------------------------------------------------------
# callback state
# ---------------------------------------------------------------------------
def test_checkpoint_carries_and_restores_callback_state(tmp_path):
    """The patience/best of the callbacks travels in the checkpoint file."""
    early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    ckpt = ModelCheckpoint(monitor="val_loss", mode="min",
                           filepath=str(tmp_path / "best-{epoch:02d}-{val_loss:.4f}.pt"),
                           state_providers=[early])
    model, opt = RecordingModel(), torch.optim.SGD(RecordingModel().parameters(),
                                                   lr=0.01)
    # The trainer dispatches EarlyStopping before ModelCheckpoint (the
    # documented ordering), so the state saved with an epoch is that epoch's.
    assert early.on_validation_end(0, {"val_loss": 0.5}, None, None) is False
    assert ckpt.on_validation_end(0, {"val_loss": 0.5}, model, opt) is None
    paths = list((tmp_path).glob("best-*.pt"))
    saved = torch.load(paths[0], map_location="cpu")
    assert saved["callbacks"]["EarlyStopping"] == {
        "best_metric": 0.5, "epochs_since_improvement": 0}
    assert saved["callbacks"]["ModelCheckpoint"] == {"best_metric": 0.5}

    # a stall: patience advances, and no new file is written
    early.best_metric, early.epochs_since_improvement = 0.5, 1
    ckpt.on_validation_end(1, {"val_loss": 0.6}, model, opt)
    assert list(tmp_path.glob("best-*.pt")) == paths

    # resuming restores both callbacks
    fresh_early = EarlyStopping(monitor="val_loss", mode="min", patience=3)
    fresh_ckpt = ModelCheckpoint(monitor="val_loss", mode="min",
                                filepath=str(tmp_path / "other-{epoch}-{val_loss}.pt"))
    fresh_early.load_state_dict(saved["callbacks"]["EarlyStopping"])
    fresh_ckpt.load_state_dict(saved["callbacks"]["ModelCheckpoint"])
    assert fresh_early.best_metric == 0.5
    assert fresh_ckpt.best_metric == 0.5
    # an empty/absent state must be a no-op
    assert fresh_early.load_state_dict({}) is fresh_early


def test_early_stopping_stops_after_patience_when_resumed():
    """Seeded with the resumed best value, patience counts stalls from there."""
    early = EarlyStopping(monitor="val_loss", mode="min", patience=2)
    early.load_state_dict({"best_metric": 0.10, "epochs_since_improvement": 0})
    assert early.on_validation_end(5, {"val_loss": 0.11}, None, None) is False
    assert early.on_validation_end(6, {"val_loss": 0.12}, None, None) is True


# ---------------------------------------------------------------------------
# trainer resume
# ---------------------------------------------------------------------------
def _resumable(store, tmp_path, callbacks=None, epochs=1):
    return _trainer(store, RecordingModel(), DiceLoss(), ConstSampler(0.0),
                    epochs=epochs, callbacks=callbacks)


def test_load_checkpoint_restores_epoch_weights_optimizer_and_callbacks(
        real_store, tmp_path):
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    ckpt = ModelCheckpoint(
        monitor="val_loss", mode="min",
        filepath=str(tmp_path / "checkpoints/best-{epoch:02d}-{val_loss:.4f}.pt"),
        state_providers=[early])
    trainer = _resumable(real_store, tmp_path, callbacks=[early, ckpt])
    trainer.fit()
    path = next((tmp_path / "checkpoints").glob("best-*.pt"))
    weights = {k: v.clone() for k, v in trainer.model.state_dict().items()}
    assert early.best_metric is not None

    fresh_early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    fresh_ckpt = ModelCheckpoint(
        monitor="val_loss", mode="min",
        filepath=str(tmp_path / "checkpoints/best-{epoch:02d}-{val_loss:.4f}.pt"),
        state_providers=[fresh_early])
    resumed = _resumable(real_store, tmp_path, callbacks=[fresh_early, fresh_ckpt])
    resumed.model = RecordingModel()          # start from different weights
    info = resumed.load_checkpoint(path)

    assert resumed.current_epoch == 1          # epoch 0 completed
    assert info["epoch"] == 0 and info["continues_at"] == 1
    assert set(info["restored"]) == {"EarlyStopping", "ModelCheckpoint"}
    for key, value in weights.items():
        assert torch.equal(resumed.model.state_dict()[key], value)
    # plain SGD keeps no per-parameter state, so round-trip something visible
    resumed.optimizer.param_groups[0]["lr"] = 0.777
    resumed.load_checkpoint(path)
    assert resumed.current_epoch == 1
    assert resumed.optimizer.param_groups[0]["lr"] != 0.777
    assert fresh_early.best_metric == pytest.approx(early.best_metric)
    assert fresh_ckpt.best_metric == pytest.approx(trainer.callbacks[0].best_metric)


def test_resumed_fit_continues_the_epoch_counter(real_store, tmp_path):
    ckpt = ModelCheckpoint(
        monitor="val_loss", mode="min",
        filepath=str(tmp_path / "checkpoints/best-{epoch:02d}-{val_loss:.4f}.pt"))
    _resumable(real_store, tmp_path, callbacks=[ckpt]).fit()
    path = next((tmp_path / "checkpoints").glob("best-*.pt"))

    seen = []

    class _Capture:
        def on_validation_end(self, epoch, metrics, model, optimizer):
            seen.append((epoch, metrics["epoch"]))
            return False

    trainer = _resumable(real_store, tmp_path, epochs=3, callbacks=[_Capture()])
    trainer.load_checkpoint(path)
    trainer.fit()                              # fresh checkpoint callback list
    assert seen == [(1, 1), (2, 2)]            # continues at 1, target is absolute


def test_scheduled_sampling_probability_at_the_resumed_epoch(real_store, tmp_path):
    """The resumed epoch drives the sampler, not a restarted counter."""
    from wildfire_simulator.scheduled_sampler import ScheduledSampler

    sampler = ScheduledSampler(k=0.1, t0=40)
    assert sampler.get_prob(120) > 0.999        # self-driven, as at epoch 120
    assert sampler.get_prob(0) < 0.02           # what a restart would impose


# ---------------------------------------------------------------------------
# tensorboard continuity
# ---------------------------------------------------------------------------
def test_tensorboard_series_continues_across_a_resume(real_store, tmp_path):
    """A second process writing into the same run dir extends the same series."""
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )
    from torch.utils.tensorboard import SummaryWriter

    from finetune.tb import RealTensorBoardCallback

    run_dir = tmp_path / "run" / "training"

    def phase(steps_start, n_epochs):
        writer = SummaryWriter(str(run_dir / "val"))
        cb = RealTensorBoardCallback(train_writer=SummaryWriter(str(run_dir / "train")),
                                     val_writer=writer)
        trainer = _trainer(real_store, RecordingModel(), DiceLoss(),
                           ConstSampler(0.0), epochs=steps_start + n_epochs,
                           callbacks=[cb])
        trainer.current_epoch = steps_start       # as a resume would leave it
        trainer.fit()
        writer.close()

    phase(steps_start=0, n_epochs=2)
    phase(steps_start=2, n_epochs=2)              # resumed: 2 more epochs

    acc = EventAccumulator(str(run_dir / "val"))
    acc.Reload()
    assert [e.step for e in acc.Scalars("IOU")] == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# the script's checkpoint resolution
# ---------------------------------------------------------------------------
def test_resolve_resume_picks_the_lowest_val_loss(tmp_path):
    ckpt_dir = tmp_path / "checkpoints"
    ckpt_dir.mkdir()
    for name in ("best-model-10-0.3100.pt", "best-model-97-0.2900.pt",
                 "best-model-40-0.3050.pt"):
        (ckpt_dir / name).touch()
    assert resolve_resume("auto", tmp_path).name == "best-model-97-0.2900.pt"
    assert resolve_resume(None, tmp_path) is None
    explicit = ckpt_dir / "best-model-40-0.3050.pt"
    assert resolve_resume(str(explicit), tmp_path) == explicit


def test_check_resume_epochs_refuses_a_finished_target(tmp_path):
    import torch as _torch
    path = tmp_path / "best-model-01-0.5000.pt"
    _torch.save({"epoch": 1}, path)
    check_resume_epochs(path, 4)                      # room to run: fine
    for epochs in (2, 1):                             # nothing to run
        with pytest.raises(SystemExit, match="not beyond it"):
            check_resume_epochs(path, epochs)


def test_resolve_resume_fails_loudly(tmp_path):
    with pytest.raises(SystemExit, match="no best-model"):
        resolve_resume("auto", tmp_path)
    with pytest.raises(SystemExit, match="no such checkpoint"):
        resolve_resume(str(tmp_path / "nope.pt"), tmp_path)
