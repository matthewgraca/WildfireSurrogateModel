"""Tests for the real-data scheduled-sampling trainer (finetune.trainer)."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from wildfire_simulator.callbacks import ModelCheckpoint
from wildfire_simulator.losses import DiceLoss
from wildfire_simulator.models import MK_UNet_Regression
from finetune.dataset import (RealChainDataset, RealDataStore,
                                             collate_chains)
from finetune.trainer import (HOURS_PER_T_UNIT, RealChainTrainer,
                                             masked_iou)

NORM = "norm_cache/stats.json"
FIRE_A = "V5_2020_2020_00001"
FIRE_B = "V5_2020_2020_00002"


class ConstSampler:
    def __init__(self, prob):
        self.prob = prob

    def get_prob(self, epoch):
        return self.prob


class RecordingModel(torch.nn.Module):
    """Stub that returns a fixed mask and records every input it is given."""

    def __init__(self, value=1.0, box=(60, 70, 60, 70)):
        super().__init__()
        # A real parameter so the trainer can build an optimizer (the value is
        # unused: the stub's output is fixed).
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.value = value
        self.box = box
        self.inputs = []

    def forward(self, x):
        self.inputs.append(x.detach().clone())
        b, _, h, w = x.shape
        out = torch.zeros((b, 1, h, w), dtype=x.dtype, device=x.device)
        r0, r1, c0, c1 = self.box
        out[:, :, r0:r1, c0:c1] = self.value
        # a dependency on the parameter so backward() has something to do
        return [out + self.dummy * 0.0]


class CaptureLoss(torch.nn.Module):
    """Records what the trainer hands the loss; returns a differentiable value."""

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pred, target):
        self.calls.append((pred.detach().clone(), target.detach().clone()))
        return (pred - target).abs().mean()


def _loader(store, chains, batch_size=4):
    ds = RealChainDataset(store, chains, transform=store.transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      collate_fn=collate_chains)


def _batch(store, chains):
    return next(iter(_loader(store, chains)))


def _trainer(store, model, loss_fn, sampler, epochs=1, callbacks=None, amp=False):
    return RealChainTrainer(
        model=model,
        optimizer=torch.optim.SGD(model.parameters(), lr=0.01),
        loss_fn=loss_fn,
        train_loader_factory=lambda epoch: _loader(
            store, [(FIRE_A, 1, 2), (FIRE_B, 0, 3)]),
        val_loader=_loader(store, [(FIRE_B, 0, None)]),
        sampler=sampler,
        epochs=epochs,
        device=torch.device("cpu"),
        callbacks=callbacks,
        progress=False,
        amp=amp,
    )


# ---------------------------------------------------------------------------
# state / scheduled sampling contract
# ---------------------------------------------------------------------------
def test_trainer_shows_gt_state_when_sampler_forces_it(real_store):
    # get_prob is p(use the model's own forecast): 0.0 = always the observed
    # state (teacher forcing), which is the early-training regime.
    model = RecordingModel()
    trainer = _trainer(real_store, model, CaptureLoss(), ConstSampler(0.0))
    trainer._run(_loader(real_store, [(FIRE_B, 0, 3)]), 0, "t", train=True)

    chain = real_store.build_chain(FIRE_B, 0, chain_len=3)
    # step 0: the observed seed; step 1+: the observed burned state (not the
    # model's own forecast), cropped back from the 32-multiple padding.
    for k in range(chain.length):
        seen = model.inputs[k][0, 0, :135, :135]
        assert torch.equal(seen > 0.5, torch.from_numpy(chain.x[k, 0] > 0.5))


def test_trainer_rolls_its_own_prediction_when_sampler_says_so(real_store):
    # get_prob = 1.0: always the model's own forecast (fully self-driven)
    model = RecordingModel(value=1.0, box=(60, 70, 60, 70))
    trainer = _trainer(real_store, model, CaptureLoss(), ConstSampler(1.0))
    trainer._run(_loader(real_store, [(FIRE_B, 0, 3)]), 0, "t", train=True)

    chain = real_store.build_chain(FIRE_B, 0, chain_len=3)
    seed = torch.from_numpy(chain.perim0)
    box = torch.zeros_like(seed)
    box[60:70, 60:70] = True

    step0 = model.inputs[0][0, 0, :135, :135] > 0.5
    step1 = model.inputs[1][0, 0, :135, :135] > 0.5
    step2 = model.inputs[2][0, 0, :135, :135] > 0.5
    assert torch.equal(step0, seed)                           # step 0 = the seed
    assert torch.equal(step1, seed | box)                     # union with the forecast
    assert torch.equal(step2, seed | box)                     # monotone, never unburned
    assert not torch.equal(step1, torch.from_numpy(chain.x[1, 0] > 0.5))


def test_trainer_time_channel_is_hours_over_24(real_store):
    model = RecordingModel()
    trainer = _trainer(real_store, model, CaptureLoss(), ConstSampler(1.0))
    trainer._run(_loader(real_store, [(FIRE_B, 0, 3)]), 0, "t", train=True)
    chain = real_store.build_chain(FIRE_B, 0, chain_len=3)
    for k in range(chain.length):
        # last input channel, sampled at a burned cell to avoid pad zeros
        got = model.inputs[k][0, -1, :135, :135][chain.y[k]]
        assert torch.allclose(got, torch.full_like(
            got, float(chain.t_hours[k]) / HOURS_PER_T_UNIT))


def test_no_gradient_flows_through_the_state_handed_to_the_next_step(real_store):
    """The mask fed back is a detached hard mask: a step's loss trains that
    step only (ROLLOUT.md)."""
    model = RecordingModel()
    trainer = _trainer(real_store, model, CaptureLoss(), ConstSampler(0.0))
    trainer._run(_loader(real_store, [(FIRE_B, 0, 3)]), 0, "t", train=True)
    for seen in model.inputs:
        assert not seen.requires_grad


# ---------------------------------------------------------------------------
# masking
# ---------------------------------------------------------------------------
def test_loss_sees_only_scored_cells(real_store):
    loss = CaptureLoss()
    model = RecordingModel()
    trainer = _trainer(real_store, model, loss, ConstSampler(1.0))
    trainer._run(_loader(real_store, [(FIRE_B, 0, 3)]), 0, "t", train=True)

    chain = real_store.build_chain(FIRE_B, 0, chain_len=3)
    assert len(loss.calls) == chain.length
    with_unscored = 0
    for k, (pred, target) in enumerate(loss.calls):
        valid = torch.from_numpy(chain.scored[k])[None, None]
        assert (pred[~valid] == 0).all()          # nothing leaks from unscored cells
        assert (target[~valid] == 0).all()
        assert (target[valid] > 0).any()          # the truth survives
        if not valid.all():
            with_unscored += 1
    assert with_unscored >= 1, "the fixture must exercise an unscored region"


def test_padded_steps_are_skipped(real_store):
    """A short chain in the batch contributes no step beyond its own length."""
    loss = CaptureLoss()
    model = RecordingModel()
    trainer = _trainer(real_store, model, loss, ConstSampler(1.0))
    # FIRE_A from step 1 runs 2 more steps, FIRE_B runs 3
    loader = _loader(real_store, [(FIRE_A, 1, 2), (FIRE_B, 0, 3)])
    trainer._run(loader, 0, "t", train=True)
    assert [int(p.shape[0]) for p, _ in loss.calls] == [2, 2, 1]


def test_report_iou_matches_a_manual_computation_for_mixed_length_chains(real_store):
    """Chains of different lengths in one batch are scored per sample, not
    per (sample, step) pair: the reported IoU is the mean over samples of each
    sample's mean over its own steps, matching the per-fire ``avg_step_iou``.
    """
    from wildfire_simulator import real_spread as rs

    cases = ((0.0, "teacher-forced: state = the observed burned state"),
             (1.0, "self-driven: stub predicts nothing, state = the seed"))
    for prob, note in cases:
        model = RecordingModel(value=0.0)
        trainer = _trainer(real_store, model, CaptureLoss(), ConstSampler(prob))
        _, iou, _ = trainer._run(
            _loader(real_store, [(FIRE_A, 1, 2), (FIRE_B, 0, 3)]), 0, "t",
            train=True)

        per_sample = []
        for fire_id, start, length in ((FIRE_A, 1, 2), (FIRE_B, 0, 3)):
            chain = real_store.build_chain(fire_id, start, chain_len=length)
            states = []
            for k in range(length):
                states.append(chain.x[k, 0] > 0.5 if prob == 0.0
                              else chain.perim0)
            per_sample.append(np.mean([
                rs.iou(state, chain.y[k], chain.scored[k])
                for k, state in enumerate(states)
            ]))
        assert iou == pytest.approx(float(np.mean(per_sample)), abs=1e-6), note


def test_masked_iou_ignores_unscored_cells():
    pred = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    target = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    valid = torch.zeros((1, 1, 4, 4), dtype=torch.bool)
    pred[0, 0, 0, 0] = True
    target[0, 0, 0, 0] = True
    valid[0, 0, :2, :2] = True
    # a disagreement outside `valid` must not change the score
    pred[0, 0, 3, 3] = True
    assert masked_iou(pred, target, valid).item() == pytest.approx(1.0)
    target[0, 0, 1, 1] = True
    # inside valid: inter 1, union 2
    assert masked_iou(pred, target, valid).item() == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# fit / callbacks
# ---------------------------------------------------------------------------
def test_fit_trains_reports_metrics_and_checkpoints(real_store, tmp_path):
    torch.manual_seed(0)
    model = MK_UNet_Regression(in_channels=len(real_store.channels) + 1,
                               out_channels=1, channels=[4, 8, 16, 32, 48],
                               final_activation="sigmoid")
    before = {k: v.clone() for k, v in model.state_dict().items()}
    ckpt = ModelCheckpoint(
        monitor="val_loss", mode="min",
        filepath=str(tmp_path / "ckpts/best-{epoch:02d}-{val_loss:.4f}.pt"))
    trainer = _trainer(real_store, model, DiceLoss(), ConstSampler(1.0),
                       epochs=2, callbacks=[ckpt])
    metrics = trainer.fit()

    for key in ("train_loss", "train_iou", "val_loss", "val_iou"):
        assert np.isfinite(metrics[key]), key
    assert 0.0 <= metrics["val_iou"] <= 1.0
    assert any(not torch.equal(v, before[k])
               for k, v in model.state_dict().items())
    assert trainer.current_epoch == 2
    assert list((tmp_path / "ckpts").glob("best-*.pt"))


def test_evaluate_returns_validation_metrics(real_store):
    model = RecordingModel()
    trainer = _trainer(real_store, model, DiceLoss(), ConstSampler(0.0))
    out = trainer.evaluate()
    assert set(out) == {"val_loss", "val_iou"}
    assert 0.0 <= out["val_iou"] <= 1.0


def test_amp_is_gpu_only_and_trains_under_autocast(real_store, tmp_path):
    """amp on CPU is a no-op; on CUDA the bf16 forward must still train."""
    model = RecordingModel()
    cpu_trainer = _trainer(real_store, model, DiceLoss(), ConstSampler(0.0),
                           amp=True)
    assert cpu_trainer.amp is False            # no CUDA, so no autocast

    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    from wildfire_simulator.models import MK_UNet_Regression

    torch.manual_seed(0)
    gpu_model = MK_UNet_Regression(in_channels=len(real_store.channels) + 1,
                                   out_channels=1, channels=[4, 8, 16, 32, 48],
                                   final_activation="sigmoid")
    before = gpu_model.state_dict()["out.weight"].detach().cpu().clone()
    trainer = RealChainTrainer(
        model=gpu_model,
        optimizer=torch.optim.SGD(gpu_model.parameters(), lr=0.05),
        # BCELoss is the strict case: it rejects a bf16 prediction against an
        # fp32 target, so it pins "the loss always sees fp32" under autocast.
        loss_fn=torch.nn.BCELoss(),
        train_loader_factory=lambda epoch: _loader(
            real_store, [(FIRE_A, 1, 2), (FIRE_B, 0, 3)]),
        val_loader=_loader(real_store, [(FIRE_B, 0, 3)]),
        sampler=ConstSampler(0.0), epochs=1,
        device=torch.device("cuda"), amp=True, progress=False,
    )
    assert trainer.amp is True
    metrics = trainer.fit()
    assert np.isfinite(metrics["train_loss"]) and np.isfinite(metrics["val_loss"])
    assert 0.0 <= metrics["val_iou"] <= 1.0
    assert not torch.equal(
        gpu_model.state_dict()["out.weight"].detach().cpu(), before)


def test_store_and_trainer_agree_on_input_width(real_store):
    """The trainer builds C + 1 channels; the model must accept exactly that."""
    model = RecordingModel()
    trainer = _trainer(real_store, model, CaptureLoss(), ConstSampler(1.0))
    trainer._run(_loader(real_store, [(FIRE_B, 0, 2)]), 0, "t", train=True)
    assert model.inputs[0].shape[1] == len(real_store.channels) + 1


def test_where_broadcast_contract_this_trainer_relies_on():
    """Pins the framework behaviour the scheduled-sampling draw depends on.

    torch 2.14.1 in this environment miscomputes broadcast shapes for rank-3
    conditions with a leading dim > 1, so the trainer uses rank-4 (B,1,1,1)
    conditions. If this test starts failing after a torch upgrade, the trainer's
    condition shape can be simplified again — but the failure means the
    broadcast behaviour changed, not that the trainer is broken.
    """
    cond = torch.zeros(2, 1, 1, 1, dtype=torch.bool)
    a = torch.zeros(2, 1, 4, 5)
    b = torch.ones(2, 1, 4, 5)
    assert tuple(torch.where(cond, a, b).shape) == (2, 1, 4, 5)
    # the trainer's other full-shape operations
    assert tuple(torch.maximum(a, b).shape) == (2, 1, 4, 5)
    assert tuple(a.expand_as(b).shape) == (2, 1, 4, 5)
