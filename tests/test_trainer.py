import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import torch.nn as nn
from pathlib import Path
import os
import shutil
import re
import numpy as np

from wildfire_simulator.callbacks import ModelCheckpoint, TensorBoardCallback, EarlyStopping
from wildfire_simulator.datasets import WildfireDataset, TransformedDataset
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.losses import DiceLoss
from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.trainers import ForwardBurnTrainer, BurnerBatchProcessor
from wildfire_simulator.scheduled_sampler import ScheduledSampler
from wildfire_simulator.utils import ScalarRNG

def test_batch_processor(dataloader):
    dataset = WildfireDataset(dataloader)

    burner = ForwardBurnProcess()

    sampler = ScheduledSampler(k=0.1, t0=40)

    class ConstSampler():
        def __init__(self, prob):
            self.prob = prob
        def get_prob(self, epoch):
            return self.prob

    class DummyScalarRNG:
        def __init__(self, constant_value = 0.5):
            self.constant_value = constant_value
            self.last_seed_used = None

        def seed(self, seed_value):
            self.last_seed_used = seed_value

        def rand(self):
            return torch.tensor(self.constant_value, dtype=torch.float32)

    rng = DummyScalarRNG(0.5)
    batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=30,
        eval=False,
        sampler=ConstSampler(0.4),
        rng=rng
    )
    assert batch_processor.dt == 30

    pred = torch.stack([dataset[1], dataset[0]])
    true = torch.stack([dataset[0], dataset[1]])

    input_tensor, output_tensor = batch_processor(pred, true, epoch=6, batch_idx=7, t=60)
    assert rng.last_seed_used == 60007

    input_tensor_expected = torch.load("tests/baseline/batch_processor/input_forced.pt")
    output_tensor_expected = torch.load("tests/baseline/batch_processor/output_forced.pt")

    assert (input_tensor == input_tensor_expected).all()
    # Target is now only mask channel (channel 0)
    assert (output_tensor == output_tensor_expected[:, 0:1]).all()

    rng = DummyScalarRNG(0.2)
    batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=30,
        eval=False,
        sampler=ConstSampler(0.3),
        rng=rng
    )
    assert batch_processor.dt == 30

    input_tensor, output_tensor = batch_processor(pred, true, epoch=8, batch_idx=9, t=90)
    assert rng.last_seed_used == 80009

    batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=30,
        eval=True
    )
    assert batch_processor.dt == 30

    pred = torch.stack([dataset[1], dataset[0]])
    true = torch.stack([dataset[0], dataset[1]])

    input_tensor2, output_tensor2 = batch_processor(pred, true, epoch=10, batch_idx=11, t=90)

    input_tensor_expected = torch.load("tests/baseline/batch_processor/input_autoreg.pt")
    output_tensor_expected = torch.load("tests/baseline/batch_processor/output_autoreg.pt")

    assert (input_tensor == input_tensor_expected).all()
    assert (input_tensor2 == input_tensor_expected).all()
    assert (output_tensor == output_tensor_expected[:, 0:1]).all()
    assert (output_tensor2 == output_tensor_expected[:, 0:1]).all()


def test_trainer(dataloader):
    torch.manual_seed(42)

    dataset = WildfireDataset(dataloader)
    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    dataset = TransformedDataset(dataset, transform)

    burner = ForwardBurnProcess()

    train_batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=1/48,
        eval=False,
        sampler=ScheduledSampler(k=0.1, t0=40),
        rng=ScalarRNG()
    )
    val_batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=1/48,
        eval=True
    )

    # share loader for train and val
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    def get_trainer(epochs):
        model = MK_UNet_Regression(
            in_channels=14,
            out_channels=1,
            channels=[16, 32, 64, 96, 160],
            final_activation='sigmoid'
        )

        checkpoint_cb = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            filepath='./checkpoints_test/best-model-{epoch:02d}-{val_loss:.2f}.pt'
        )

        train_writer = SummaryWriter("training_test/train")
        val_writer = SummaryWriter("training_test/val")

        tensorboard_cb = TensorBoardCallback(
            train_writer=train_writer,
            val_writer=val_writer
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            5e-4,
            weight_decay=1e-4
        )

        trainer = ForwardBurnTrainer(
            model=model,
            optimizer=optimizer,
            loss_fn=nn.BCELoss(),
            train_loader=loader,
            val_loader=loader,
            train_batch_processor = train_batch_processor,
            val_batch_processor = val_batch_processor,
            callbacks=[checkpoint_cb, tensorboard_cb],
            epochs=epochs,
            max_t=3/48
        )

        return trainer

    shutil.rmtree('./training_test', ignore_errors=True)

    trainer = get_trainer(epochs=5)

    eval_before = trainer.evaluate()
    assert isinstance(eval_before['val_loss'], float)

    shutil.rmtree('./checkpoints_test', ignore_errors=True)

    trainer.fit()

    eval_after = trainer.evaluate()
    assert eval_after['val_loss'] < eval_before['val_loss']

    folder = Path('./checkpoints_test')
    pattern = re.compile(r"best-model-\d{2}-\d+\.\d{2}\.pt")
    matching_files = [
        p for p in folder.iterdir() 
        if p.is_file() and pattern.fullmatch(p.name)
    ]
    assert matching_files

    last_checkpoint = max(matching_files, key=lambda p: p.name)

    trainer = get_trainer(epochs=10)
    trainer.load_checkpoint(last_checkpoint)

    eval_before_resumed = trainer.evaluate()
    assert eval_before_resumed['val_loss'] <= eval_after['val_loss']

    trainer.fit()

    eval_after_resumed = trainer.evaluate()
    assert eval_after_resumed['val_loss'] < eval_after['val_loss']

    train_acc = EventAccumulator("training_test/train")
    train_acc.Reload()
    assert len(set(s.step for s in train_acc.Scalars("Loss"))) == 10

    val_acc = EventAccumulator("training_test/val")
    val_acc.Reload()
    assert len(set(s.step for s in val_acc.Scalars("Loss"))) == 10




def test_per_scene_val_metrics(dataloader):
    torch.manual_seed(0)

    dataset = WildfireDataset(dataloader)
    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    tds = TransformedDataset(dataset, transform)

    burner = ForwardBurnProcess()
    train_bp = BurnerBatchProcessor(
        burner=burner, dt=1/48, eval=False,
        sampler=ScheduledSampler(k=0.1, t0=40), rng=ScalarRNG(),
    )
    val_bp = BurnerBatchProcessor(burner=burner, dt=1/48, eval=True)

    loader = DataLoader(tds, batch_size=1, shuffle=False, num_workers=0)

    model = MK_UNet_Regression(
        in_channels=14, out_channels=1,
        channels=[8, 16, 16, 16, 16], final_activation='sigmoid',
    )

    # Spy callback captures the metrics dict emitted at validation end.
    class SpyCallback:
        def __init__(self):
            self.metrics = None
        def on_validation_end(self, epoch, metrics, model, optimizer):
            self.metrics = metrics

    spy = SpyCallback()

    trainer = ForwardBurnTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), 5e-4),
        loss_fn=nn.BCELoss(),
        train_loader=loader,
        val_loader=loader,
        val_loaders={"old": loader, "new": loader},
        train_batch_processor=train_bp,
        val_batch_processor=val_bp,
        callbacks=[spy],
        epochs=1,
        max_t=2/48,
    )

    trainer.fit()

    # Combined and both per-scene val losses are present and finite.
    assert 'val_loss' in spy.metrics
    assert 'val_loss/old' in spy.metrics
    assert 'val_loss/new' in spy.metrics
    assert isinstance(spy.metrics['val_loss/old'], float)
    assert isinstance(spy.metrics['val_loss/new'], float)


def test_early_stopping_stops_after_patience_stalls():
    # min mode: an improvement is a strictly lower val_loss.
    cb = EarlyStopping(monitor='val_loss', mode='min', patience=3)

    # The first epoch always counts as an improvement.
    assert cb.on_validation_end(0, {'val_loss': 1.0}, None, None) is False
    assert cb.on_validation_end(1, {'val_loss': 0.9}, None, None) is False  # improvement
    assert cb.on_validation_end(2, {'val_loss': 0.8}, None, None) is False  # improvement
    # Three consecutive stalls -> stop on the third.
    assert cb.on_validation_end(3, {'val_loss': 0.85}, None, None) is False  # stall 1
    assert cb.on_validation_end(4, {'val_loss': 0.9}, None, None) is False   # stall 2
    assert cb.on_validation_end(5, {'val_loss': 0.95}, None, None) is True   # stall 3 -> stop


def test_early_stopping_resets_counter_on_improvement():
    cb = EarlyStopping(monitor='val_loss', mode='min', patience=2)
    assert cb.on_validation_end(0, {'val_loss': 1.0}, None, None) is False  # improvement (best None)
    assert cb.on_validation_end(1, {'val_loss': 1.1}, None, None) is False  # stall 1
    # A new best resets the consecutive-stall counter.
    assert cb.on_validation_end(2, {'val_loss': 0.5}, None, None) is False  # improvement
    assert cb.on_validation_end(3, {'val_loss': 0.6}, None, None) is False  # stall 1
    assert cb.on_validation_end(4, {'val_loss': 0.7}, None, None) is True   # stall 2 -> stop


def test_early_stopping_max_mode():
    cb = EarlyStopping(monitor='accuracy', mode='max', patience=1)
    assert cb.on_validation_end(0, {'accuracy': 0.5}, None, None) is False  # improvement
    assert cb.on_validation_end(1, {'accuracy': 0.4}, None, None) is True   # 1 stall -> stop


def test_trainer_stops_early(dataloader):
    # fit() must break when a callback's on_validation_end returns truthy,
    # rather than running the full epoch budget.
    torch.manual_seed(0)
    dataset = WildfireDataset(dataloader)
    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    tds = TransformedDataset(dataset, transform)
    burner = ForwardBurnProcess()
    train_bp = BurnerBatchProcessor(
        burner=burner, dt=1/48, eval=False,
        sampler=ScheduledSampler(k=0.1, t0=40), rng=ScalarRNG(),
    )
    val_bp = BurnerBatchProcessor(burner=burner, dt=1/48, eval=True)
    loader = DataLoader(tds, batch_size=1, shuffle=False, num_workers=0)
    model = MK_UNet_Regression(
        in_channels=14, out_channels=1,
        channels=[8, 16, 16, 16, 16], final_activation='sigmoid',
    )

    class StopAfterNCalls:
        def __init__(self, n):
            self.n = n
            self.calls = 0
        def on_validation_end(self, epoch, metrics, model, optimizer):
            self.calls += 1
            return self.calls >= self.n

    stopper = StopAfterNCalls(n=2)
    trainer = ForwardBurnTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), 5e-4),
        loss_fn=nn.BCELoss(),
        train_loader=loader,
        val_loader=loader,
        train_batch_processor=train_bp,
        val_batch_processor=val_bp,
        callbacks=[stopper],
        epochs=5,
        max_t=2/48,
    )

    trainer.fit()

    # fit() honored the stop signal: only 2 epochs ran despite epochs=5.
    assert stopper.calls == 2
    assert trainer.current_epoch == 2


def test_validate_iou_perfect_predictor(dataloader):
    # A model that returns the processor's target mask at every step is a
    # perfect predictor: its rollout ends on the GT burned-so-far mask at
    # t = max_t, so the final-mask IoU must be exactly 1.0. (The previous
    # inverted ground-truth expression produced an empty GT mask and 0.0.)
    dataset = WildfireDataset(dataloader)
    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    dataset = TransformedDataset(dataset, transform)
    burner = ForwardBurnProcess()
    proc = BurnerBatchProcessor(burner=burner, dt=1/8, eval=True)

    state = {"targets": None}

    class _PerfectModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Dummy parameter so AdamW has something to own.
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return state["targets"]

    class _TargetCapturingProcessor:
        def __init__(self, inner):
            self._inner = inner
            self.dt = inner.dt
            self.device = inner.device

        def __call__(self, *args, **kwargs):
            inputs, targets = self._inner(*args, **kwargs)
            state["targets"] = targets
            return inputs, targets

    wrapped = _TargetCapturingProcessor(proc)
    model = _PerfectModel()

    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False,
                        drop_last=False, num_workers=0)
    trainer = ForwardBurnTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), 5e-4),
        loss_fn=DiceLoss(),
        train_loader=loader,
        val_loader=loader,
        train_batch_processor=wrapped,
        val_batch_processor=wrapped,
        epochs=1,
        max_t=1.0,
        device=torch.device("cpu"),
    )

    val_loss, val_iou, _ = trainer._validate(0, 1)
    assert val_iou == 1.0


def test_validate_records_positions_across_batches(dataloader):
    # Recorded positions must track the loader position across batches.
    # (The position counter used to stay at 0, so only first-batch
    # positions ever matched: multi-batch val sets recorded nothing, and
    # first-batch records accumulated frames from every batch.)
    base = WildfireDataset(dataloader)
    transform = MinMaxPerChannel(base.min_val, base.max_val)
    base = TransformedDataset(base, transform)

    class _Repeated(Dataset):
        def __init__(self, ds, times):
            self.ds, self.times = ds, times

        def __len__(self):
            return len(self.ds) * self.times

        def __getitem__(self, idx):
            return self.ds[idx % len(self.ds)]

    dataset = _Repeated(base, 4)  # 8 samples, batch_size 2 -> 4 batches
    proc = BurnerBatchProcessor(burner=ForwardBurnProcess(), dt=1/8, eval=True)

    state = {"targets": None}

    class _PerfectModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return state["targets"]

    class _TargetCapturingProcessor:
        def __init__(self, inner):
            self._inner = inner
            self.dt = inner.dt
            self.device = inner.device

        def __call__(self, *args, **kwargs):
            inputs, targets = self._inner(*args, **kwargs)
            state["targets"] = targets
            return inputs, targets

    wrapped = _TargetCapturingProcessor(proc)
    model = _PerfectModel()

    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=False,
                        drop_last=False, num_workers=0)
    trainer = ForwardBurnTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), 5e-4),
        loss_fn=DiceLoss(),
        train_loader=loader,
        val_loader=loader,
        train_batch_processor=wrapped,
        val_batch_processor=wrapped,
        epochs=1,
        max_t=1.0,
        device=torch.device("cpu"),
    )

    record_indices = [4, 5, 6, 7]  # last two batches
    _, _, records = trainer._validate(0, 1, record_indices=record_indices)

    num_steps = 7  # len(np.arange(1/8, 1.0, 1/8)): last step's t+dt hits 1.0
    assert set(records) == set(record_indices)
    for p in record_indices:
        rec = records[p]
        assert len(rec['pred_history']) == num_steps
        assert len(rec['gt_history']) == num_steps
        # GT identity: the final frame must be sample p's own burn state
        # at t = max_t (mask/FAT zeroed where FAT > max_t).
        sample = dataset[p][0:2]
        expected = sample.clone()
        not_burnt = sample[1:2] > 1.0
        expected[0:1][not_burnt] = 0.0
        expected[1:2][not_burnt] = 0.0
        assert torch.allclose(rec['gt_history'][-1], expected)


def test_train_step1_self_input_seeded_with_gt(dataloader):
    # Under full scheduled sampling (prob=1), every sample is self-driven at
    # step 1. The step-1 input must be the GT state at dt (fire burn + full
    # landscape), matching _validate's first-step initialization. Previously
    # the training rollout was seeded with an all-zero tensor, so the model
    # saw a blank input (no landscape, no fire seed) at step 1 and could not
    # predict the next burn.
    dataset = WildfireDataset(dataloader)
    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)
    dataset = TransformedDataset(dataset, transform)
    burner = ForwardBurnProcess()

    class ConstSampler:
        def get_prob(self, epoch):
            return 1.0

    proc = BurnerBatchProcessor(
        burner=burner, dt=1/8, eval=False,
        sampler=ConstSampler(), rng=ScalarRNG(),
    )

    state = {"step1_inputs": None, "step": 0}

    class _RecordingProcessor:
        def __init__(self, inner):
            self._inner = inner
            self.dt = inner.dt
            self.device = inner.device

        def __call__(self, *args, **kwargs):
            inputs, targets = self._inner(*args, **kwargs)
            if state["step"] == 0:
                state["step1_inputs"] = inputs
            state["step"] += 1
            return inputs, targets

    wrapped = _RecordingProcessor(proc)

    class _ZerosModel(nn.Module):
        def __init__(self):
            super().__init__()
            # Dummy parameter (≈0) so the output is differentiable and AdamW
            # owns a parameter.
            self.dummy = nn.Parameter(torch.zeros(1))

        def forward(self, x):
            return self.dummy * torch.ones(x.shape[0], 1, x.shape[2], x.shape[3])

    model = _ZerosModel()

    loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True,
                        drop_last=False, num_workers=0)
    trainer = ForwardBurnTrainer(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), 5e-4),
        loss_fn=DiceLoss(),
        train_loader=loader,
        val_loader=loader,
        train_batch_processor=wrapped,
        val_batch_processor=wrapped,
        epochs=1,
        max_t=1.0,
        device=torch.device("cpu"),
    )

    trainer._train_epoch(0, 1)

    inputs = state["step1_inputs"]
    assert inputs is not None
    # Landscape channels (2:13) must carry the real terrain/fuel/wind, not
    # zeros. This is the regression: a blank init erased them at step 1.
    assert inputs[:, 2:13, :, :].abs().sum() > 0
    # The fire seed must be present too: some mask pixels burned at dt.
    H, W = inputs.shape[-2], inputs.shape[-1]
    gt_burn = inputs[:, 0, :H, :W] > 0
    assert gt_burn.sum() > 0
    # Step count matches the actual rollout (7), not max_t/dt (8).
    assert state["step"] == 7 * len(loader)
