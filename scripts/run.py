import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from wildfire_simulator.models import MK_UNet_Regression
from wildfire_simulator.checkpoints import init_from_checkpoint
from wildfire_simulator.callbacks import ModelCheckpoint, TensorBoardCallback, EarlyStopping
from wildfire_simulator.forward_burn_process import ForwardBurnProcess
from wildfire_simulator.trainers import ForwardBurnTrainer, BurnerBatchProcessor
from wildfire_simulator.datasets import build_multiscene_dataset, TransformedDataset
from wildfire_simulator.transforms import MinMaxPerChannel
from wildfire_simulator.scheduled_sampler import ScheduledSampler
from wildfire_simulator.losses import DiceLoss, DistanceMapLoss
from wildfire_simulator.utils import ScalarRNG
from wildfire_simulator.config import load_config
from wildfire_simulator.viz import select_viz_indices

def main():
    parser = argparse.ArgumentParser(description="Train wildfire surrogate model")
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Directory to save checkpoints and training logs (default: current directory)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file (default: config.yaml in project root)')
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Build the multi-scene dataset (one scene per landscape/trials/ignitions),
    # concatenated under a single shared normalization.
    dataset = build_multiscene_dataset(config)

    # Stratified per-scene 80/20 split: guarantees every scene (including the
    # terrain-aware ones) is present in validation, and yields per-scene val
    # index sets for separate metric tracking.
    train_indices, val_indices, per_scene_val = dataset.stratified_split(
        val_frac=0.2, seed=42
    )
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, val_indices)

    transform = MinMaxPerChannel(dataset.min_val, dataset.max_val)

    train_set = TransformedDataset(train_set, transform)
    test_set = TransformedDataset(test_set, transform)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=16,
        shuffle=True,
        num_workers=4,
    )
    
    val_loader = DataLoader(
        dataset=test_set,
        batch_size=16,
        shuffle=False,
        drop_last=False,
        num_workers=4,
    )

    # One validation loader per scene (over that scene's held-out val indices),
    # so per-scene val_loss is tracked separately during training.
    val_loaders = {}
    for scene_name, scene_val_indices in per_scene_val.items():
        if not scene_val_indices:
            continue
        scene_val_set = TransformedDataset(
            Subset(dataset, scene_val_indices), transform
        )
        val_loaders[scene_name] = DataLoader(
            dataset=scene_val_set,
            batch_size=16,
            shuffle=False,
            drop_last=False,
            num_workers=4,
        )

    # TensorBoard image viz: fixed per-scene subset of validation samples,
    # recorded (and rendered) every viz_every epochs. Seed 42 keeps the
    # tracked samples identical to the split/subsample convention, so the
    # same samples are watched across the whole run.
    viz_every = int(config.get('viz_every', 10))
    viz_n = int(config.get('viz_samples_per_scene', 10))
    viz_record_indices = {}
    if viz_every > 0:
        for scene_name, scene_val_indices in per_scene_val.items():
            if not scene_val_indices:
                continue
            viz_record_indices[scene_name] = select_viz_indices(
                len(scene_val_indices), n=viz_n, seed=42
            )
    
    model = MK_UNet_Regression(
        in_channels=config['in_channels'],
        out_channels=1,
        channels=[16, 32, 64, 96, 160],
        final_activation='sigmoid'
    )

    # Fine-tune: optionally initialize weights from a prior checkpoint
    # (weights only — fresh optimizer, epoch counter starts at 0). No-op when
    # unset, so from-scratch runs are unaffected.
    if config['finetune']['init_checkpoint']:
        init_from_checkpoint(model, config['finetune']['init_checkpoint'])
        print(f"Initialized weights from {config['finetune']['init_checkpoint']}")
    

    # Loss selection: "dice" (default, existing behavior) or "dml"
    # (Distance Map Loss, Caliva et al. 2019 — boundary-weighted cross-entropy).
    loss_name = config.get('loss', 'dice')
    if loss_name == 'dice':
        loss_fn = DiceLoss()
    elif loss_name == 'dml':
        loss_fn = DistanceMapLoss()
    else:
        raise ValueError(f"Unknown loss '{loss_name}' (expected 'dice' or 'dml')")
    checkpoint_cb = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        filepath=os.path.join(output_dir, 'checkpoints/best-model-{epoch:02d}-{val_loss:.2f}.pt')
    )

    # Early stopping is opt-out: set ``early_stopping: false`` in the config
    # to train the full ``epochs`` count unconditionally.
    early_stop_cb = None
    if config.get('early_stopping', True):
        early_stop_cb = EarlyStopping(
            monitor='val_loss',
            mode='min',
            patience=20
        )

    train_writer = SummaryWriter(os.path.join(output_dir, "training/train"))
    val_writer = SummaryWriter(os.path.join(output_dir, "training/val"))

    tensorboard_cb = TensorBoardCallback(
        train_writer=train_writer,
        val_writer=val_writer
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        config['lr'],
        weight_decay=1e-4
    )

    class ConstSampler():
        def __init__(self, prob):
            self.prob = prob
        def get_prob(self, epoch):
            return self.prob

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    burner = ForwardBurnProcess()
    sampler = ScheduledSampler(
        k=config['scheduled_sampling']['k'],
        t0=config['scheduled_sampling']['t0']
    )
    # sampler = ConstSampler(0.0)
    rng = ScalarRNG()
    train_batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=config['dt'],
        eval=False,
        sampler=sampler,
        rng=rng,
        device=device
    )
    val_batch_processor = BurnerBatchProcessor(
        burner=burner,
        dt=config['dt'],
        eval=True,
        device=device
    )
    
    trainer = ForwardBurnTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        val_loaders=val_loaders,
        train_batch_processor=train_batch_processor,
        val_batch_processor=val_batch_processor,
        callbacks=[cb for cb in [checkpoint_cb, early_stop_cb, tensorboard_cb]
                   if cb is not None],
        epochs=config['epochs'],
        max_t=config['max_t'],
        device=device,
        viz_every=viz_every,
        viz_record_indices=viz_record_indices,
    )

    trainer.fit()

    # trainer.load_checkpoint("./checkpoints/best-model-04-0.00.pt")
    # print(trainer.evaluate())

if __name__ == "__main__":
    main()
