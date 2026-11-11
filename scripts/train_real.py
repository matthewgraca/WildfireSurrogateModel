#!/usr/bin/env python3
"""train_real - train the MK-UNet surrogate on real wildfire data only.

This is the real-data baseline: the same MK-UNet architecture as the MTT
surrogate, trained from scratch on the v5 wildfire chains
(``/mnt/wildfire/real-data/v5_wildfires_*``) with scheduled sampling. No MTT
data is read and no pretrained weights are loaded.

Each epoch, every training fire is cut into back-to-back 12-step (6-day) chains
from a random start (``finetune.dataset.fire_chain_cuts``), so the cut points move
from epoch to epoch, and the fires are streamed fire by fire so each one is
decompressed once per worker per epoch (``finetune.dataset.StreamingChainDataset``).
``ROLLOUT.md``'s schedule is used:

  * the model sees each step's own meteorology (the chains carry it),
  * with probability ``1/(1 + exp(-k*(epoch - t0)))`` a step is given the
    observed burned state instead of the model's own forecast,
  * no gradient flows back through the mask handed to the next step,
  * burned land does not unburn.

Validation during training holds out the frozen ``val`` split: one deterministic
chain per fire from its first real sample, reported as ``val_loss`` (checkpoint
selection) and ``val_iou`` (mean per-step IoU). The headline per-fire metrics
and the figures come from ``scripts/eval_real.py``.

    python scripts/train_real.py --config configs/real_baseline.yaml \
        --output-dir /mnt/wildfire/surrogate-model/real-baseline
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from wildfire_simulator.callbacks import EarlyStopping, ModelCheckpoint
from wildfire_simulator.losses import DiceLoss
from wildfire_simulator.scheduled_sampler import ScheduledSampler
from wildfire_simulator.utils import ScalarRNG
from wildfire_simulator.viz import select_viz_indices

from finetune.dataset import (RealChainDataset, RealDataStore,
                              StreamingChainDataset, build_val_chains,
                              collate_chains)
from finetune.infer import build_model
from finetune.tb import RealTensorBoardCallback
from finetune.trainer import RealChainTrainer


def load_real_config(path):
    """Load the real-data training config, applying defaults."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    base = Path(path).resolve().parent
    if cfg.get("real_data"):
        cfg["real_data"] = str((base / cfg["real_data"]).resolve()
                               if not Path(cfg["real_data"]).is_absolute()
                               else Path(cfg["real_data"]))
    if cfg.get("norm_stats") and not Path(cfg["norm_stats"]).is_absolute():
        cfg["norm_stats"] = str(base / cfg["norm_stats"])
    cfg.setdefault("data_channels", None)
    cfg.setdefault("split_key", "train")
    cfg.setdefault("val_split_key", "val")
    cfg.setdefault("chain_len", 12)
    cfg.setdefault("val_chain_len", cfg["chain_len"])
    cfg.setdefault("extra_chains", True)
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("amp", True)
    cfg.setdefault("data_cache", None)
    cfg.setdefault("num_workers", 8)
    cfg.setdefault("lr", 5e-4)
    cfg.setdefault("weight_decay", 1e-4)
    cfg.setdefault("epochs", 120)
    cfg.setdefault("early_stopping", False)
    cfg.setdefault("patience", 20)
    cfg.setdefault("loss", "dice")
    cfg.setdefault("scheduled_sampling", {"k": 0.1, "t0": 40})
    cfg.setdefault("model_channels", [16, 32, 64, 96, 160])
    cfg.setdefault("cache_gb", 0.6)
    cfg.setdefault("seed", 42)
    # TensorBoard observability: tracked validation samples every N epochs.
    cfg.setdefault("viz_every", 5)
    cfg.setdefault("viz_fires", 6)
    return cfg


def build_loss(name):
    if name == "dice":
        return DiceLoss()
    if name == "bce":
        return torch.nn.BCELoss()
    raise ValueError(f"unknown loss {name!r} (expected 'dice' or 'bce')")


def resolve_resume(value, out_dir):
    """Checkpoint path to continue from.

    ``value`` is an explicit path, or ``"auto"`` for the lowest-``val_loss``
    checkpoint in ``<out_dir>/checkpoints`` (files are only written when the
    monitored metric improves, so the best one is also the most recent).
    """
    if value is None:
        return None
    if value != "auto":
        path = Path(value)
        if not path.is_file():
            raise SystemExit(f"--resume {value}: no such checkpoint")
        return path
    ckpt_dir = Path(out_dir) / "checkpoints"
    files = sorted(ckpt_dir.glob("best-model-*.pt"))

    def val_loss(p):
        try:
            return float(p.stem.split("-")[-1])
        except ValueError:
            return float("inf")

    if not files:
        raise SystemExit(f"--resume auto: no best-model-*.pt in {ckpt_dir}")
    return min(files, key=val_loss)


def check_resume_epochs(checkpoint_path, epochs):
    """Refuse a resume that has nothing to run, before workers/writers exist.

    ``--epochs`` is the run's total target, so it has to be beyond the resumed
    epoch; catching it here means a typo does not spawn dataloader workers or
    leave an empty TensorBoard event file in the run directory.
    """
    done = torch.load(checkpoint_path, map_location="cpu").get("epoch")
    if done is not None and done + 1 >= epochs:
        raise SystemExit(
            f"--resume {Path(checkpoint_path).name} is at epoch {done} and "
            f"--epochs {epochs} is not beyond it: --epochs is the total target, "
            f"not the number of extra epochs"
        )


def main(argv=None):
    p = argparse.ArgumentParser(description="train the real-data MK-UNet baseline")
    p.add_argument("--config", default=str(Path(__file__).parent.parent /
                                           "configs/real_baseline.yaml"))
    p.add_argument("--output-dir", required=True,
                   help="run directory (checkpoints, TensorBoard, resolved config)")
    p.add_argument("--real-data", help="override the real-data root")
    p.add_argument("--data-channels", help="comma-separated channel names override")
    p.add_argument("--norm-stats", help="override the normalization-stats JSON")
    p.add_argument("--data-cache", help="decompressed .npy cache of the samples "
                                       "(see scripts/prepare_real_cache.py)")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--num-workers", type=int)
    p.add_argument("--chain-len", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--viz-every", type=int,
                   help="log tracked-sample figures every N epochs (0 = off)")
    p.add_argument("--viz-fires", type=int, help="how many fires to track")
    p.add_argument("--resume", nargs="?", const="auto", default=None,
                   help="continue a previous run: a checkpoint path, or omit "
                        "the value for 'auto' (the lowest-val_loss checkpoint "
                        "in <output-dir>/checkpoints). Uses the same "
                        "--output-dir so TensorBoard keeps one series")
    p.add_argument("--early-stopping", action=argparse.BooleanOptionalAction,
                   default=None, help="stop when val_loss stalls (default: config)")
    p.add_argument("--patience", type=int, default=None,
                   help="stalled epochs tolerated by --early-stopping")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None,
                   help="bf16 autocast forward (default: from the config)")
    p.add_argument("--limit-fires", type=int, default=None,
                   help="use only the first N train fires (smoke tests)")
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    cfg = load_real_config(args.config)
    for key in ("real_data", "norm_stats", "data_cache"):
        value = getattr(args, key)
        if value:
            cfg[key] = value
    if args.data_channels:
        cfg["data_channels"] = [c.strip() for c in args.data_channels.split(",")
                                if c.strip()]
    for key in ("epochs", "batch_size", "lr", "num_workers", "chain_len", "seed",
                "viz_every", "viz_fires", "amp", "early_stopping", "patience"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
            if key == "chain_len":
                cfg["val_chain_len"] = value

    if not cfg["real_data"]:
        p.error("--real-data or `real_data` in the config is required")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(cfg["seed"])

    # Resolve --resume now and refuse an impossible target before anything is
    # built; the weights/optimizer/callback state are loaded once the trainer
    # exists (below).
    resume_path = resolve_resume(args.resume, out_dir)
    if resume_path is not None:
        check_resume_epochs(resume_path, cfg["epochs"])
        print(f"resume: {resume_path}")

    store = RealDataStore(
        cfg["real_data"], channels=cfg["data_channels"],
        norm_stats=cfg["norm_stats"],
        cache_bytes=int(float(cfg["cache_gb"]) * (1 << 30)),
        arrays_cache=cfg["data_cache"],
    )
    transform = store.transform()
    train_fires = store.split_fires(cfg["split_key"])
    val_fires = store.split_fires(cfg["val_split_key"])
    if args.limit_fires is not None:
        train_fires = train_fires[:args.limit_fires]
        val_fires = val_fires[:args.limit_fires]

    model = build_model(store, cfg["model_channels"])
    n_params = sum(q.numel() for q in model.parameters())
    print(store.describe())
    print(f"device: {device}   params: {n_params/1e6:.2f} M   "
          f"in_channels: {len(store.channels) + 1}   "
          f"amp: {bool(cfg['amp']) and device.type == 'cuda'}")
    print(f"train fires: {len(train_fires)}   val fires: {len(val_fires)}")
    print(f"channels: {store.channels}")

    def train_loader_factory(epoch):
        # Streaming: each fire is decompressed once per worker and all of its
        # chains for this epoch come from RAM (see StreamingChainDataset).
        dataset = StreamingChainDataset(
            store, train_fires, cfg["chain_len"], cfg["seed"], epoch=epoch,
            transform=transform, extra_chains=cfg["extra_chains"],
            num_workers=cfg["num_workers"])
        return DataLoader(
            dataset, batch_size=cfg["batch_size"], num_workers=cfg["num_workers"],
            collate_fn=collate_chains, pin_memory=(device.type == "cuda"),
            drop_last=False, persistent_workers=False,
        )

    val_chains = build_val_chains(store, val_fires, cfg["val_chain_len"])
    val_loader = DataLoader(
        RealChainDataset(store, val_chains, transform=transform),
        batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["num_workers"], collate_fn=collate_chains,
        pin_memory=(device.type == "cuda"), drop_last=False,
    )
    print(f"train fires: {len(train_fires)} streamed over "
          f"{cfg['num_workers']} workers   val chains: {len(val_loader.dataset)}")

    optimizer = torch.optim.AdamW(model.parameters(), cfg["lr"],
                                  weight_decay=cfg["weight_decay"])
    loss_fn = build_loss(cfg["loss"])
    sampler = ScheduledSampler(k=cfg["scheduled_sampling"]["k"],
                               t0=cfg["scheduled_sampling"]["t0"])

    callbacks = []
    # EarlyStopping first: ModelCheckpoint reads its state_dict() when it writes
    # a file, so listing it before makes the saved patience current for the
    # epoch that is being saved.
    early_stop_cb = None
    if cfg["early_stopping"]:
        early_stop_cb = EarlyStopping(monitor="val_loss", mode="min",
                                      patience=int(cfg["patience"]))
        callbacks.append(early_stop_cb)
    callbacks.append(ModelCheckpoint(
        monitor="val_loss", mode="min",
        filepath=str(out_dir / "checkpoints/best-model-{epoch:02d}-{val_loss:.4f}.pt"),
        state_providers=[early_stop_cb] if early_stop_cb else (),
    ))
    callbacks.append(RealTensorBoardCallback(
        train_writer=SummaryWriter(str(out_dir / "training/train")),
        val_writer=SummaryWriter(str(out_dir / "training/val")),
    ))

    # TensorBoard observability: watch a fixed subset of the validation chains
    # across the whole run (deterministic positions, so the same fires are
    # tracked every epoch; see finetune.tb for the tags).
    viz_every = int(cfg["viz_every"] or 0)
    viz_positions = (select_viz_indices(len(val_loader.dataset),
                                       n=int(cfg["viz_fires"]), seed=cfg["seed"])
                     if viz_every > 0 else [])
    print(f"tensorboard: {out_dir}/training/val   (IOU per epoch; "
          f"{len(viz_positions)} tracked fires every {viz_every} epochs)")
    if viz_positions:
        print("tracked fires: " + ", ".join(
            val_loader.dataset.chains[i][0] for i in viz_positions))

    # Persist the resolved config next to the checkpoints: eval_real.py reads it
    # so the model, channels and normalization cannot drift apart from the run.
    resolved = dict(cfg)
    resolved["data_channels"] = list(store.channels)
    resolved["norm_stats"] = store.norm_stats_path
    resolved["data_cache"] = (str(store.arrays_cache)
                              if store.arrays_cache is not None else None)
    resolved["in_channels"] = len(store.channels) + 1
    with open(out_dir / "config.resolved.yaml", "w") as f:
        yaml.safe_dump(resolved, f, sort_keys=False)
    print(f"wrote {out_dir / 'config.resolved.yaml'}")

    trainer = RealChainTrainer(
        model=model, optimizer=optimizer, loss_fn=loss_fn,
        train_loader_factory=train_loader_factory, val_loader=val_loader,
        sampler=sampler, epochs=cfg["epochs"], device=device,
        callbacks=callbacks, rng=ScalarRNG(), amp=cfg["amp"],
        viz_every=viz_every, viz_positions=viz_positions,
        # one full training chain = the arrival montage's 1.0
        fat_norm_hours=cfg["chain_len"] * 12.0,
    )
    if resume_path is not None:
        info = trainer.load_checkpoint(resume_path)
        print(f"  epoch {info['epoch']} completed; continuing at "
              f"{info['continues_at']} (target {cfg['epochs']})")
        if info['callbacks']:
            print(f"  callback state: {info['callbacks']}")
        print(f"  tensorboard continues in {out_dir}/training (same run dir)")

    metrics = trainer.fit()
    print(f"finished: epoch={metrics.get('epoch')} "
          f"train_loss={metrics.get('train_loss'):.4f} "
          f"val_loss={metrics.get('val_loss'):.4f} "
          f"val_iou={metrics.get('val_iou'):.4f} "
          f"seconds={metrics.get('seconds', float('nan')):.1f}")
    print("next: python scripts/eval_real.py "
          f"--run-dir {out_dir} --real-data {cfg['real_data']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
