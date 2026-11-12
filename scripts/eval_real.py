#!/usr/bin/env python3
"""eval_real - metrics and visualizations for the real-data baseline.

Validation flow: the frozen ``val`` split is the input, every fire is rolled
forward autoregressively from its first real observed perimeter to the end of
its recorded life (``finetune.infer.rollout_fire``), and the metrics and figures
are computed from that rollout.

Metrics (modular: see ``finetune.metrics``, add one with ``register_metric``).
Reported by default:

  * ``final_iou``       - the final model footprint vs the fire's final burned
                          footprint (every perimeter unioned to the last step),
  * ``avg_step_iou``    - mean per-step IoU, i.e. step-by-step spread quality,
  * ``precision_final`` - of the cells called burned at the end, the fraction
                          that really burned (drops when the predicted union
                          runs away),
  * ``recall_final``    - of the cells that really burned, the fraction found
                          (stays high under that same saturation).

All four are recorded per fire and averaged over the validation set; the
precision/recall pair is what makes a low ``final_iou`` interpretable. Other
registered metrics (dice, arrival MAE, persistence baseline, ...) are opt-in
with ``--metrics name,name`` or ``--metrics all``.

Figures, one directory per fire (terrain from the real data, no basemap API)

  * ``final_arrival_comparison.png`` - real vs predicted fire arrival time
    (YlOrRd, early = yellow, late = red) over elevation,
  * ``wind_coherence.png``           - the same pair with the mean rollout wind
    as a quiver field,
  * ``direct_comparison.png``        - final real vs predicted footprint on
    elevation: yellow = both, blue = predicted only, red = real only, gray =
    the fire already burning at the forecast start.

    # whole validation set, 5 figures, default metrics
    python scripts/eval_real.py --run-dir /mnt/wildfire/surrogate-model/real-baseline \
        --real-data /mnt/wildfire/real-data/v5_wildfires_2026-10-08

    # one specific fire, extra metrics
    python scripts/eval_real.py --run-dir ... --real-data ... \
        --fire-name "Camp" --year 2020 --metrics all
"""

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import torch
import yaml
from tqdm import tqdm

from finetune import metrics as rm
from finetune import viz as rv
from finetune.dataset import RealDataStore
from finetune.infer import build_model, rollout_fire

_DEFAULT_METRICS = ",".join(rm.DEFAULT_METRICS)


def _load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _resolve_config(args):
    """Config from the run directory (preferred) or an explicit file."""
    candidates = []
    if args.run_dir:
        candidates.append(Path(args.run_dir) / "config.resolved.yaml")
    if args.config:
        candidates.append(Path(args.config))
    for path in candidates:
        if path.is_file():
            return _load_yaml(path), path
    raise SystemExit("no config found; pass --run-dir (with a trained run) or "
                     "--config")


def _resolve_checkpoint(args, cfg):
    if args.checkpoint:
        return Path(args.checkpoint)
    run_dir = args.run_dir or cfg.get("output_dir")
    if not run_dir:
        raise SystemExit("pass --checkpoint or --run-dir")
    ckpt_dir = Path(run_dir) / "checkpoints"
    files = sorted(ckpt_dir.glob("best-model-*.pt"))

    def val_loss(p):
        try:
            return float(p.stem.split("-")[-1])
        except ValueError:
            return float("inf")

    if not files:
        raise SystemExit(f"no checkpoints in {ckpt_dir}")
    return min(files, key=val_loss)


def _metric_names(spec):
    if spec in (None, "", "default"):
        return list(rm.DEFAULT_METRICS)
    if spec == "all":
        return rm.available_metrics()
    return [s.strip() for s in spec.split(",") if s.strip()]


def _slug(name, fire_id):
    keep = [c if (c.isalnum() or c in "-_") else "_" for c in str(name)]
    return f"{''.join(keep)}_{fire_id}"


def _print_summary(summary, names):
    print("\n=== validation summary ===")
    print(f"{'metric':<22}{'mean':>10}{'median':>10}{'std':>10}"
          f"{'min':>10}{'max':>10}{'n':>6}")
    for name in names:
        s = summary.get(name)
        if not s:
            print(f"{name:<22}{'n/a':>10}")
            continue
        print(f"{name:<22}{s['mean']:>10.4f}{s['median']:>10.4f}"
              f"{s['std']:>10.4f}{s['min']:>10.4f}{s['max']:>10.4f}{s['n']:>6}")


def make_deterministic(device):
    """Pin evaluation to bit-reproducible kernels.

    Reported metrics are measurements: two invocations of the same checkpoint on
    the same data must agree exactly. They do not by default. cuDNN's
    ``benchmark`` autotuning picks a convolution algorithm by timing, and that
    choice varies between processes; with output probabilities sitting within
    float noise of the 0.5 mask threshold, a few cells flip, and a single fire's
    IoU moves by ~1 pp (measured: 43 of 109 val fires, up to ±153 cells,
    per-fire IoU up to 0.014 — while the set-wide mean moves only 4e-4).

    Training deliberately keeps ``benchmark`` on: it is a speed win there, and
    training runs are not claimed to be bit-reproducible anyway.
    """
    if device.type != "cuda":
        return
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def main(argv=None):
    p = argparse.ArgumentParser(description="real-data baseline evaluation")
    p.add_argument("--run-dir", help="trained run directory (checkpoints + config)")
    p.add_argument("--checkpoint", help="explicit checkpoint path")
    p.add_argument("--config", help="explicit config (used with --checkpoint)")
    p.add_argument("--real-data", help="override the real-data root")
    p.add_argument("--data-cache", help="decompressed .npy sample cache "
                                       "(default: from the run's config)")
    p.add_argument("--output-dir", help="default <run-dir>/real_eval")
    p.add_argument("--split-key", default="val",
                   help="frozen split to score ('val' default, or 'all')")
    p.add_argument("--metrics", default=_DEFAULT_METRICS,
                   help=f"comma-separated metric names, or 'all' "
                        f"(default {_DEFAULT_METRICS})")
    p.add_argument("--num-visuals", type=int, default=5,
                   help="how many fires to draw figures for (0 = none)")
    p.add_argument("--fire-id", help="visualize a specific real-data fire id")
    p.add_argument("--fire-name", help="fire name to visualize")
    p.add_argument("--year", type=int, help="year paired with --fire-name")
    p.add_argument("--max-fires", type=int, default=None,
                   help="score only the first N fires (debug)")
    p.add_argument("--metrics-only", action="store_true")
    p.add_argument("--vis-only", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default=None)
    args = p.parse_args(argv)

    cfg, cfg_path = _resolve_config(args)
    real_data = args.real_data or cfg.get("real_data")
    if not real_data:
        p.error("--real-data or `real_data` in the config is required")

    device = torch.device(args.device or
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    make_deterministic(device)
    checkpoint = _resolve_checkpoint(args, cfg)
    out_dir = Path(args.output_dir) if args.output_dir else (
        (Path(args.run_dir) if args.run_dir else Path.cwd()) / "real_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    metric_names = _metric_names(args.metrics)

    store = RealDataStore(real_data, channels=cfg.get("data_channels"),
                          norm_stats=cfg.get("norm_stats"),
                          arrays_cache=args.data_cache or cfg.get("data_cache"))
    model = build_model(store, cfg.get("model_channels"))
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.to(device).eval()
    print(f"config:     {cfg_path}")
    print(f"checkpoint: {checkpoint} (epoch {state.get('epoch', '?')})")
    print(f"{store.describe()}")
    print(f"device: {device}   metrics: {metric_names}")

    all_fire_ids = [str(f) for f in store.fires["fire_id"]]
    if args.split_key == "all":
        fire_ids = all_fire_ids
    else:
        fire_ids = store.split_fires(args.split_key)
    if args.max_fires is not None:
        fire_ids = fire_ids[:args.max_fires]
    print(f"scoring {len(fire_ids)} fires from split {args.split_key!r}")

    # --- inference + metrics ---------------------------------------------
    rollouts, per_fire, skipped = {}, {}, []
    if not args.vis_only:
        for fire_id in tqdm(fire_ids, desc="Rollouts", unit="fire"):
            n = store.n_samples(fire_id)
            if n - store.first_fire_step(fire_id) < 2:
                skipped.append(fire_id)
                continue
            rollout = rollout_fire(model, store, fire_id, device=device)
            rollouts[fire_id] = rollout
            per_fire[fire_id] = rm.compute_metrics(rollout, metric_names)
    else:
        print("skipping metrics (--vis-only)")

    summary = rm.aggregate(per_fire) if per_fire else {}
    if per_fire:
        _print_summary(summary, metric_names)
    if skipped:
        print(f"skipped {len(skipped)} fires without a full-life rollout: "
              f"{skipped}")

    # --- choose fires to draw -------------------------------------------
    vis_fires, outside_split = [], []
    if args.fire_id or args.fire_name:
        fire_id = store.resolve_fire(name=args.fire_name, year=args.year,
                                     fire_id=args.fire_id)
        vis_fires = [fire_id]
        if fire_id not in per_fire:
            outside_split = [fire_id]
    elif args.num_visuals and not args.metrics_only:
        # When nothing was scored (--vis-only) fall back to the split's fires.
        pool = sorted(per_fire) or sorted(fire_ids)
        pool = [f for f in pool
                if store.n_samples(f) - store.first_fire_step(f) >= 2]
        vis_fires = sorted(random.Random(args.seed).sample(
            pool, min(args.num_visuals, len(pool))))

    # --- figures ----------------------------------------------------------
    written = []
    per_fire_extra = {}
    for fire_id in vis_fires:
        rollout = rollouts.get(fire_id)
        if rollout is None:
            rollout = rollout_fire(model, store, fire_id, device=device)
        subtitle = ""
        metrics = per_fire.get(fire_id)
        if metrics is None:
            metrics = rm.compute_metrics(rollout, metric_names)
            per_fire_extra[fire_id] = metrics
        bits = [f"{k}={metrics[k]:.3f}" for k in metric_names if k in metrics]
        if bits:
            subtitle = "  ".join(bits)
        fig_dir = out_dir / "fires" / _slug(rollout.name, fire_id)
        paths = {
            "final_arrival_comparison.png":
                rv.render_final_arrival_comparison(rollout, fig_dir / "final_arrival_comparison.png", subtitle),
            "wind_coherence.png":
                rv.render_wind_coherence(rollout, fig_dir / "wind_coherence.png", subtitle),
            "direct_comparison.png":
                rv.render_direct_comparison(rollout, fig_dir / "direct_comparison.png", subtitle),
        }
        written.append({"fire_id": fire_id, "dir": str(fig_dir), **paths})
        print(f"  figures: {fig_dir}")

    # --- dump -------------------------------------------------------------
    metrics_path = out_dir / "metrics.json"
    payload = {
        "config": str(cfg_path),
        "checkpoint": str(checkpoint),
        "checkpoint_epoch": state.get("epoch"),
        "real_data": str(real_data),
        "split_key": args.split_key,
        "metrics": metric_names,
        "n_fires": len(per_fire),
        "summary": summary,
        "per_fire": per_fire,
        "skipped": skipped,
        "visualized": [v["fire_id"] for v in written],
        # fires drawn but outside the scored set (e.g. --fire-name, --max-fires)
        "per_fire_extra": per_fire_extra,
        "figures": written,
    }
    if per_fire:
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=2)
        csv_path = out_dir / "per_fire.csv"
        keys = ["fire_id", "name", "year", "state", "county", "start_step",
                "n_steps", "seed_cells", "real_cells", "pred_cells",
                *metric_names]
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            for fire_id, m in per_fire.items():
                rollout = rollouts[fire_id]
                w.writerow({
                    "fire_id": fire_id, "name": rollout.name,
                    "year": rollout.year, "state": rollout.state,
                    "county": rollout.county, "start_step": rollout.start_step,
                    "n_steps": rollout.n_steps, "seed_cells": rollout.seed_cells,
                    "real_cells": rollout.n_cells,
                    "pred_cells": rollout.pred_cells, **m,
                })
        print(f"wrote {metrics_path}")
        print(f"wrote {csv_path}")

    if outside_split:
        print(f"note: {outside_split} is outside the scored set "
              f"(split {args.split_key!r}, {len(fire_ids)} fires scored); its "
              f"metrics are in metrics.json per_fire_extra and excluded from "
              f"the summary")
    print(f"figures: {out_dir / 'fires'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
