#!/usr/bin/env python3
"""plot_duration_metrics - metric vs fire length from an evaluation run.

Reads an ``eval_real.py`` output directory (``metrics.json`` + ``per_fire.csv``)
and writes one figure: each metric's median over quantile bins of fire length,
with the inter-quartile range as the error bar. No per-fire scatter — 109+ points
overplot into noise, and the bins carry both the trend and the spread.

The interesting question it answers is whether prediction quality degrades as a
fire keeps burning. ``avg_step_iou`` says how well the model spreads step by
step; ``precision_final`` and ``recall_final`` say *why* the footprint metric
moves — recall saturating with precision collapsing is the model filling the
window (over-prediction), the reverse is missed spread. ``final_iou`` is
excluded by default because it degenerates into a ratio of true to predicted
area once the prediction engulfs the fire; ask for it explicitly if wanted.

Usage:
    python scripts/plot_duration_metrics.py --run-dir /mnt/wildfire/.../20261109-run
    python scripts/plot_duration_metrics.py --eval-dir <run>/real_eval \
        --metrics avg_step_iou,precision_final,recall_final --bins 8

The run directory may be either a training run (its ``real_eval/`` subdirectory
is used) or an evaluation directory itself.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from finetune.viz import duration_bin_summary, render_duration_metrics

# What the figure plots unless --metrics says otherwise.
DEFAULT_SERIES = ("avg_step_iou", "precision_final", "recall_final")


def resolve_eval_dir(run_dir=None, eval_dir=None):
    """Accept a training run dir (with ``real_eval/``) or an eval dir directly."""
    for candidate in (eval_dir, run_dir):
        if candidate is None:
            continue
        path = Path(candidate)
        for probe in (path, path / "real_eval"):
            if (probe / "metrics.json").is_file() and (probe / "per_fire.csv").is_file():
                return probe
    raise SystemExit(
        "no evaluation artifacts found: pass --run-dir <training run> "
        "(expects real_eval/ inside) or --eval-dir <dir with metrics.json "
        "and per_fire.csv>")


def load_per_fire(eval_dir):
    """Per-fire table; fails loudly when the columns this figure needs are gone."""
    df = pd.read_csv(Path(eval_dir) / "per_fire.csv")
    if "n_steps" not in df.columns:
        raise SystemExit(f"{eval_dir}/per_fire.csv has no n_steps column")
    return df


def resolve_metrics(spec, df, eval_dir):
    """Metric columns to plot: an explicit list, or 'all' = every registered one
    that this evaluation happens to have computed."""
    if spec == "all":
        from finetune.metrics import available_metrics
        chosen = [m for m in available_metrics() if m in df.columns]
        if not chosen:
            raise SystemExit(f"{eval_dir}/per_fire.csv has no registered metric "
                             f"columns to plot")
        return chosen
    metrics = [m.strip() for m in spec.split(",") if m.strip()]
    missing = [m for m in metrics if m not in df.columns]
    if missing:
        raise SystemExit(
            f"{eval_dir}/per_fire.csv has no {missing}: that evaluation did not "
            f"compute them. Re-run eval_real.py with a --metrics list that "
            f"includes them (or --metrics all).")
    return metrics


def main(argv=None):
    p = argparse.ArgumentParser(
        description="plot per-fire metrics against fire length",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", help="training run directory (uses its real_eval/)")
    p.add_argument("--eval-dir", help="evaluation directory (metrics.json + per_fire.csv)")
    p.add_argument("--output", help="figure path (default <eval-dir>/duration_metrics.png)")
    p.add_argument("--metrics", default=",".join(DEFAULT_SERIES),
                   help=f"comma-separated metric columns, or 'all' for every "
                        f"registered metric present (default "
                        f"{','.join(DEFAULT_SERIES)})")
    p.add_argument("--bins", type=int, default=6, help="quantile bins (default 6)")
    p.add_argument("--title", default=None, help="figure title override")
    args = p.parse_args(argv)

    eval_dir = resolve_eval_dir(args.run_dir, args.eval_dir)
    df = load_per_fire(eval_dir)
    metrics = resolve_metrics(args.metrics, df, eval_dir)
    meta = json.load(open(Path(eval_dir) / "metrics.json"))

    title = args.title or (
        f"metric vs fire length — {len(df)} fires, "
        f"checkpoint epoch {meta.get('checkpoint_epoch')}, split {meta.get('split_key')}\n"
        f"median per duration bin, bars = inter-quartile range")
    summary = duration_bin_summary({m: df[m].to_numpy() for m in metrics},
                                   df["n_steps"].to_numpy(), bins=args.bins)
    out = Path(args.output) if args.output else eval_dir / "duration_metrics.png"
    path = render_duration_metrics(summary, out, title=title)

    print(f"{len(df)} fires from {eval_dir}")
    print(f"{'bin (steps)':>14} {'n':>4} " +
          " ".join(f"{m[:14]:>16}" for m in metrics))
    for row in summary["bins"]:
        cells = " ".join(f"{row[m]['median']:>10.3f}"
                         f" [{row[m]['q1']:.2f},{row[m]['q3']:.2f}]" for m in metrics)
        print(f"{row['lo']:6.0f}-{row['hi']:6.0f} {row['n']:>4} {cells}")
    print("\nSpearman vs fire length: " + ", ".join(
        f"{m} ρ={summary['spearman'][m][0]:+.3f} (p={summary['spearman'][m][1]:.1e})"
        for m in metrics))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
