#!/usr/bin/env python3
"""compute_norm_stats - per-channel min/max from the training dataset(s).

The WildfireSurrogateModel checkpoint stores only the network weights; the
per-channel min/max used for normalization are computed on the training data
at dataset-build time and are NOT persisted in the .pt. This helper rebuilds
the exact ``WildfireDataset`` statistics path used at training time (so a value
normalized here is comparable to what the model saw), and writes the result to
a small .npz the inference driver loads as ``model.norm_stats``.

Two modes:

1. ``--config <training config.yaml>`` - build the SAME ``MultiSceneDataset``
   the checkpoint was trained on (the config's ``scenes`` list, including any
   per-scene ``limit``/``seed`` subsampling), and take its SHARED normalization
   (per-channel min = min across all scenes, max = max across all scenes).
   Use this for any checkpoint trained on one or more scenes.

2. ``--landscape/--trials/--ignitions`` - a single scene, computed directly via
   ``WildfireDataset`` (used by the repo's tests/baseline and single-scene runs).

Example
-------
    # multi-scene (reflect the exact training normalization):
    python scripts/compute_norm_stats.py --config config.yaml --out ./norm-stats.npz

    # single scene:
    python scripts/compute_norm_stats.py \
      --landscape <train landscape.tif> --trials <train Trials/> \
      --ignitions <train Ignitions/> --out ./norm-stats.npz
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from wildfire_simulator.dataloader import (
    TrialCollection,
    TrialFileLoader,
    WildfireDataLoader,
)
from wildfire_simulator.datasets import WildfireDataset


def build_parser():
    p = argparse.ArgumentParser(
        prog="compute_norm_stats.py",
        description="Compute per-channel min/max over the training data and "
                    "write them to an .npz for use by infer_real.py.",
    )
    p.add_argument("--config", default=None,
                   help="training config.yaml with a [scenes] list; builds the "
                        "same MultiSceneDataset (shared normalization) the "
                        "checkpoint was trained on")
    p.add_argument("--landscape", default=None,
                   help="single-scene mode: 8-band training landscape GeoTIFF")
    p.add_argument("--trials", default=None,
                   help="single-scene mode: directory of arrival-time trials")
    p.add_argument("--ignitions", default=None,
                   help="single-scene mode: directory of ignition_<N>.shp "
                        "seed points")
    p.add_argument("--out", required=True,
                   help="output .npz path (written with min_val/max_val arrays)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.config:
        from wildfire_simulator.config import load_config
        from wildfire_simulator.datasets import build_multiscene_dataset
        config = load_config(args.config)
        dataset = build_multiscene_dataset(config)
        n_scenes = dataset.num_scenes()
        source = f"{args.config} ({n_scenes} scene(s))"
    elif args.landscape and args.trials and args.ignitions:
        trials = TrialCollection(TrialFileLoader(), trials_dir=args.trials)
        loader = WildfireDataLoader(
            trials,
            landscape_path=args.landscape,
            ignitions_dir=args.ignitions,
        )
        dataset = WildfireDataset(loader)
        source = f"single scene ({args.landscape})"
    else:
        print("error: provide either --config <config.yaml> or all of "
              "--landscape/--trials/--ignitions", file=sys.stderr)
        raise SystemExit(2)

    np.savez(
        args.out,
        min_val=np.asarray(dataset.min_val, dtype=np.float64),
        max_val=np.asarray(dataset.max_val, dtype=np.float64),
    )
    print(f"norm stats saved: {args.out}   [{source}]")
    print(f"  min: {np.array2string(dataset.min_val, precision=4)}")
    print(f"  max: {np.array2string(dataset.max_val, precision=4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
