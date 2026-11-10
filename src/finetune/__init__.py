"""Real-data track for the wildfire surrogate: the baseline and the fine-tune.

Everything that reads the real wildfire product (``v5_wildfires_*``: FEDS
perimeters on VIIRS, HRRR weather, LANDFIRE fuels, 135x135 windows, 12 h steps)
lives here, so ``wildfire_simulator`` stays the MTT/synthetic package:

* :mod:`finetune.dataset` — chains, normalization, torch dataset, collation,
* :mod:`finetune.trainer` — scheduled-sampling training on chains,
* :mod:`finetune.infer` — full-life autoregressive rollout and the model builder,
* :mod:`finetune.metrics` — the modular per-fire metric registry,
* :mod:`finetune.viz` — arrival-time / wind / direct-comparison figures,
* :mod:`finetune.transforms` — per-channel z-score normalization.

Dependency direction is one way: ``finetune`` imports the shared, real-data
primitives that already live in ``wildfire_simulator`` (the MK-UNet and its
padding rule, ``real_spread.shift_window``, ``real_spread.iou``, the scheduled
sampler, ``ScalarRNG``). Nothing in ``wildfire_simulator`` imports ``finetune``.
"""
