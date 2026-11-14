# Real-fire inference (`infer_real.py`)

`scripts/infer_real.py` runs the trained surrogate end-to-end on a **real** CAL
FIRE fire: it resolves the ignition point + reference perimeter (via the
farsite-docker `calfire_ignition.py`), ingests and resamples a LANDFIRE
landscape to fit the model's fixed 500×500 crop, drives HRRR→WindNinja
time-varying winds per autoregressive step, and rolls the model forward to
produce:
- `run_dir/results/result.json` — IoU between the predicted final polygon and
  the real reference perimeter (+ areas, cell size, crop span),
- `run_dir/spread.gif` — spread over time over a basemap + the real perimeter +
  the ignition marker,
- `run_dir/arrival_map.png` — a single fire-arrival-time map (same overlay).

The model checkpoint stores only weights; per-channel min/max are computed from
the training dataset by `scripts/compute_norm_stats.py` into an `.npz` that
`infer_real.py` reads as `model.norm_stats`.

`[simulation] start` is the **ignition** time. The model was trained on fire
states already grown to `t=dt` (~16 cells, the smallest observed state), so the
driver places the 16-cell seed **at** `start` and emits the first perimeter one
`dt` later (perimeters follow every `dt`, last at `end`). The model window is
shifted internally so the seed lands on `start`; `result.json` records the
shifted window in `start`/`end` and keeps the ignition in `ignition_date`.

Optionally set `[simulation] seed_from_real = true` (+ `real_data_root`) for an
**initialized forecast**: instead of the synthetic 16-cell disc, the model is
seeded with the real data's **first observed perimeter** (resampled onto the
crop), and the window auto-sizes so each `dt` step is **12 h** (matching the
product), anchored at the real data's first observation (`R0`). `start`/`end`
are overridden in this mode; `result.json` `ignition_date` = `R0`.

```bash
# 1. generate per-channel norm stats from the training data the checkpoint used
python scripts/compute_norm_stats.py \
  --landscape <train landscape.tif> --trials <train Trials/> \
  --ignitions <train Ignitions/> --out ./norm-stats.npz
# 2. fill configs/infer.example.toml, then:
python scripts/infer_real.py --config configs/infer.example.toml
```

Note: the pipeline deliberately drives **time-varying** wind (HRRR per hour),
which is out-of-distribution vs. training (the model only ever saw one static
wind field), so low IoU may reflect an unfair comparison — the goal is the
comparison, surfaced in `result.json`. When the 500-cell crop cannot fit the
fire at 30 m, the landscape is **resampled coarser** (logged via `resample.json`
and printed prominently); this changes the fire's effective resolution.
