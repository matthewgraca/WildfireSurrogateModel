# WildfireSimulator

## Environment
```bash
conda env create -f environment.yml
conda activate wildfire_env
pip install -e .
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # or thru here: https://pytorch.org/get-started/locally/
```

## Data
Data is not traditionally saved in a `.npy` file for training. Rather, `.tifs` are directly converted into the correct format for training in RAM through a DataLoader. Create a `.env` file that points to these tifs:

```bash
LANDSCAPE=/run/media/hrotovb001/9021-552E/FireDataset/palisades.tif
TRIALS=/run/media/hrotovb001/9021-552E/FireDataset/Trials/
IGNITIONS=/run/media/hrotovb001/9021-552E/FireDataset/Ignitions/
```

Currently there are 3112 samples. Each on the Palisades landscape, with parameterized:
- Ignition point (ign)
- Wind speed (ws)
- Wind direction (wd)
- Moisture (moisture)

```mermaid
graph TD
    subgraph "Raw Data (on disk)"
        A["palisades.tif<br/>(8-band landscape)"]
        B["Ignitions/*.shp<br/>(50 point shapefiles)"]
        C["Trials/*.tif<br/>(3211 arrival time rasters)"]
    end

    subgraph "Data Loading"
        D["WildfireDataLoader"]
        E["TrialCollection<br/>(lazy file loader)"]
    end

    subgraph "Dataset Assembly"
        F["WildfireDataset<br/>→ (13, 500, 500) per sample"]
        G["MinMaxPerChannel<br/>→ normalized [0,1]"]
        H["TransformedDataset"]
    end

    subgraph "Training-Time Processing"
        I["BurnerBatchProcessor(t)"]
        J["ForwardBurnProcess<br/>(slices fire state at time t)"]
        K["Input: (14, 512, 512)<br/>[fire@t + landscape + t_channel]"]
        L["Target: (2, 512, 512)<br/>[fire_mask@(t+dt), arrival@(t+dt)]"]
    end

    subgraph "Model"
        M["MK_UNet_Regression<br/>in=14, out=2"]
    end

    A --> D
    B --> D
    C --> E
    E --> D
    D --> F
    F --> G
    G --> H
    H --> I
    I --> J
    J --> K
    J --> L
    K --> M
    M -->|"pred (2, 512, 512)"| I
```

## Run training
Simply through `python run.py`.

## Real-fire inference (infer_real.py)
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

## Real-data spread comparison (compare_real_spread.py)
`scripts/compare_real_spread.py` optionally compares the model's **per-step**
inferred spread against a real per-step fire product (NASA FEDS perimeters on
VIIRS). It reuses a completed `infer_real.py` run and matches its fire
(`fire.json` name/year, disambiguated by the incident-number suffix, or an
explicit `--fire-id`/`--inc`) to a real-data folder
(`fires.csv` + `samples.csv` + `samples/<fire_id>.npz`). Each real-data step is
aligned to the nearest model step by real time (favoring the model slightly
ahead on ties) and scored by cell-wise IoU over the model's crop coverage.

Time is squared away: the surrogate window (`result.json` `start`/`end`) is UTC,
and the real-data `t_input`/`t_label` are the fire's **local** wall-clock time
(the dataset carries no timezone column). The adapter maps the fire's
`centroid_lon` to a nominal IANA US zone (approximate longitude bands) and
converts every real timestamp to UTC via `zoneinfo`, so the two timelines are
compared in a single (UTC) frame. The chain is anchored at the first sample that
actually holds fire (the observed perimeter above the small pre-fire
placeholder), and its **first frame is the initial observed perimeter** (the
small ignition fire) at that sample's `t_input`, with labels accumulated
afterward — so the real data's initial fire is shown rather than a large merged
first frame. It writes, into `<run_dir>/results/real_compare/` (or `--out`):

- `metrics.json` — per-step aligned IoU, average step IoU, final footprint IoU,
  and a `calfire` block giving the model and the VIIRS cumulative burned
  compared to the CAL FIRE reference on the chain window (IoU, Dice, precision,
  recall), plus the model's official polygon IoU (`model_iou_polygon`).
- `spread_compare.gif` — side-by-side real-vs-model spread per aligned step,
- `arrival_compare.png` — side-by-side first-burn (arrival) time maps,
- `combined_fat_map.png` — one map overlaying the basemap, the CALFIRE
  reference perimeter outline, the VIIRS fire-arrival raster (viridis), the
  model fire-arrival raster (YlOrRd), and the ignition point (magenta star). The
  two arrival rasters share one time scale so they are directly comparable.
- `timeline_gantt.png` — two-row timeline (surrogate model top, VIIRS data
  bottom) with every frame's timestamp plotted as a labelled marker (`M{i}` /
  `R{k}`) and a dashed connector from each VIIRS frame to the model frame it is
  aligned to (labelled with the per-pair IoU) — the FAT alignment is readable at
  a glance. A green square marks the surrogate's **ignition seed** on the model
  row (and a dotted green line marks the ignition time); the model's forecast
  window is shaded.

Only the model's raw `rollout.npz` masks are used (never the cleaned
largest-component perimeter) and only the real data's fire-spread channels. The
CAL FIRE reference perimeter is read only to draw the outline in
`combined_fat_map.png`; it is never used to compute any metric.

```bash
python scripts/compare_real_spread.py \
  --run-dir farsite-docker/FireBehaviorModels/SampleData/2025_palisades \
  --real-data /mnt/wildfire/real-data/v5_wildfires_2026-10-08
```
