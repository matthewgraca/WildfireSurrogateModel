# Real-data spread comparison (`compare_real_spread.py`)

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
