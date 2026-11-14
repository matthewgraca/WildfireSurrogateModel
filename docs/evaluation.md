# Evaluation: metrics and visualizations

Evaluation of the real-data baseline with `scripts/eval_real.py`, plus the
aggregate analysis script. Training and the run layout are described in
[the real-data baseline guide](real-baseline.md).

## Metrics (modular)

Each fire is rolled forward from its first observed perimeter to the end
of its recorded life, then scored per fire. Four metrics are reported by
default:

- **`final_iou`** — the final model footprint against the fire's final burned
  footprint (every perimeter unioned to the last timestep),
- **`avg_step_iou`** — mean per-step IoU over the whole rollout,
- **`precision_final`** — of the cells the model calls burned at the end, the
  fraction that really burned,
- **`recall_final`** — of the cells that really burned, the fraction the model
  found.

The precision/recall pair is there because `final_iou` alone is ambiguous: once
the predicted union engulfs the fire, IoU degenerates into a ratio of true to
predicted area, so a long rollout looks "worse" purely because its truth is
small. Reported together, the pair separates over-prediction (precision
collapses, recall stays high) from missed spread (the reverse).

All four are averaged over the validation set and written to
`<out>/metrics.json` (+ `<out>/per_fire.csv`). Evaluation is **deterministic**:
`eval_real.py` disables cuDNN autotuning and enables deterministic kernels,
because a timing-based convolution-algorithm choice otherwise flips cells whose
probability sits at the 0.5 mask threshold — measured at 43 of 109 val fires, up
to ±153 cells and 0.014 IoU per fire, while the set-wide mean moved only 4e-4.
With the pin, two invocations of the same checkpoint reproduce `per_fire.csv`
exactly. Training keeps autotuning on (it is a speed win) and does not claim
bit-reproducibility.

Other registered metrics slot in
with `--metrics name,name` (or `--metrics all`): `median_step_iou`,
`dice_final`, `growth_ratio`, `persistence_final_iou`, `arrival_mae_hours`. Add
one without touching the script:

```python
from finetune import metrics as rm

@rm.register_metric("my_metric", "what it measures")
def _my_metric(rollout: rm.FireRollout) -> float:
    ...
```

Absolute IoU alone is a weak signal on this product (a large fire filling the
window scores high even when it is mostly false positives, and 60 % of samples
have exactly zero growth), which is why the precision/recall,
`persistence_final_iou` and `growth_ratio` companions are one flag away.

## Visualizations

Per fire, all over the elevation under the fire (from the real data's `elev`
channel — no basemap or API call), for a seeded sample of fires
(`--num-visuals`, default 5) or one named fire
(`--fire-name "Claremont" --year 2020`, or `--fire-id`):

| file | content |
|---|---|
| `final_arrival_comparison.png` | real vs predicted fire arrival time on terrain, `YlOrRd` on one shared hour scale (start of the fire yellow → end red) |
| `wind_coherence.png` | the same pair with the rollout-mean wind (terrain-adjusted `wn_*`, else HRRR `u10/v10`) as a quiver field |
| `direct_comparison.png` | final real vs predicted footprint cell-by-cell: yellow = true positive, blue = false positive, red = false negative, gray = the fire already burning at the forecast start |

```
<out>/
  metrics.json            # config, per-fire metrics, summary, per_fire_extra, figures
  per_fire.csv            # one row per scored fire
  duration_metrics.png    # from plot_duration_metrics.py (below)
  fires/<name>_<fire_id>/
    final_arrival_comparison.png
    wind_coherence.png
    direct_comparison.png
```

## Aggregate analysis: metric vs fire length

`scripts/plot_duration_metrics.py` turns an evaluation directory into one figure
and a printed table: each metric's **median over quantile bins of fire length**,
with the inter-quartile range as the bar. It deliberately draws no per-fire
scatter — 100+ points overplot into noise, and the bins carry both the trend and
the spread.

```bash
python scripts/plot_duration_metrics.py --run-dir <run dir>       # uses run/real_eval
python scripts/plot_duration_metrics.py --eval-dir <run>/real_eval --bins 8
```

Default series are `avg_step_iou`, `precision_final`, `recall_final`; `--metrics`
takes any columns `per_fire.csv` has (so re-run `eval_real.py --metrics all`
first if you want others). `final_iou` is available but not plotted by default:
once the predicted union engulfs a fire it degenerates into a ratio of true to
predicted area, which the precision/recall pair explains directly. On the
109-fire v5 validation set the shape is: step IoU declining gently, precision
peaking at mid durations and collapsing on the longest fires (~0.13 beyond 100
steps) while recall climbs to 1.00 — the model progressively filling the window.
