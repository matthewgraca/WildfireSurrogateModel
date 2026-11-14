# Real-data baseline (`train_real.py` / `eval_real.py`)

The **real-data baseline** trains the same MK-UNet from scratch on the v5
wildfire chains (`/mnt/wildfire/real-data/v5_wildfires_*`) — no MTT trials, no
pretrained weights. It is the reference point the MTT baseline and any
fine-tune are compared against.

Everything that reads the real product lives in the **`finetune`** package
(`src/finetune/`), so `wildfire_simulator/` stays the MTT/synthetic package. The
dependency is one-way: `finetune` imports the shared MK-UNet, `shift_window`,
`iou`, `ScheduledSampler` and `ScalarRNG` from `wildfire_simulator`; nothing in
`wildfire_simulator` imports `finetune`.

| module | role |
|---|---|
| `finetune.dataset` | chains, channel selection, normalization, torch dataset, collation |
| `finetune.trainer` | scheduled-sampling training on chains |
| `finetune.infer` | full-life autoregressive rollout + the model builder |
| `finetune.metrics` | modular per-fire metric registry |
| `finetune.viz` | the three figures |
| `finetune.transforms` | per-channel z-score normalization |

Two scripts do the whole flow:

```bash
# 1. train (scheduled sampling, epoch-varying chains)
python scripts/train_real.py --config configs/real_baseline.yaml \
  --output-dir /mnt/wildfire/surrogate-model/real-baseline

# 2. metrics + visualizations on the validation split
python scripts/eval_real.py --run-dir /mnt/wildfire/surrogate-model/real-baseline \
  --real-data /mnt/wildfire/real-data/v5_wildfires_2026-10-08
```

Related guides: [throughput and the decompressed mirror](performance.md),
[metrics and figures](evaluation.md),
[resuming and TensorBoard](monitoring.md).

## Data contract

One 12-hour sample per step, 135×135 windows (see the dataset's `README.md` and
`ROLLOUT.md`). `finetune.dataset` reads the product directly and
builds **chains**: a contiguous run of samples laid out in its *start* sample's
window, with every later sample's inputs and labels shifted into that window
(`row0`/`col0` make the shift exact) and cells no window covers left as NaN
inputs and unscored. Per step it stores the state *before* the step, that step's
own meteorology/terrain/time, the cumulative burned truth, and the scorable
cells; the satellite-observation channels (`frp`, `n_detections`,
`new_fire_pixels`) are zeroed from step 1 on, because a forecast does not have
them.

| item | value |
|---|---|
| channels (default) | every channel in `channels.json` except `fbfm_mode` = 54. `fbfm_mode` is the one channel the product ships without normalization statistics, and it is a *nominal* FBFM40 code (32 codes between 91 and 189 in a 6-fire sample), so z-scoring it would invent distances between fuel classes. The seven `fbfm_*` composition channels carry its *group* — the code's group matches `argmax` of those fractions in 96.7 % of 12.1 M cells — but 32 codes collapse into 7 groups, so the within-group detail is not in the default input; include `fbfm_mode` explicitly (with statistics, or one-hot encoded) if you want it |
| model input | those 54 channels + one step-time channel (`hours / 24`) = **55** |
| normalization | `norm_cache/04c547cd280a3e4d.json` z-scores, `log1p` for the channels flagged `log` |
| static channels | read from the chain's start window (one landscape per fire), so terrain is never mean-filled |
| chain seed | the fire's first sample whose perimeter exceeds the 20-cell pre-fire FEDS placeholder |
| architecture | the MK-UNet body/widths of the MTT surrogate (`channels=[16,32,64,96,160]`), input stem widened to 55 channels, one sigmoid mask output (0.33 M params, same as the MTT checkpoint) |

Training follows `ROLLOUT.md`: each epoch every fire is cut into back-to-back
12-step (6-day) chains from a random start, plus a 50 % chance each of a chain
from the fire's first real sample and from a growing step; a step is given the
model's own forecast with probability `1/(1+exp(-k*(epoch-t0)))` (teacher-forced
early, half and half at `t0`, self-driven late); no gradient flows back through
the mask handed to the next step; burned land never unburned. Validation during
training is one deterministic 12-step chain per val fire, reported as `val_loss`
(checkpoint selection) and `val_iou`.

Useful flags: `--split-key val|train|test|all` (default `val`), `--max-fires N`,
`--metrics-only`, `--vis-only`, `--limit-fires N` (training smoke test),
`--chain-len`, `--batch-size`, `--num-workers`, `--epochs`, `--amp/--no-amp`,
`--data-cache DIR`, `--device`.
