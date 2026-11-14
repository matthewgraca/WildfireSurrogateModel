# Monitoring and resuming

Both apply to the real-data baseline (`scripts/train_real.py`); setup is in
[the real-data baseline guide](real-baseline.md).

## Resuming a run

```bash
# continue the same run: --resume with no value picks the lowest-val_loss
# checkpoint in <output-dir>/checkpoints (files are written only on
# improvement, so the best one is the most recent)
python scripts/train_real.py --output-dir <same run dir> --resume \
    --epochs 200 --early-stopping --patience 20
```

`--epochs` is the **total** target, not the number of extra epochs: resuming a
120-epoch run with `--epochs 200` trains epochs 120–199. Asking for an
`--epochs` that is not beyond the resumed epoch exits with a message rather than
silently running nothing.

What a resume restores, all of it from the checkpoint file itself:

| restored | why it matters |
|---|---|
| model weights + **optimizer state** | AdamW's moments continue; no re-warmup |
| **epoch counter** | keeps the scheduled-sampling curve where it was (self-driven by epoch 120, and a restart would drop it back to ~0.02 teacher-forcing) and keeps the per-epoch chain-cut seeds moving |
| `ModelCheckpoint.best_metric` | the first resumed epoch is not mistaken for a new best |
| `EarlyStopping.best_metric` + `epochs_since_improvement` | patience counts stalls relative to the best value of the whole run, not a fresh grace period |

The callback state rides in the checkpoint rather than in a side file, so
resuming from an older checkpoint restores *that* file's state; nothing can
drift out of sync. It is saved by `ModelCheckpoint`, which reads the state
`dict` of the callbacks listed in its `state_providers` — the script therefore
lists `EarlyStopping` **before** `ModelCheckpoint`, which is what makes the saved
patience current for the epoch being written.

TensorBoard: keep the same `--output-dir`. The new process appends an event file
to the same run directory and starts at the resumed epoch, so `Loss`, `IOU`,
`Time/epoch_seconds` and the tracked fires extend the existing series; a
different `--output-dir` forks it into a second run.

Two things to keep constant, since the checkpoint stores no config: the
effective channel list, norm stats, `model_channels` and `chain_len`. A width
mismatch fails loudly at load; a changed norm-stats path would not, so compare
against the run's `config.resolved.yaml` (which a resume rewrites — copy it aside
first if you want the phase-1 record).

## TensorBoard

Training writes to `<output-dir>/training/{train,val}`:

```bash
tensorboard --logdir /mnt/wildfire/surrogate-model/real-baseline/training
```

Every epoch: `Loss` (train/val), `IOU` (train/val) and the per-component losses
(`Loss/dice`, …) when the loss function exposes them. Every `viz_every` epochs
(default `5`; `--viz-every 0` disables) a fixed subset of `viz_fires`
validation fires is drawn — the positions are seeded, so the *same* fires are
tracked for the whole run:

| tag | content |
|---|---|
| `viz/val/fat_montage` | predicted (top) vs real (bottom) final arrival-time maps for the tracked fires, one column each, one shared hour scale — the same montage the MTT run logs |
| `viz/fire/<fire_id>/direct` | that fire's final footprint cell by cell: yellow = true positive, blue = false positive, red = false negative, gray = fire at the forecast start |
| `IOU/fire/<fire_id>` | the tracked fire's final-step IoU, so hard fires stand out of the average |

Two things to read them with in mind: the tracked chains are the *training-time*
validation chains (`val_chain_len` steps), not the full-life rollouts of the
metrics section, and the montage's colour scale is one full chain
(`chain_len × 12 h`), so a fire whose window ends before it stops burning never
reaches red.
