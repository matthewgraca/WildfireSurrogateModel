"""TensorBoard callback for the real-data baseline.

Wraps ``wildfire_simulator.callbacks.TensorBoardCallback`` (same writers, same
scalars: ``Loss`` train/val, per-component losses, ``IOU`` = validation IoU) and
adds:

* ``IOU/train`` — the training pass's IoU, so the two curves can be compared,
* ``IOU/fire/<fire_id>`` — the tracked validation fire's **final-step** IoU on
  its own chain, which shows *which* fires are hard (the aggregate ``IOU`` is
  averaged over every step and every fire),
* ``viz/val/fat_montage`` — the MTT run's snapshot: predicted vs ground-truth
  final arrival-time maps (``YlOrRd``, unburned blank) for the tracked
  validation samples, one column per fire, on one shared hour scale,
* ``viz/fire/<fire_id>/direct`` — the same footprint as the evaluation script's
  direct comparison (yellow = true positive, blue = false positive, red = false
  negative, gray = fire at the forecast start), without terrain so it stays
  legible at TensorBoard size.

The tracked samples are the trainer's ``viz_positions`` on the *validation
loader* — fixed positions, so the same fires are watched across the whole run.
Their chains are the training-time validation chains (``val_chain_len`` steps),
not the full-life rollouts ``scripts/eval_real.py`` uses; the arrival maps cover
that window (``fat_norm_hours`` = one full chain).
"""

import numpy as np

from wildfire_simulator.callbacks import TensorBoardCallback
from wildfire_simulator.viz import render_fat_montage

from finetune import viz as rv


def chain_iou(payload):
    """IoU of a tracked payload's final predicted mask vs its final truth."""
    from wildfire_simulator.real_spread import iou

    pred = payload['pred_history'][-1][0].numpy() > 0.5
    truth = payload['gt_history'][-1][0].numpy() > 0.5
    scored = (payload['scored'][-1].numpy()
              if payload['scored'] else np.ones(truth.shape, dtype=bool))
    return float(iou(pred, truth, scored))


class RealTensorBoardCallback(TensorBoardCallback):
    """Scalars + tracked-sample figures for the real-data baseline."""

    def __init__(self, train_writer, val_writer, dpi=60):
        super().__init__(train_writer=train_writer, val_writer=val_writer)
        self.dpi = dpi

    def on_validation_end(self, epoch, metrics, model, optimizer):
        super().on_validation_end(epoch, metrics, model, optimizer)
        # The base callback logs train_loss/val_loss and val_iou; training IoU
        # and per-fire IoU are the additions this baseline wants to watch.
        train_iou = metrics.get('train_iou')
        if train_iou is not None and np.isfinite(train_iou):
            self.train_writer.add_scalar('IOU', train_iou, epoch)
        if metrics.get('seconds') is not None:
            self.train_writer.add_scalar('Time/epoch_seconds',
                                         metrics['seconds'], epoch)

        for payload in metrics.get('viz', {}).get('val', []):
            self.val_writer.add_scalar(
                f"IOU/fire/{payload['fire_id']}", chain_iou(payload), epoch)

    def _log_viz_images(self, epoch, metrics):
        payloads = metrics.get('viz', {}).get('val', [])
        if not payloads:
            return

        # Snapshot of final arrival-time maps (the MTT run's montage), one
        # shared colour scale across the whole montage.
        self._add_viz_image(
            'viz/val/fat_montage',
            render_fat_montage(
                [p['pred_history'][-1][1] for p in payloads],
                [p['gt_history'][-1][1] for p in payloads],
                list(range(len(payloads))),
                title=f"real val — epoch {epoch}",
                dpi=self.dpi,
            ),
            epoch,
        )

        # Per fire: the final footprint classified cell by cell.
        for p in payloads:
            hours = p['hours'][-1] if p['hours'] else float('nan')
            image = rv.render_direct_comparison_array(
                p['pred_history'][-1][0].numpy() > 0.5,
                p['gt_history'][-1][0].numpy() > 0.5,
                p['scored'][-1].numpy() if p['scored'] else None,
                p['perim0'].numpy() > 0.5,
                title=f"{p['fire_id']}  start step {p['start']}  "
                      f"{hours:.0f} h  epoch {epoch}\n"
                      f"IoU {chain_iou(p):.3f}",
                dpi=self.dpi,
            )
            self._add_viz_image(f"viz/fire/{p['fire_id']}/direct", image, epoch)
