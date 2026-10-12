import torch
import os

from wildfire_simulator import viz

class ModelCheckpoint:
    def __init__(self, monitor, mode, filepath):
        self.monitor = monitor
        self.mode = mode
        self.filepath_template = filepath
        self.best_metric = None

        if self.mode == 'min':
            self.compare = lambda current, best: current < best
        elif self.mode == 'max':
            self.compare = lambda current, best: current > best
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def on_validation_end(self, epoch, metrics, model, optimizer):
        current = metrics.get(self.monitor)
        if current is None:
            return
        if self.best_metric is None or self.compare(current, self.best_metric):
            self.best_metric = current
            path = self.filepath_template.format(epoch=epoch, val_loss=current)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(checkpoint, path)

class EarlyStopping:
    """Stop training once the monitored metric stops improving.

    Mirrors ``ModelCheckpoint``'s monitor/mode semantics so "improvement" is
    defined identically to when a checkpoint is saved: the first epoch always
    counts as an improvement, and each later epoch that does not beat the best
    value seen so far advances a consecutive-stall counter. Once that counter
    reaches ``patience``, the callback signals that training should stop.
    """

    def __init__(self, monitor, mode, patience):
        self.monitor = monitor
        self.mode = mode
        self.patience = patience
        self.best_metric = None
        self.epochs_since_improvement = 0

        if self.mode == 'min':
            self.compare = lambda current, best: current < best
        elif self.mode == 'max':
            self.compare = lambda current, best: current > best
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def on_validation_end(self, epoch, metrics, model, optimizer):
        """Return True when ``patience`` stalled epochs have elapsed."""
        current = metrics.get(self.monitor)
        if current is None:
            return False
        if self.best_metric is None or self.compare(current, self.best_metric):
            self.best_metric = current
            self.epochs_since_improvement = 0
        else:
            self.epochs_since_improvement += 1
        return self.epochs_since_improvement >= self.patience

class TensorBoardCallback:
    def __init__(self, train_writer, val_writer):
        self.train_writer = train_writer
        self.val_writer = val_writer

    def on_validation_end(self, epoch, metrics, model, optimizer):
        self.train_writer.add_scalar("Loss", metrics['train_loss'], epoch)
        self.val_writer.add_scalar("Loss", metrics['val_loss'], epoch)

        # Per-scene validation losses (keys like 'val_loss/<scene>'), so the
        # new-regime (terrain-aware) scenes are tracked separately.
        for key, value in metrics.items():
            if key.startswith('val_loss/'):
                scene = key.split('/', 1)[1]
                self.val_writer.add_scalar(f"Loss/scene/{scene}", value, epoch)

        # Per-scene final-mask IoU over the full validation sets (keys like
        # 'val_iou/<scene>').
        if 'val_iou' in metrics:
            self.val_writer.add_scalar("IOU", metrics['val_iou'], epoch)
            for key, value in metrics.items():
                if key.startswith('val_iou/'):
                    scene = key.split('/', 1)[1]
                    self.val_writer.add_scalar(f"IOU/scene/{scene}", value, epoch)

        # Log per-component losses if available
        for key in ['bce', 'dice', 'focal', 'mask_loss', 'arrival_loss', 'ce', 'penalty']:
            train_key = f'train_{key}'
            val_key = f'val_{key}'
            if train_key in metrics:
                self.train_writer.add_scalar(f"Loss/{key}", metrics[train_key], epoch)
            if val_key in metrics:
                self.val_writer.add_scalar(f"Loss/{key}", metrics[val_key], epoch)

        self._log_viz_images(epoch, metrics)

    def _add_viz_image(self, tag, arr, step):
        """Log a uint8 (H, W, 3) render to the val writer as an image."""
        self.val_writer.add_image(tag, arr, step, dataformats="HWC")

    def _log_viz_images(self, epoch, metrics):
        """Render and log the recorded validation samples as images.

        Per scene: one FAT montage snapshot (predicted vs ground-truth final
        arrival-time maps, one column per sample). Low DPI: a TB dashboard
        snapshot, not an archival figure.
        """
        viz_data = metrics.get('viz')
        if not viz_data:
            return
        for scene, payloads in viz_data.items():
            self._add_viz_image(
                f"viz/{scene}/fat_montage",
                viz.render_fat_montage(
                    [p['pred_history'][-1][1] for p in payloads],
                    [p['gt_history'][-1][1] for p in payloads],
                    [p['idx'] for p in payloads],
                    title=f"{scene} — epoch {epoch}",
                ),
                epoch,
            )
