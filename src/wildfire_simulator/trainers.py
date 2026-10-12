import torch
import torch.nn.functional as F

from tqdm import tqdm

import numpy as np

def _pad_to_multiple(tensor, multiple=32):
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, h, w
    # pad last dim (width) then second-last (height)
    padded = F.pad(tensor, (0, pad_w, 0, pad_h))
    return padded, h, w


class BurnerBatchProcessor:
    def __init__(
        self,
        burner,
        dt,
        eval,
        sampler=None,
        rng=None,
        device=None
    ):
        self.burner = burner
        self.dt = dt
        self.rng = rng
        self.eval = eval
        self.sampler = sampler
        self.device = device or torch.device('cpu')

    def __call__(self, pred, true, epoch, batch_idx, t):
        if not self.eval:
            self.rng.seed(epoch * 10_000 + batch_idx)

        N = true.size(0)
        H, W = true.shape[-2], true.shape[-1]

        # Move true to device once (stays on GPU for all operations)
        true = true.to(self.device)
        pred = pred.to(self.device)

        # --- Vectorized burn at time t (inputs) ---
        not_burnt_t = true[:, 1:2, :, :] > t  # (N, 1, H, W)
        burned_t = true.clone()
        burned_t[:, 0:1][not_burnt_t] = 0.0
        burned_t[:, 1:2][not_burnt_t] = 0.0

        # Quantize FAT to discrete time steps (ceil to next dt boundary)
        # This matches inference where FAT is assigned t+dt at each step
        fat = burned_t[:, 1:2]
        fat_nonzero = fat > 0
        burned_t[:, 1:2][fat_nonzero] = torch.ceil(fat[fat_nonzero] / self.dt) * self.dt

        # --- Vectorized burn at time t+dt (targets) ---
        not_burnt_dt = true[:, 1:2, :, :] > (t + self.dt)  # (N, 1, H, W)
        burned_dt = true.clone()
        burned_dt[:, 0:1][not_burnt_dt] = 0.0
        burned_dt[:, 1:2][not_burnt_dt] = 0.0

        # --- Scheduled sampling: pick pred or burned ground truth per sample ---
        if self.eval:
            # Crop pred back to original spatial dims so the time channel and
            # final padding behave identically to the non-eval path.
            in_frames = pred[:, :, :H, :W]
        else:
            prob = self.sampler.get_prob(epoch)
            use_pred_mask = torch.tensor(
                [self.rng.rand().item() < prob for _ in range(N)],
                dtype=torch.bool, device=self.device
            ).view(N, 1, 1, 1)
            # Crop pred to true's spatial dimensions for torch.where
            pred_cropped = pred[:, :, :H, :W]
            in_frames = torch.where(use_pred_mask, pred_cropped, burned_t)

        # --- Append time channel at original spatial dims (on device) ---
        t_channel = torch.full(
            (N, 1, H, W), t,
            device=self.device, dtype=true.dtype
        )
        inputs = torch.cat([in_frames, t_channel], dim=1)  # (N, 14, H, W)

        # --- Extract target (mask channel only of burned_dt) ---
        targets = burned_dt[:, 0:1]  # (N, 1, H, W)

        # --- Pad spatial dims to multiple of 32 ---
        inputs, _, _ = _pad_to_multiple(inputs, multiple=32)
        targets, _, _ = _pad_to_multiple(targets, multiple=32)

        return inputs, targets


class ForwardBurnTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        train_batch_processor,
        val_batch_processor,
        callbacks=None,
        epochs=1,
        max_t=1,
        device=None,
        val_loaders=None,
        viz_every=0,
        viz_record_indices=None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        # Optional per-scene validation loaders: {scene_name: DataLoader}.
        # When provided, per-scene val_loss is computed and reported alongside
        # the combined val_loss so new-regime performance is visible separately.
        self.val_loaders = val_loaders or {}
        # TensorBoard image viz: every ``viz_every`` epochs (0 disables),
        # record full rollouts for the fixed per-scene sample subsets in
        # ``viz_record_indices`` ({scene: [loader positions]}); the callback
        # renders them as FAT montage snapshots.
        self.viz_every = viz_every or 0
        self.viz_record_indices = viz_record_indices or {}
        self.train_batch_processor = train_batch_processor
        self.val_batch_processor = val_batch_processor
        self.callbacks = callbacks or []
        self.epochs = epochs
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self.current_epoch = 0
        self.max_t = max_t

        # Ensure batch processors use the same device as the trainer
        self.train_batch_processor.device = self.device
        self.val_batch_processor.device = self.device

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.current_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def _train_epoch(self, epoch, total_epochs):
        self.model.train()
        total_loss = 0.0
        n_samples = len(self.train_loader.dataset)

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        samples_seen = 0
        for batch_idx, batch in enumerate(pbar):
            N = batch.size(0)
            dt = self.train_batch_processor.dt

            # Number of steps actually rolled out (len of the arange below),
            # identical to _validate so train/val losses share the same
            # normalization.
            num_steps = len(np.arange(dt, self.max_t, dt))

            # Seed the first step with the GT state at dt (fire burn + full
            # landscape), exactly as _validate does. Scheduled sampling
            # substitutes this state for the model's previous output; an
            # all-zero init would erase the landscape and fire seed for
            # self-driven samples.
            preds_padded = batch.to(self.device).clone()
            not_burnt = preds_padded[:, 1:2, :, :] > dt
            preds_padded[:, 0:1][not_burnt] = 0.0
            preds_padded[:, 1:2][not_burnt] = 0.0

            for t in np.arange(dt, self.max_t, dt):
                inputs, targets = self.train_batch_processor(
                    preds_padded, batch, epoch=epoch, batch_idx=batch_idx, t=t
                )
                # inputs and targets are already on device (batch processor handles it)

                self.optimizer.zero_grad()

                pred_out = self.model(inputs)
                if isinstance(pred_out, (list, tuple)):
                    pred_out = pred_out[0]

                loss = self.loss_fn(pred_out, targets)
                loss.backward()
                self.optimizer.step()

                # Update preds for next time step (stay on device)
                preds_padded = inputs[:, :13, :, :].detach().clone()
                # Threshold predicted mask to binary
                pred_mask = (pred_out[:, 0:1].detach() > 0.5).float()
                preds_padded[:, 0:1, :, :] = pred_mask
                # Deterministic FAT update: newly burned pixels get arrival time = t + dt
                newly_burned = (pred_mask == 1) & (preds_padded[:, 1:2, :, :] == 0)
                preds_padded[:, 1:2, :, :][newly_burned] = t + dt

                total_loss += loss.item() * N / num_steps

            samples_seen += N
            pbar.set_postfix(loss=f"{total_loss / samples_seen:.4f}")

        return total_loss / n_samples

    def _validate(self, epoch, total_epochs, loader=None, desc="Validating",
                  record_indices=None):
        """Run one validation pass (autoregressive rollout over the loader).

        Returns:
            (val_loss, final_mask_iou, viz_records). ``viz_records`` is a
            ``{position: record}`` dict when ``record_indices`` is given,
            else None. Each record holds the predicted / ground-truth
            (mask, FAT) frame history for one validation sample.
        """
        loader = loader if loader is not None else self.val_loader
        self.model.eval()
        total_loss = 0.0
        iou_sum = 0.0
        n_samples = len(loader.dataset)
        record_set = set(record_indices) if record_indices else None

        pbar = tqdm(loader, desc=desc)
        samples_seen = 0
        position = 0
        recorded = {} if record_set is not None else None
        with torch.no_grad():
            for batch_idx, batch in enumerate(pbar):
                preds_padded = None

                dt = self.val_batch_processor.dt
                num_steps = len(np.arange(dt, self.max_t, dt))
                batch_loss = 0.0

                # True state on device, for final-mask IoU and viz
                # recording (the batch processor moves its own copies).
                true_mask = batch[:, 0:1].to(self.device)
                true_fat = batch[:, 1:2].to(self.device)

                N = batch.size(0)
                H, W = batch.shape[-2], batch.shape[-1]

                local_rec = None
                if record_set is not None:
                    local_rec = [i for i in range(N) if position + i in record_set]
                    if local_rec:
                        for i in local_rec:
                            recorded[position + i] = {
                                'pred_history': [],
                                'gt_history': [],
                            }
                    else:
                        local_rec = None

                for t in np.arange(dt, self.max_t, dt):
                    if preds_padded is None:
                        # Initialize with burned state at dt (gives model
                        # the initial fire seed, matching training behavior)
                        pred_input = batch.to(self.device).clone()
                        not_burnt = pred_input[:, 1:2, :, :] > dt
                        pred_input[:, 0:1][not_burnt] = 0.0
                        pred_input[:, 1:2][not_burnt] = 0.0
                    else:
                        pred_input = preds_padded

                    inputs, targets = self.val_batch_processor(
                        pred_input,
                        batch,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        t=t,
                    )
                    # inputs and targets are already on device

                    pred_out = self.model(inputs)
                    if isinstance(pred_out, (list, tuple)):
                        pred_out = pred_out[0]

                    loss = self.loss_fn(pred_out, targets)
                    batch_loss += loss.item()

                    # Update preds for next time step (stay on device)
                    preds_padded = inputs[:, :13, :, :].detach().clone()
                    # Threshold predicted mask to binary
                    pred_mask = (pred_out[:, 0:1].detach() > 0.5).float()
                    preds_padded[:, 0:1, :, :] = pred_mask
                    # Deterministic FAT update: newly burned pixels get arrival time = t + dt
                    newly_burned = (pred_mask == 1) & (preds_padded[:, 1:2, :, :] == 0)
                    preds_padded[:, 1:2, :, :][newly_burned] = t + dt

                    # Record (mask, FAT) state at t + dt for the tracked
                    # samples: predicted state, plus the ground-truth burn
                    # of the original FAT at the same time.
                    if local_rec:
                        tau = t + dt
                        for i in local_rec:
                            rec = recorded[position + i]
                            rec['pred_history'].append(
                                preds_padded[i, :2, :H, :W].detach().cpu()
                            )
                            gt_mask = true_mask[i].clone()
                            gt_fat = true_fat[i].clone()
                            not_burnt = gt_fat > tau
                            gt_mask[not_burnt] = 0.0
                            gt_fat[not_burnt] = 0.0
                            rec['gt_history'].append(
                                torch.cat([gt_mask, gt_fat], dim=0).cpu()
                            )

                # Final-mask IoU over the whole batch: predicted rollout
                # state vs the GT mask at t = max_t (the batch processor's
                # burn zeroes the mask where FAT > t; a pixel burns within
                # the window iff FAT <= max_t).
                gt_final_mask = true_mask.clone()
                gt_final_mask[true_fat > self.max_t] = 0.0
                pred_final = preds_padded[:, 0:1, :H, :W]
                intersection = (pred_final * gt_final_mask).sum()
                union = ((pred_final + gt_final_mask) > 0).sum()
                iou_sum += (intersection / (union + 1e-8)).item() * N

                total_loss += (batch_loss / num_steps) * batch.size(0)
                samples_seen += batch.size(0)
                # Track the loader position so recorded indices beyond the
                # first batch are found (and each recorded sample's history
                # contains only its own frames).
                position += batch.size(0)
                pbar.set_postfix(val_loss=f"{total_loss / samples_seen:.4f}")

        return total_loss / n_samples, iou_sum / n_samples, recorded

    def fit(self):
        total_epochs = self.epochs
        for epoch in range(self.current_epoch, total_epochs):
            train_loss = self._train_epoch(epoch, total_epochs)

            # Capture per-component losses from training (last batch)
            train_components = {}
            for attr in ['last_bce', 'last_dice', 'last_focal', 'last_mask_loss', 'last_arrival_loss', 'last_ce', 'last_penalty']:
                val = getattr(self.loss_fn, attr, None)
                if val is not None:
                    key = attr.replace('last_', '')
                    train_components[f'train_{key}'] = val

            val_loss, val_iou, _ = self._validate(epoch, total_epochs)
            metrics = {
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_iou': val_iou,
            }
            metrics.update(train_components)

            # Per-scene validation loss (guarantees new-regime performance is
            # measured separately, not blended into the combined val_loss).
            for scene_name, scene_loader in self.val_loaders.items():
                if len(scene_loader.dataset) == 0:
                    continue
                do_record = (
                    self.viz_every > 0
                    and epoch % self.viz_every == 0
                    and scene_name in self.viz_record_indices
                )
                scene_val, scene_iou, records = self._validate(
                    epoch, total_epochs, loader=scene_loader,
                    desc=f"Val[{scene_name}]",
                    record_indices=(
                        self.viz_record_indices.get(scene_name) if do_record else None
                    ),
                )
                metrics[f'val_loss/{scene_name}'] = scene_val
                metrics[f'val_iou/{scene_name}'] = scene_iou
                if records:
                    viz = metrics.setdefault('viz', {})
                    viz[scene_name] = [
                        {
                            'idx': position,
                            'pred_history': record['pred_history'],
                            'gt_history': record['gt_history'],
                        }
                        for position, record in sorted(records.items())
                    ]

            # Capture per-component losses from validation (last batch)
            for attr in ['last_bce', 'last_dice', 'last_focal', 'last_mask_loss', 'last_arrival_loss', 'last_ce', 'last_penalty']:
                val = getattr(self.loss_fn, attr, None)
                if val is not None:
                    key = attr.replace('last_', '')
                    metrics[f'val_{key}'] = val

            stop = False
            for cb in self.callbacks:
                if cb.on_validation_end(epoch=epoch, metrics=metrics, model=self.model, optimizer=self.optimizer):
                    stop = True
            self.current_epoch += 1
            if stop:
                break

    def evaluate(self):
        val_loss, val_iou, _ = self._validate(epoch=0, total_epochs=1)
        return {'val_loss': val_loss, 'val_iou': val_iou}
