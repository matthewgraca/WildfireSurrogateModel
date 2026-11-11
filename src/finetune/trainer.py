"""Scheduled-sampling trainer for the real-data (v5 chain) surrogate baseline.

The existing ``ForwardBurnTrainer`` rolls a *single* MTT trial over normalized
burn time ``t = dt .. max_t``, taking every step's meteorology from the same
static trial frame. Real data is a variable-length chain: each step has its own
12-hour sample (its own weather), and the fire's length differs per fire. This
trainer therefore rolls a chain and, at every step, reads that step's own
channels from the batch (:mod:`finetune.dataset`), substituting the model's own mask
for channel 0 according to scheduled sampling.

Schedule and gradient scope follow ``ROLLOUT.md`` (the dataset author's
training recipe):

* scheduled sampling: step ``k`` is given the model's *own* forecast for the
  previous step with probability ``sampler.get_prob(epoch)`` and the observed
  burned state otherwise — i.e. teacher forcing early (p ≈ 0), half and half at
  ``t0``, fully self-driven late (p ≈ 1). Same convention and same
  ``ScheduledSampler`` curve as ``trainers.BurnerBatchProcessor``;
* no gradient flows back through the mask handed to the next step — every step
  is trained on its own loss (the state is thresholded and detached);
* burned land does not unburn: the state handed to the next step is the union
  of the state handed in and the step's prediction.

Loss and per-step IoU are computed only over the cells the real data can score
(``scored``); predictions elsewhere still enter the state, so nothing is
silently rewarded outside the scored region. The loss is averaged over
(sample, step) pairs (the natural SGD signal, identical to the MTT trainer's
when chain lengths are equal); the reported IoU is the mean over samples of each
sample's mean-over-its-own-steps, matching the evaluation's per-fire
``avg_step_iou``.

Environment note: the torch build in this environment (2.14.1) miscomputes
broadcast shapes for rank-3 tensors with a leading dim > 1 (e.g.
``(2,1,1) + (2,1,4,5) -> (2,2,4,5)``, ``expand``/``broadcast_to`` raise). Every
per-sample choice here therefore uses rank-4 ``(B,1,1,1)`` conditions — the same
convention as ``trainers.BurnerBatchProcessor`` — and the temporal/state tensors
are kept at the batch's full spatial shape, so no rank-3 broadcast is needed.
"""

from __future__ import annotations

import time

import torch
from tqdm import tqdm

from wildfire_simulator.trainers import _pad_to_multiple
from wildfire_simulator.utils import ScalarRNG

# The model's time channel is ``hours / HOURS_PER_T_UNIT``. 24 h reuses the
# surrogate's time unit (its trials span max_t = 1 over a 24 h burn), so a real
# 12 h step lands at 0.5 and the channel keeps the surrogate's scale.
HOURS_PER_T_UNIT = 24.0


def masked_iou(pred_bool, target_bool, valid) -> torch.Tensor:
    """Per-sample IoU over ``valid`` cells, ``(B,)`` tensor (0 where empty)."""
    valid = valid.bool()
    inter = (pred_bool & target_bool & valid).sum(dim=(-2, -1)).float()
    union = ((pred_bool | target_bool) & valid).sum(dim=(-2, -1)).float()
    return inter / union.clamp(min=1e-8)


class RealChainTrainer:
    """Train/validate an MK-UNet on real-data chains.

    Args:
        model: ``MK_UNet_Regression``-compatible module returning ``[mask]``.
        optimizer: torch optimizer.
        loss_fn: called as ``loss_fn(pred, target)`` on masked tensors.
        train_loader_factory: ``epoch -> DataLoader`` (the chain cut points move
            per epoch, so the loader is rebuilt each epoch).
        val_loader: deterministic validation ``DataLoader`` (or ``None``).
        sampler: object with ``get_prob(epoch)`` (scheduled sampling).
        epochs: total epochs to run.
        device: torch device.
        callbacks: objects with ``on_validation_end(epoch, metrics, model, optimizer)``.
        viz_every: record tracked validation samples every N epochs (0 = never).
        viz_positions: loader positions (into the validation loader's dataset) to
            record. Payloads are put in ``metrics['viz']`` in the same format
            the MTT trainer uses, so ``wildfire_simulator.callbacks.TensorBoard-
            Callback`` and ``finetune.tb`` can log them unchanged.
        fat_norm_hours: arrival hour that maps to 1.0 in the recorded
            arrival-time channel. The default is a full 12-step chain (6 days),
            so tracked chains share one colour scale; a chain truncated by the
            end of a fire simply does not reach 1.0.
        amp: run the forward pass in bf16 autocast (fp32 master weights, and the
            loss in fp32 because BCE is unsafe under autocast). Measured ~1.35x
            faster per step on the real baseline; off by default so a run's
            numbers are plain fp32 unless asked for.
        progress: show tqdm bars.
    """

    def __init__(self, model, optimizer, loss_fn, train_loader_factory,
                 val_loader=None, sampler=None, epochs=1, device=None,
                 callbacks=None, rng=None, progress=True, viz_every=0,
                 viz_positions=None, fat_norm_hours=144.0, amp=False):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.train_loader_factory = train_loader_factory
        self.val_loader = val_loader
        self.sampler = sampler
        self.epochs = epochs
        self.callbacks = callbacks or []
        self.rng = rng if rng is not None else ScalarRNG()
        self.progress = progress
        self.viz_every = int(viz_every or 0)
        self.viz_positions = sorted(viz_positions or [])
        self.fat_norm_hours = float(fat_norm_hours)
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.amp = bool(amp) and self.device.type == 'cuda'
        if self.device.type == 'cuda':
            # Fixed window size and batch layout: let cuDNN pick the fastest
            # algorithm instead of the heuristic default.
            torch.backends.cudnn.benchmark = True
        self.model.to(self.device)
        self.current_epoch = 0

    # -- checkpointing -----------------------------------------------------
    def load_checkpoint(self, checkpoint_path):
        """Restore epoch, weights, optimizer and callback state.

        Callback state (``ModelCheckpoint``'s best metric, ``EarlyStopping``'s
        patience) rides in the checkpoint, so resuming continues the run the
        file came from rather than starting a fresh grace period. Returns a
        summary dict for logging.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.current_epoch = checkpoint['epoch'] + 1
        self.model.load_state_dict(checkpoint['model'])
        if checkpoint.get('optimizer') is not None:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        states = checkpoint.get('callbacks') or {}
        restored = []
        for cb in self.callbacks:
            state = states.get(type(cb).__name__)
            if state is not None and hasattr(cb, 'load_state_dict'):
                cb.load_state_dict(state)
                restored.append(type(cb).__name__)
        return {
            'path': str(checkpoint_path),
            'epoch': checkpoint['epoch'],
            'continues_at': self.current_epoch,
            'callbacks': states,
            'restored': restored,
        }

    # -- rollout -----------------------------------------------------------
    def _run(self, loader, epoch, desc, train, record_positions=None):
        """One pass over ``loader``.

        Returns ``(loss, iou, records)``; ``records`` is ``{loader position:
        payload}`` when ``record_positions`` is given, else ``None``. Each
        payload holds the MTT-style ``pred_history``/``gt_history`` frames
        (channel 0 = mask, channel 1 = arrival hours / ``fat_norm_hours``) plus
        the masks, coverage and times the diagnostic figures need.
        """
        self.model.train(train)
        total, total_iou, total_n = 0.0, 0.0, 0
        position = 0
        want = set(record_positions) if record_positions is not None else None
        records = {} if want is not None else None
        pbar = tqdm(loader, desc=desc, disable=not self.progress)
        for batch_idx, batch in enumerate(pbar):
            x = batch['x'].to(self.device, non_blocking=True)
            y = batch['y'].to(self.device, non_blocking=True)
            scored = batch['scored'].to(self.device, non_blocking=True)
            perim0 = batch['perim0'].to(self.device, non_blocking=True)
            t_hours = batch['t_hours'].to(self.device, non_blocking=True)
            extent_hours = batch['extent_hours'].to(self.device, non_blocking=True)
            valid_step = batch['valid_step'].to(self.device, non_blocking=True)

            B, L, C, H, W = x.shape
            state = (perim0 > 0.5).unsqueeze(1)       # (B, 1, H, W)

            # Tracked samples of this batch, with their arrival-time maps (the
            # seed is already burning at hour 0, exactly as the evaluation
            # script's arrival maps treat it).
            local = None
            if records is not None:
                local = [i for i in range(B) if position + i in want]
                for i in local:
                    seed = perim0[i] > 0.5
                    records[position + i] = {
                        'idx': position + i,
                        'fire_id': batch['fire_id'][i],
                        'start': int(batch['start'][i]),
                        'perim0': seed.detach().cpu(),
                        'scored': [],
                        'hours': [],
                        'pred_history': [],
                        'gt_history': [],
                        'pred_fat': torch.where(seed, torch.zeros_like(seed, dtype=torch.float32),
                                                torch.full_like(seed, float('nan'), dtype=torch.float32)).cpu(),
                        'gt_fat': torch.where(seed, torch.zeros_like(seed, dtype=torch.float32),
                                              torch.full_like(seed, float('nan'), dtype=torch.float32)).cpu(),
                        'pred_assigned': seed.detach().cpu(),
                        'gt_assigned': seed.detach().cpu(),
                    }
                if not local:
                    local = None

            if train:
                # p(model's own forecast); the rest is the observed state
                prob = self.sampler.get_prob(epoch) if self.sampler else 0.0
                # Same RNG convention as the MTT trainer: reseed per batch so
                # the draw is reproducible from (epoch, batch_idx).
                self.rng.seed(epoch * 10_000 + batch_idx)
                # Rank-4 condition (B,1,1,1), matching BurnerBatchProcessor.
                # Do NOT collapse this to (B,1,1): the torch build in this
                # environment (2.14.1) miscomputes broadcast shapes for rank-3
                # conditions whose leading dim is > 1 — e.g.
                # torch.where(cond(2,1,1), a(2,1,H,W), b(2,1,H,W)) returns
                # (2,2,H,W) instead of (2,1,H,W) — which would silently widen
                # the fire-state channel. tests/test_finetune_trainer.py pins this.
                use_pred = torch.tensor(
                    [self.rng.rand().item() < prob for _ in range(B)],
                    dtype=torch.bool, device=self.device,
                ).view(B, 1, 1, 1)

            batch_loss, batch_n = 0.0, 0
            # Per-sample IoU accumulators: the reported IoU is the mean over
            # samples of each sample's mean-over-its-own-steps, which is the
            # convention of the reported per-fire avg_step_iou (chains in a
            # batch have different lengths, so a step-weighted mean would not
            # match the evaluation).
            iou_sum = torch.zeros(B, device=self.device, dtype=torch.float64)
            step_count = torch.zeros(B, device=self.device, dtype=torch.float64)
            for k in range(L):
                sel = valid_step[:, k]
                if not bool(sel.any()):
                    break
                xk = x[:, k]                          # (B, C, H, W)
                gt_state = xk[:, 0:1] > 0.5           # burned before this step
                if train:
                    inp_state = torch.where(use_pred, state, gt_state)
                else:
                    inp_state = state

                inp = xk.clone()
                inp[:, 0:1] = inp_state.to(xk.dtype)
                t_ch = (t_hours[:, k] / HOURS_PER_T_UNIT).view(B, 1, 1, 1).expand(
                    B, 1, H, W)
                inputs = torch.cat([inp, t_ch], dim=1)     # (B, C + 1, H, W)
                target = y[:, k].unsqueeze(1)              # (B, 1, H, W)
                valid = scored[:, k].unsqueeze(1)

                inputs_p, _, _ = _pad_to_multiple(inputs, 32)
                target_p, _, _ = _pad_to_multiple(target, 32)
                valid_p, _, _ = _pad_to_multiple(valid, 32)

                with torch.set_grad_enabled(train):
                    # bf16 autocast for the forward only: the loss stays fp32
                    # (BCE is unsafe under autocast, and the mask output feeds a
                    # threshold), and autocast never touches the master weights.
                    with torch.autocast('cuda', dtype=torch.bfloat16,
                                        enabled=self.amp):
                        pred_out = self.model(inputs_p)
                    if isinstance(pred_out, (list, tuple)):
                        pred_out = pred_out[0]
                    pred = pred_out[:, :, :H, :W]

                    # The loss always sees fp32: autocast may hand back a bf16
                    # prediction, and a loss that mixes dtypes (BCELoss) would
                    # fail, while the mask path needs an exact threshold.
                    valid_f = valid_p[:, :, :H, :W].to(torch.float32)
                    # Zero the unscored cells in *both* sides: the loss then
                    # sees exactly the scored region (Dice's intersection and
                    # cardinalities are all masked).
                    target_m = target_p[:, :, :H, :W] * valid_f
                    pred_m = pred.float() * valid_f
                    loss = self.loss_fn(pred_m[sel], target_m[sel])

                    if train:
                        self.optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        self.optimizer.step()

                with torch.no_grad():
                    new = pred.detach() > 0.5
                    # Union: burned land does not unburn. Uses the raw
                    # prediction (not the loss-masked one) so the state is not
                    # shaped by the real data's window coverage.
                    state = inp_state | new
                    # Score only the steps' selected samples: chains in a batch
                    # have different lengths, and a full-batch state would
                    # broadcast against the selected targets (silently mixing
                    # samples) instead of raising.
                    iou = masked_iou(state[sel].squeeze(1), target[sel, 0] > 0.5,
                                     valid_f[sel, 0] > 0.5)
                    batch_loss += loss.item() * int(sel.sum())
                    batch_n += int(sel.sum())
                    iou_sum[sel] += iou.to(torch.float64)
                    step_count[sel] += 1.0

                    if local:
                        for i in local:
                            if not bool(valid_step[i, k]):
                                continue
                            payload = records[position + i]
                            hours = float(extent_hours[i, k])
                            # Arrival maps: a cell keeps the hour it first
                            # burned (the seed's cells stay at 0).
                            for key, mask in (('pred', state[i, 0]),
                                              ('gt', target[i, 0] > 0.5)):
                                assigned = payload[f'{key}_assigned']
                                newly = (mask.detach().cpu() & ~assigned)
                                fat = payload[f'{key}_fat']
                                fat[newly] = hours
                                payload[f'{key}_assigned'] = assigned | newly.cpu()
                                payload[f'{key}_history'].append(torch.stack([
                                    mask.detach().float().cpu(),
                                    torch.nan_to_num(fat, nan=0.0)
                                    / self.fat_norm_hours,
                                ]))
                            payload['scored'].append(
                                valid_f[i, 0].detach().cpu() > 0.5)
                            payload['hours'].append(hours)

            position += B
            if batch_n:
                scored = step_count > 0
                batch_iou = float((iou_sum[scored]
                                   / step_count[scored]).mean().item())
                total += (batch_loss / batch_n) * B
                total_iou += batch_iou * int(scored.sum())
                total_n += int(scored.sum())
            pbar.set_postfix(loss=f"{total / max(total_n, 1):.4f}",
                             iou=f"{total_iou / max(total_n, 1):.4f}")

        return (total / max(total_n, 1), total_iou / max(total_n, 1), records)

    def fit(self):
        """Run epochs ``current_epoch .. epochs - 1``, calling the callbacks
        after each validation pass.

        Returns the last epoch's metrics, or ``{}`` when there was nothing to
        run (``current_epoch >= epochs``) — the caller that set those bounds
        (a resume) is responsible for saying so loudly.
        """
        metrics = {}
        for epoch in range(self.current_epoch, self.epochs):
            started = time.perf_counter()
            loader = (self.train_loader_factory(epoch)
                      if self.train_loader_factory is not None
                      else None)
            train_loss, train_iou, _ = (
                self._run(loader, epoch, f"Epoch {epoch}", train=True)
                if loader is not None
                else (float('nan'), float('nan'), None))
            do_record = (self.viz_every > 0 and self.viz_positions
                         and epoch % self.viz_every == 0)
            if self.val_loader is not None:
                val_loss, val_iou, records = self._run(
                    self.val_loader, epoch, "Validating", train=False,
                    record_positions=self.viz_positions if do_record else None)
            else:
                val_loss, val_iou, records = float('nan'), float('nan'), None

            metrics = {
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'epoch': epoch,
                # Wall time of the whole epoch (train + validation + tracking).
                # The first epoch of a process also pays cuDNN autotuning when
                # cudnn.benchmark is on, so it is normal for it to be slower.
                'seconds': time.perf_counter() - started,
            }
            if records:
                # Same key/shape the MTT trainer publishes, so the stock
                # TensorBoardCallback works; the extra per-fire keys are what
                # finetune.tb uses for the diagnostic figures.
                metrics['viz'] = {'val': [records[k]
                                          for k in sorted(records)]}
            tqdm.write(f"epoch {epoch}: {metrics['seconds']:.1f}s  "
                       f"train_loss {train_loss:.4f}  val_loss {val_loss:.4f}  "
                       f"val_iou {val_iou:.4f}")
            for cb in self.callbacks:
                if cb.on_validation_end(epoch=epoch, metrics=metrics,
                                        model=self.model,
                                        optimizer=self.optimizer):
                    self.current_epoch = epoch + 1
                    return metrics
            self.current_epoch = epoch + 1
        return metrics

    def evaluate(self):
        if self.val_loader is None:
            return {}
        val_loss, val_iou, _ = self._run(self.val_loader, 0, "Evaluating",
                                         train=False)
        return {'val_loss': val_loss, 'val_iou': val_iou}
