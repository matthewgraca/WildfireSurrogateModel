"""Checkpoint utilities for initializing models from prior training runs.

Separated from callbacks.py (which owns *saving* via ModelCheckpoint) so that
*loading* weights for fine-tuning has a single, testable home.
"""

import torch


def init_from_checkpoint(model, path, map_location="cpu"):
    """Initialize ``model`` weights from a checkpoint saved by ``ModelCheckpoint``.

    The checkpoint is a dict with a ``"model"`` key holding a ``state_dict``
    (see ``wildfire_simulator.callbacks.ModelCheckpoint``). This performs a
    **strict** load: the checkpoint architecture must match ``model`` exactly.
    Unlike ``ForwardBurnTrainer.load_checkpoint``, it loads *weights only* — it
    does not touch the optimizer state or any epoch counter, which is the
    behavior wanted when starting a fresh fine-tuning run (new LR, epoch 0).

    Extension point (future work): when fine-tuning introduces different or
    additional input channels, the input stem's first-conv weight shape will no
    longer match and a strict load will fail. At that point this function is the
    single place to implement partial transfer — e.g. ``load_state_dict(...,
    strict=False)`` combined with copying the overlapping channel slices of the
    input stem and initializing the new channels (typically zeros so the
    fine-tune begins as a no-op perturbation of the pretrained model). Keeping
    that logic here means callers (run.py) do not change when it grows.

    Args:
        model: an ``nn.Module`` to load weights into (modified in place).
        path: path to a checkpoint file produced by ``ModelCheckpoint``.
        map_location: device mapping passed to ``torch.load`` (default "cpu").

    Returns:
        The same ``model``, with weights loaded.
    """
    checkpoint = torch.load(path, map_location=map_location)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    return model
