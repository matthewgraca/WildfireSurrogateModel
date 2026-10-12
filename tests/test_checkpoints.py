import torch

from wildfire_simulator.checkpoints import init_from_checkpoint
from wildfire_simulator.models import MK_UNet_Regression


def _make_model():
    # Small channel widths keep the test fast; only weight transfer is exercised.
    return MK_UNet_Regression(
        in_channels=14,
        out_channels=1,
        channels=[8, 16, 16, 16, 16],
        final_activation='sigmoid',
    )


def test_init_from_checkpoint_restores_weights(tmp_path):
    src = _make_model()

    # Save in the same format ModelCheckpoint uses.
    ckpt_path = tmp_path / "ckpt.pt"
    torch.save({"epoch": 3, "model": src.state_dict(), "optimizer": {}}, ckpt_path)

    # A fresh model has different random weights.
    dst = _make_model()
    src_sd, dst_sd = src.state_dict(), dst.state_dict()
    # Sanity: at least one parameter differs before loading.
    assert any(
        not torch.equal(src_sd[k], dst_sd[k]) for k in src_sd
    ), "fresh models unexpectedly identical"

    returned = init_from_checkpoint(dst, str(ckpt_path))

    # Returns the same model instance, now matching the source exactly.
    assert returned is dst
    loaded_sd = dst.state_dict()
    for k in src_sd:
        assert torch.equal(loaded_sd[k], src_sd[k]), f"param {k} not restored"


def test_init_from_checkpoint_accepts_bare_state_dict(tmp_path):
    # Also accept a checkpoint that is a bare state_dict (no 'model' wrapper).
    src = _make_model()
    ckpt_path = tmp_path / "bare.pt"
    torch.save(src.state_dict(), ckpt_path)

    dst = _make_model()
    init_from_checkpoint(dst, str(ckpt_path))

    src_sd, loaded_sd = src.state_dict(), dst.state_dict()
    for k in src_sd:
        assert torch.equal(loaded_sd[k], src_sd[k])
