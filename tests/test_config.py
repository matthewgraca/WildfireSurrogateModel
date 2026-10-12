import os

from wildfire_simulator.config import load_config


def _write(tmp_path, text):
    p = tmp_path / "config.yaml"
    p.write_text(text)
    return str(p)


def test_config_defaults_when_absent(tmp_path):
    path = _write(tmp_path, "dt: \"1/8\"\nmax_t: 1.0\n")
    cfg = load_config(path)

    # Defaults keep existing runs behavior-identical.
    assert cfg['lr'] == 5e-4
    assert cfg['in_channels'] == 14
    assert cfg['epochs'] == 100
    assert cfg['early_stopping'] is True
    assert cfg['finetune']['init_checkpoint'] is None


def test_config_reads_explicit_values(tmp_path):
    path = _write(
        tmp_path,
        "dt: \"1/8\"\nmax_t: 1.0\nlr: 1.0e-4\nin_channels: 15\n"
        "epochs: 5\nearly_stopping: false\n"
        "finetune:\n  init_checkpoint: ckpts/best.pt\n",
    )
    cfg = load_config(path)

    assert cfg['lr'] == 1e-4
    assert cfg['in_channels'] == 15
    assert cfg['epochs'] == 5
    assert cfg['early_stopping'] is False
    # Relative init_checkpoint is resolved against the config file's directory.
    resolved = cfg['finetune']['init_checkpoint']
    assert os.path.isabs(resolved)
    assert resolved.endswith(os.path.join("ckpts", "best.pt"))
    assert str(tmp_path) in resolved
