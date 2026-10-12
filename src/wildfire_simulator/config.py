import yaml
from pathlib import Path

DEFAULT_CONFIG_PATH = Path(__file__).parent.parent.parent / "config.yaml"


def load_config(path=None):
    """Load configuration from YAML file.
    
    Args:
        path: Path to config file. Defaults to project root config.yaml.
    
    Returns:
        dict with parsed config values. Fraction strings (e.g. "1/48") 
        are evaluated to floats.
    """
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Evaluate fraction strings to floats
    if isinstance(config.get('dt'), str):
        config['dt'] = _eval_fraction(config['dt'])

    # Resolve scene paths relative to the config file's directory so training
    # works regardless of the current working directory.
    base_dir = config_path.parent
    scenes = config.get('scenes') or []
    resolved_scenes = []
    for scene in scenes:
        resolved = {
            'landscape': _resolve_path(base_dir, scene.get('landscape')),
            'trials': _resolve_path(base_dir, scene.get('trials')),
            'ignitions': _resolve_path(base_dir, scene.get('ignitions')),
        }
        # Optional per-scene reproducible subsampling knobs (pass through).
        if scene.get('limit') is not None:
            resolved['limit'] = int(scene['limit'])
        if scene.get('seed') is not None:
            resolved['seed'] = int(scene['seed'])
        resolved_scenes.append(resolved)
    config['scenes'] = resolved_scenes

    # Model / optimizer defaults (keep existing runs behavior-identical).
    config.setdefault('lr', 5e-4)
    config.setdefault('in_channels', 14)
    config.setdefault('loss', 'dice')
    # Training-time TensorBoard image viz: log rollout imagery for a fixed
    # per-scene sample subset every ``viz_every`` epochs (0 disables).
    config.setdefault('viz_every', 10)
    config.setdefault('viz_samples_per_scene', 10)

    # Training schedule: total epochs, and whether the EarlyStopping
    # callback (patience 20 on val_loss) is active.
    config.setdefault('epochs', 100)
    config.setdefault('early_stopping', True)

    # Fine-tuning: optional weights-only initialization from a prior checkpoint.
    finetune = config.get('finetune') or {}
    init_ckpt = finetune.get('init_checkpoint')
    finetune['init_checkpoint'] = _resolve_path(base_dir, init_ckpt)
    config['finetune'] = finetune

    return config


def _resolve_path(base_dir, p):
    """Resolve ``p`` against ``base_dir`` if relative; leave absolute paths as-is."""
    if p is None:
        return None
    path = Path(p)
    if not path.is_absolute():
        path = base_dir / path
    return str(path)


def _eval_fraction(s):
    """Safely evaluate a fraction string like '1/48' to a float."""
    parts = s.strip().split('/')
    if len(parts) == 2:
        return float(parts[0]) / float(parts[1])
    return float(s)
