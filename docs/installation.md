# Installation

## Environment

```bash
conda env create -f environment.yml
conda activate wildfire_env
pip install -e .
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 # or thru here: https://pytorch.org/get-started/locally/
```

The package requires Python ≥ 3.14 (`pyproject.toml`).

## Imports

`wildfire_simulator` and `finetune` must be importable. The scripts add `src/`
to `sys.path` themselves, and `pytest` needs `PYTHONPATH=src` (or a
`pip install -e .`) when the environment's editable install points elsewhere.
