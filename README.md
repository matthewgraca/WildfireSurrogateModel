# WildfireSimulator

Surrogate models for wildfire spread: an MK-UNet trained on MTT-generated
Farsite trials (`src/wildfire_simulator/`), plus a real-data baseline trained
on real CAL FIRE / NASA FEDS spread chains (`src/finetune/`).

## Quick start

```bash
conda env create -f environment.yml
conda activate wildfire_env
pip install -e .
```

PyTorch usually needs a CUDA-matched reinstall; see
[docs/installation.md](docs/installation.md).

Create a `.env` pointing at the landscape, trials and ignitions
([details](docs/synthetic-training.md#data)), then train the synthetic model
with `python scripts/run.py`.

## Documentation

| Guide | Contents |
|---|---|
| [Installation](docs/installation.md) | environment, PyTorch, editable install and imports |
| [Synthetic training](docs/synthetic-training.md) | MTT data layout, `.env`, the DataLoader pipeline, `scripts/run.py` |
| [Real-fire inference](docs/real-fire-inference.md) | end-to-end rollout on a real CAL FIRE fire, norm stats, initialized forecasts |
| [Real-spread comparison](docs/real-spread-comparison.md) | per-step IoU against NASA FEDS/VIIRS perimeters |
| [Real-data baseline](docs/real-baseline.md) | `finetune` package, training/evaluation flow, data contract |
| [Performance](docs/performance.md) | throughput measurements, decompressed sample mirror |
| [Evaluation](docs/evaluation.md) | metrics, per-fire figures, aggregate analysis |
| [Monitoring and resuming](docs/monitoring.md) | checkpoint/resume semantics, TensorBoard |

## Repository layout

| path | contents |
|---|---|
| `src/wildfire_simulator/` | synthetic (MTT) pipeline: datasets, forward-burn process, model, trainer |
| `src/finetune/` | real-data pipeline: chains, trainer, inference, metrics, figures |
| `scripts/` | entry points for training, inference, evaluation, preprocessing and analysis |
| `configs/` | example inference TOMLs and the real-baseline YAML |
| `tests/` | pytest suite (needs `PYTHONPATH=src`) |
