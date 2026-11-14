# Performance

Throughput measurements for the real-data baseline and the optional
decompressed sample mirror.

## Throughput

Measured on this machine: 120 training fires, batch 8, 8 workers, chain length
12, one warm epoch (the first epoch of a process also pays cuDNN autotuning):

| config | s/epoch | GPU utilization (mean) |
|---|---|---|
| fp32, `cudnn.benchmark` off | 33.2 | 34 % |
| fp32, `cudnn.benchmark` on | 22.0 | 52 % |
| bf16 autocast, `benchmark` off | 153.9 | 6 % |
| bf16 autocast, `benchmark` on | 21.0 | 45 % |

Three things that table settles:

- **`cudnn.benchmark` must stay on** (it is, in `RealChainTrainer` for CUDA):
  worth 1.5x in fp32, and without it bf16 is *7x slower* than fp32 — cuDNN's
  default algorithm choice for these depthwise shapes is very poor. The price is
  a one-off autotuning pass per process, visible as a slow first epoch.
- **bf16 (`amp: true`) buys ~5 % at the epoch level**, not the 1.35x the
  isolated step loop shows, because the epoch is input-bound: the GPU is only
  running at ~45-50 % duty even with everything else in place.
- Cold disk reads at 2.3 GB/s here, so the disk is not the constraint. The
  compressed member inflate is.

Why the input side dominates — one step of a batch of 8 chains, everything
resident in GPU memory, no dataloader:

| batch | ms/step | sample-steps/s | GPU utilization |
|---|---|---|---|
| 2 | 10.5 | 190 | 49 % |
| 8 | 18.2 | 440 | 92 % |
| 16 | 33.3 | 480 | 99 % |
| 32 | 64.2 | 499 | 99 % |
| 8 + bf16 | 13.4 | 600 | 93 % |
| 16 + bf16 | 24.9 | 643 | 100 % |

So the step loop itself is healthy at batch 8 and up: the real pipeline's 0-60 %
sawtooth is starvation. The cause is the format — each fire's samples are one
**DEFLATE-compressed `npy` member** inside its npz, so *any* access to a fire
costs a full inflate of that fire, and indexed chains re-inflate it once per
chain that lands in another worker or after eviction. **Training streams fire by
fire** (`finetune.dataset.StreamingChainDataset`): each fire goes to exactly one
worker, is decompressed once, and all of its chains for that epoch are cut from
RAM.

| loader (120 fires) | s/epoch | GPU mean |
|---|---|---|
| per-chain indexing (previous) | 55.2 | 17 % |
| fire-major streaming | 30.5 | 43 % |

and with the same warm-epoch split into its two passes (fp32, nothing else on
the machine), streaming in both rows:

| storage | train pass | val pass (24 chains) | epoch |
|---|---|---|---|
| npz (inflate per fire) | 30.3 s | 9.4 s | 39.7 s |
| decompressed mirror | 24.6 s | 3.2 s | 27.8 s |

(The three storage numbers were taken in separate clean runs; earlier readings
of 108 s and 300 s for the same work came from a concurrent job on the same box
and from cuDNN autotuning being counted inside the first epoch, not from the
loader.)

Decompressions drop from one per chain (369 for those fires) to one per fire
(120) — verified directly: 4 fires / 18 chains → 4 decompressions, 14 cache
hits. Coverage is unchanged (the random back-to-back cut still moves every
epoch, `fire_chain_cuts`); the trade-off is that a batch can hold several chains
of one fire instead of a global shuffle.

Measured costs per 12-sample chain of real fires: **153 ms** through the npz
(`np.load`, i.e. pure inflate at ~270 MB/s per core) versus **26 ms** from a
memory-mapped raw array — which is what the optional
[preprocessing step](#preprocessing-optional-decompressed-sample-mirror) below
buys.

Two costs that are *not* per-epoch but will surprise you first:

- **cuDNN autotuning** (`cudnn.benchmark`, ~270 s for the bf16 shapes here) is
  paid inside the first epoch of each process: one measured warmup epoch took
  291 s while the steady-state epoch in the same harness took 22 s. Short runs
  and benchmarks should therefore warm up once and time a *later* epoch.
- **`build_val_chains`** builds one validation chain per fire, and each one needs
  that fire's array — 109 fires / 6,139 samples took **76.1 s** through the npz
  (0.32 GB/s) versus **5.6 s** through the mirror (4.35 GB/s). That is a
  once-per-launch serial cost in the parent process, before the first epoch.

## Preprocessing (optional): decompressed sample mirror

The single biggest host-side cost is that every `.npz` member is DEFLATE-
compressed, so touching a fire costs a full inflate. Measured on real fires,
building a 12-sample chain costs **153 ms** from the npz (`np.load`) versus
**26 ms** from a memory-mapped raw array — 5.9x, and the npz figure is pure
inflate (~270 MB/s of raw data per core).

`scripts/prepare_real_cache.py` decompresses the product **once** into plain
`.npy` sidecars that training memory-maps instead:

```bash
python scripts/prepare_real_cache.py \
  --real-data /mnt/wildfire/real-data/v5_wildfires_2026-10-08 \
  --out /mnt/wildfire/real-data/v5_wildfires_2026-10-08-uncompressed \
  --splits train,val,test --workers 8
# then:
python scripts/train_real.py --output-dir <run> --data-cache <out>
```

- **All 55 channels are kept**, so the mirror is a drop-in for the npz: the
  channel subset stays a training-time choice, no indexing changes, and a test
  asserts the chains built from it are bit-identical to the npz path
  (`test_decompressed_cache_is_a_drop_in_for_the_npz`). `manifest.json` pins the
  source root, the dtype and a **per-fire fingerprint (size + mtime) of the
  source npz**, so a cache built from another dataset, a mixed-dtype cache, or a
  mirror left behind by a *rebuilt* product all fail loudly at the first read
  instead of silently training on the wrong samples.
- **Effect, measured:** a 12-sample chain costs **153 ms** through the npz and
  **26 ms** through the mirror (5.9x), and the per-launch validation chain list
  (`build_val_chains`, 109 fires / 6,139 samples) costs **76.1 s** versus
  **5.6 s** (13.6x, 0.32 -> 4.35 GB/s of raw data). What is left per sample is
  chain assembly (~2 ms) rather than inflate (~13 ms), so the GPU stops waiting
  on the host.
- **Cost:** one pass, measured **120 fires / 19 GB in 8 s** with 8 workers
  (~2.4 GB/s aggregate), so the whole v5 (38,147 samples, 152 GB raw) is a
  couple of minutes. Resumable (`--overwrite` to redo, and converting another
  split keeps the other splits' fingerprints), `--limit-fires N` for a subset.
  Disk: ~149 GB for all three splits in fp32 (~102 GB train-only), or ~75 GB
  with `--dtype float16` (~1 m quantization on elevation-scale channels).
- **Wiring:** the path goes into the run's `config.resolved.yaml`, so
  `eval_real.py` reads the same mirror without being told again.

It is optional because the streaming loader already removes the *redundant*
inflates (above); the mirror removes the remaining ones.
