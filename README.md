# Microstructure Encoder / Decoder

Upload a micrograph → encode to a compact latent → decode → faithful reconstruction.
**Fidelity is the deliverable, not generation.**

Built and measured end-to-end on a MacBook Air M1 (8 GB unified memory, 7-core GPU,
fanless), PyTorch **MPS**, fp32 only, peak memory budget 4 GB.

---

## Install

```bash
cd "/Volumes/Hard Disk/Microstructure Encoder Decoder"
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision lpips pytorch-msssim mcrpy pyyaml pandas gradio
```

Python **3.13** is required: `mcrpy` needs TensorFlow, which has no cp314 wheel.

The dataset is cached on the internal SSD for training reads (the external drive is the
bottleneck, measured 53.5 img/s external vs 192.8 img/s cached, a **3.6×** speedup);
checkpoints and reports stay on the external drive.

```bash
rsync -a --exclude='._*' --exclude='.DS_Store' dataset/ ~/ms_cache/dataset/
```

## Run

```bash
# 0  environment + throughput probe
python scripts/probe.py

# 1  rebuild the split (the delivered split is leaked)
python prep/build_manifest.py

# 2  dataloader checks
python scripts/phase2_check.py

# 3  overfit gate
python scripts/phase3_overfit.py

# 4  calibration run
python src/train.py --steps 2000 --tag calib --report reports/phase4_calib.json

# 5  full training run: resumes from the calibration checkpoint, trains to the 5-hour
#    budget in phase4_calib.json, and kills + restarts itself once early to prove resume
python scripts/phase5_run.py

# 6  evaluation at native resolution + microstructure descriptors
python src/eval.py --ckpt checkpoints/full_best.pt
python scripts/descriptors.py

# 7  prototype
python src/infer.py path/to/micrograph.jpg          # CLI
python src/app.py                                    # local Gradio app -> http://127.0.0.1:7860
python scripts/update_readme.py                      # writes the measured tables below

# 8  optional PatchGAN sharpening - kept only if LPIPS AND descriptors both improve
python src/train_gan.py --steps 5000
python src/eval.py --ckpt checkpoints/gan_best.pt --out reports/eval_gan.json \
    --grid reports/eval_grid_gan.png --desc-npz reports/desc_sample_gan.npz
python scripts/descriptors.py --npz reports/desc_sample_gan.npz --eval reports/eval_gan.json
python scripts/phase8_compare.py
```

Training runs at **batch 8** with `num_workers: 0`. On this 8 GB machine batch 16 thrashes
swap (measured 0.008 it/s vs 1.16 it/s at batch 8), and every spawned data-loader worker
re-imports torch (~400 MB) while the model only needs ~10 patches/s. See
`reports/phase4_memory_diag.json`.

## Prototype

`src/infer.py` takes an image and writes a reconstruction PNG, the latent as `.npy`
(fp16) and a per-image metrics JSON.

`src/app.py` is a local Gradio app: upload a micrograph, see original vs reconstruction
side by side with PSNR / MS-SSIM / LPIPS and the achieved compression ratio, and download
both the reconstruction and the latent. Runs on MPS inside the 4 GB budget.

## Architecture

Fully convolutional, so it patch-trains small and infers at native resolution.

- **f = 8**, 3 downsample stages, base 32 channels (32 / 64 / 128), **3.77 M parameters**
- Latent: **KL, 4 channels** (default) or VQ 512×64 (`model.latent: vq`)
- Self-attention only at the lowest resolution (measured cost 3% of step time, well under
  the 15% drop threshold)
- Decoder upsamples **nearest-neighbour + conv**: never transposed convolution
- Decoder output is unbounded and clamped at inference: `tanh` saturates on the 14.6% of
  micrograph pixels beyond |x| > 0.9 and kills their gradient
- Loss: `L1 + 1.0 · LPIPS(AlexNet) + 0.25 · (1 − MS-SSIM)`. MSE-only is forbidden, it is
  what made the previous attempt blurry. MS-SSIM uses an 11-px window at ≥161 px and a
  7-px window on the 128-px training patches.

## Data

The delivered 70/15/15 split is **leaked** (245 exact + 284 near-duplicate groups straddle
splits). `prep/build_manifest.py` rebuilds it:

- enumerates 10,569 real images (AppleDouble `._*` sidecars skipped)
- clusters duplicates by union-find over md5, 256-bit dHash (skipped below a 64×64 std of
  6.0, where featureless images all collide) and base-name families
- detects SEM databars on both edges and stores `crop_top` / `crop_bottom`, files on disk
  are never modified
- assigns a scale band from the radially averaged power spectrum
- splits **whole clusters** 70/15/15 stratified by dominant scale band, seed 1337

Leak gate, verified independently of the cluster ids by re-checking raw md5 and dHash
against the final split: **zero cross-split duplicate groups**.

## Results

See `reports/`: every phase writes its measured numbers there.

## Documentation

Full documentation lives in [`docs/`](docs/):

| Page | Contents |
|---|---|
| [Overview](docs/index.md) | project summary, status table, source map |
| [Architecture](docs/architecture.md) | model design, latent heads, losses, every decision with its measured rationale |
| [Data pipeline](docs/data.md) | dataset, leak audit, re-split, dataloader |
| [Training](docs/training.md) | config, the two runs, resume protocol, monitoring |
| [Evaluation methodology](docs/evaluation.md) | metrics, tiling, descriptors, gate discipline |
| [Results](docs/results.md) | measured fidelity + descriptor tables and interpretation |
| [Usage](docs/usage.md) | install, app, CLI, evaluation commands, troubleshooting |
| [Operations](docs/operations.md) | memory rules, throttling, runbook for long jobs |
| [Roadmap](docs/roadmap.md) | open items and priorities |
| [Reports index](docs/reports.md) | the `reports/` audit trail and gate history |

