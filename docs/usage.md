# Usage

## Install

```bash
cd "/Volumes/Hard Disk/Microstructure Encoder Decoder"
python3.13 -m venv .venv && source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch torchvision lpips pytorch-msssim mcrpy pyyaml pandas gradio
```

**Python 3.13 is required**: `mcrpy` needs TensorFlow, which has no cp314 wheel.

Cache the dataset on the internal SSD (3.6× faster reads than the external drive):

```bash
rsync -a --exclude='._*' --exclude='.DS_Store' dataset/ ~/ms_cache/dataset/
```

## Quick start, the app

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
python -u src/app.py        # -> http://127.0.0.1:7860
```

Upload a micrograph → the app encodes it, decodes it, and shows original vs
reconstruction side by side with PSNR, MS-SSIM, LPIPS (AlexNet) and the achieved
compression ratio. Both the reconstruction PNG and the latent (`.npy`, fp16) are
downloadable. Runs on MPS inside the 4 GB budget.

> The app's default inference mode is **tile 512 / overlap 64**. Do not switch to
> whole-image mode on this machine: it peaks 5.07 GB and thrashes (see
> [operations](operations.md#the-memory-trap)).

Default checkpoint: `checkpoints/full_best.pt` (override with `MSAE_CKPT=/path/to.pt`).

## CLI inference

```bash
python src/infer.py path/to/micrograph.jpg
# options: --ckpt PATH  --tile 0|512  --overlap 64  --no-ema  --outdir outputs
```

Writes three artifacts per image into `outputs/`:

| File | Content |
|---|---|
| `<stem>_recon.png` | 8-bit grayscale reconstruction, clamped [−1,1] → [0,255] |
| `<stem>_latent.npy` | latent tensor, fp16, shape `[4, H/8, W/8]` |
| `<stem>_metrics.json` | PSNR, MS-SSIM, LPIPS, compression ratio, timing, mode |

Behaviour details:

- Images are converted to grayscale and centre-cropped to a multiple of 8.
- Encode uses the latent **mean** (`sample=False`), deterministic output.
- Compression ratio is computed against 8-bit grayscale input bytes: **8:1 at fp16**,
  4:1 at fp32.
- `--tile 0` runs whole-image; on allocation failure it falls back to 512/64. On MPS
  the exception never fires (macOS swaps instead), so prefer passing `--tile 512`
  explicitly for anything ≥ ~1024².

## Evaluation

```bash
python src/eval.py --ckpt checkpoints/full_best.pt \
    --tile 512 --overlap 64 --limit 300 \
    --out reports/eval.json --grid reports/eval_grid.png
python scripts/descriptors.py          # MCRpy descriptors, separate process
```

Always pass `--tile 512 --overlap 64` explicitly (see the memory trap in
[operations](operations.md#the-memory-trap)). `src/eval.py` writes its JSON only at
the end, interrupting costs time, never data.

## Rebuilding the data layer

```bash
python prep/build_manifest.py    # re-enumerate, re-cluster, re-split (seed 1337)
python scripts/phase2_check.py   # dataloader speed + databar compliance audit
```

The split is deterministic (seed 1337, whole-cluster assignment). **Never re-split
after Phase 1**: every downstream number is tied to this exact manifest.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `UnidentifiedImageError` on many files | AppleDouble `._*` sidecars, the pipeline filters them; don't bypass the manifest |
| Very slow first run | external-drive reads; build the SSD cache (`rsync` above) |
| Run slows over minutes | thermal throttling on the fanless M1, expected (see [operations](operations.md)) |
| Process swells, it/s collapses | memory thrashing, switch to tiled mode, close apps, check swap (see [operations](operations.md)) |
| `mcrpy` import fails | needs Python 3.13 + TensorFlow; run descriptors in a separate process from torch |
