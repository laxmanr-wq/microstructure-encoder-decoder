# Evaluation Methodology

**How the numbers in [results](results.md) are produced, and why the method is
trustworthy.**

## Principles

1. **Native-resolution evaluation.** Reconstruction quality is measured at full
   1024×1024, never on training patches, patch-space PSNR flatters a model that
   patch-trains.
2. **EMA weights.** Every reported number loads the EMA shadow (`load_net(use_ema=True)`),
   not the raw optimizer weights.
3. **Stratified sampling.** Test images are sampled per scale band so coarse- and
   fine-scale micrographs contribute proportionally rather than being drowned out.
4. **Two independent measurement stacks.** Image fidelity is torch (PSNR, MS-SSIM,
   LPIPS); microstructure descriptors are MCRpy/TensorFlow plus a NumPy/FFT
   re-implementation running as a cross-check. The two stacks never share a process
   (TensorFlow + torch cannot coexist in 8 GB).
5. **Chained, not re-estimated.** Each phase reads the previous phase's measured JSON.

## Image-fidelity metrics

| Metric | Implementation | Notes |
|---|---|---|
| PSNR | `losses.psnr`, data_range 2.0 on [−1,1] tensors | per-image, averaged |
| MS-SSIM | `pytorch_msssim`, window adapts to size (11 px ≥161 px, 7 px on 128 px) | `(win−1)·16 < size` constraint |
| LPIPS | `lpips` package, **AlexNet** backbone, grayscale replicated to 3ch | frozen weights, eval mode |
| Compression ratio | input 8-bit bytes ÷ latent fp16 bytes | 8:1 (4:1 at fp32) |

## Tiled reconstruction

Large images are reconstructed in 512-px tiles with 64-px overlap, blended by a
separable linear-ramp window:

```
ramp = linspace(0,1,64) at both tile edges; win = ramp_y ⊗ ramp_x
out += tile_recon · win ;  wsum += win ;  output = out / wsum
```

Tile origins walk the image with stride `tile − overlap`, with a final row/column
snapped to the image edge so nothing is left uncovered. Whole-image mode is used only
when it fits memory, on this machine that means the caller must choose tiling
explicitly (see the [memory trap](operations.md#the-memory-trap)).

## Microstructure descriptors

Phase 6 stage 2 measures whether *microstructure*, not just pixels, survives
compression. Images are binarized (Otsu) and three standard descriptors are computed
on original vs reconstruction, 16 pairs per band, 256-px crops:

| Descriptor | What it captures | Result (overall) |
|---|---|---|
| VolumeFractions (VF) | phase proportion | best preserved (3.4–12.9% err by band) |
| Two-Point Correlations (S2) | spatial correlation structure | main failure mode (4.8–20.2%) |
| LinealPath (LP) | chord-length / connectivity structure | main failure mode (5.1–19.6%) |

Discrepancy is aggregated per band and overall; the success criterion is **<5% overall
on held-out test**. Current: **9.00%: not met**. The NumPy cross-check (FFT-based S2,
ray-sampled lineal path) is stored alongside in `phase6_desc_sample.npz`.

## Gate discipline

Gates are pre-declared pass/fail criteria attached to each phase, written into its
report JSON, and never re-interpreted afterwards:

- Phase 3's overfit gate (≥35 dB at 500 steps) **failed as specified** (31.49 dB) and
  was recorded as FAIL even though the wiring was demonstrably sound. The decision to
  proceed is documented with the measured evidence (35 dB at step 825).
- Phase 6's success-criterion gate **failed** (9.00% vs <5%) and remains the open work
  item driving the [roadmap](roadmap.md).

This is the difference between an engineering record and a highlight reel: the FAILs
are part of the documentation because they determine what happens next.

## Reproducing the evaluation

```bash
python src/eval.py --ckpt checkpoints/full_best.pt \
    --tile 512 --overlap 64 --limit 300 \
    --out reports/eval.json --grid reports/eval_grid.png \
    --desc-npz reports/desc_sample.npz
python scripts/descriptors.py --npz reports/desc_sample.npz --eval reports/eval.json
```

`--limit 0` (default) evaluates the full 1,402-image kept test split; 300 is the
documented stratified sample. Expect ~54 min for 300 images on the M1 (0.093 img/s)
and ~35 min more for the descriptor stage in a separate process.
