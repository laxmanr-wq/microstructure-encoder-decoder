# Evaluation & Results

**Phase 6 · 300 stratified test images · native resolution, tiled 512/64 · ckpt `full_best.pt` step 11,500**

## Method

`src/eval.py` reconstructs test images at **native 1024×1024** (not patch space) with
the EMA weights, using tiled 512-px / 64-overlap inference, and reports PSNR, MS-SSIM
and LPIPS per scale band and overall. 300 images were sampled stratified by band;
already enough for stable per-band statistics; running the remaining 1,287 is not a
priority. Peak memory 3.07 GB, wall clock 53.9 min (0.093 img/s).

A second stage (`scripts/descriptors.py`) measures **quantitative microstructure
fidelity** with MCRpy (VolumeFractions, Two-Point Correlations S2, and LinealPath) on
on 16 original/reconstruction pairs per band (256-px crops), with an independent
NumPy/FFT implementation of the same three descriptors running alongside as a
cross-check. MCRpy runs in a separate process: TensorFlow and torch must not share
this 8 GB machine.

## Image fidelity (measured 17 Sep 2026, `reports/phase6_eval.json`)

| Band | n | PSNR (dB) | MS-SSIM | LPIPS |
|---|---|---|---|---|
| 0 (coarsest) | 42 | 22.60 | 0.8858 | 0.1485 |
| 1 | 63 | 25.47 | 0.9426 | 0.1011 |
| 2 | 54 | 26.87 | 0.9527 | 0.0960 |
| 3 | 71 | 27.59 | 0.9615 | 0.0820 |
| 4 (finest) | 70 | 28.82 | 0.9597 | 0.0845 |
| **overall** | **300** | **26.60** | **0.9449** | **0.0984** |

Compression: **8:1 at fp16** (latent 128×128×4×2 B vs 1024×1024×1 B), 4:1 at fp32.

## Success criterion

Target: **descriptor discrepancy < 5%** vs originals on held-out test images, measured
with the MCRpy microstructure descriptors (VF / S2 / lineal path, see
[evaluation methodology](evaluation.md#microstructure-descriptors)).

| Band | VF | S2 | Lineal path | Discrepancy | < 5%? |
|---|---|---|---|---|---|
| 0 | 12.88% | 20.21% | 19.64% | **17.58%** | NO |
| 1 | 5.02% | 9.64% | 9.38% | **8.01%** | NO |
| 2 | 3.71% | 6.00% | 6.17% | **5.29%** | NO |
| 3 | 4.51% | 11.54% | 12.99% | **9.68%** | NO |
| 4 | 3.43% | 4.80% | 5.12% | **4.45%** | **YES** |
| **overall** | | | | **9.00%** | **NO** |

## Reading the numbers

- **The success criterion is not met: 9.00% vs <5%, a factor of ~1.8.** This is honest
  project state, not a failure to measure: Phase 6 exists precisely to answer this.
- **Training worked.** The same descriptor metric at the calibration checkpoint was
  23.51%; it fell to 9.00% by step 11,500. The trend had not flattened when the time
  budget expired, only 38% of the planned steps were consumed.
- **Volume fraction is consistently the best-preserved descriptor** (3.4–5.0% on bands
  1–4). **S2 and lineal path are the failing ones**: spatial correlation structure is
  lost more than phase proportion. That signature means high-frequency texture is
  missing, the classic motivation for the Phase 8 adversarial fine-tune.
- **Band 0 (coarsest) is the worst on every metric** (17.58% discrepancy, PSNR 22.6 dB).
  Coarse-scale images are visually smooth, so their latent budget is relatively smaller,
  and 128-px training patches see only a fraction of one large grain.

## Prototype verification (user upload)

A user-uploaded fracture micrograph through the app: **PSNR 29.91 dB · MS-SSIM 0.9627 ·
LPIPS 0.0849 · 8:1 compression**, tiled mode. The app is shippable today as a
visual/compression prototype.

## Gate discipline

Every phase writes `reports/phase<N>_<name>.json` and its gate verdict; the next phase
reads those measured numbers and never re-estimates them. Two gates deserve note:

- **Phase 3 was recorded as FAIL** (31.49 dB vs the 35 dB target at 500 overfit steps)
  even though the wiring was verified sound, 35 dB was crossed at step 825 and 39.72 dB
  reached by 1,500. Gates are recorded as measured; the project proceeded on the
  documented evidence, not on a re-interpreted pass.
- **Phase 6's criterion gate is FAIL** and the roadmap exists to close it.

## Comparison to the previous attempt

The prior attempt on this dataset used MSE-only training and produced blurry
reconstructions. The loss composition (L1 + LPIPS + MS-SSIM, no MSE) is the direct
remedy; MSE-only is forbidden in the brief.

Source: `reports/phase6_eval.json`, `reports/phase6_desc.log`, `src/eval.py`,
`scripts/descriptors.py`.
