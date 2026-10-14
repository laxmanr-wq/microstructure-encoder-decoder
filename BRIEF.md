# BRIEF — Microstructure Encoder/Decoder

Read this file at the start of EVERY phase. It contains every fact you need.
Do not explore the folder to "understand" it. Do not re-derive anything stated here.

## MISSION
Upload a microstructure micrograph -> encode to a compact latent -> decode -> return a
faithful reconstruction. Fidelity is the deliverable, NOT generation. Ship a working prototype.

## PATH
/Volumes/Hard Disk/Microstructure Encoder Decoder
Contains only `dataset/`. Everything else you create.

## ENVIRONMENT TRAPS — all four verified, handle them or the build breaks
1. **exFAT external drive.** No symlinks, case-insensitive, no POSIX ownership.
2. **AppleDouble sidecars.** 10,573 `._*` files shadow the real images. `os.listdir` returns
   21,142 entries; PIL raises UnidentifiedImageError on every `._` file.
   ALWAYS filter: skip basenames starting with `._` and skip `.DS_Store`. Real images: **10,569**.
3. **`dataset/val ` has a TRAILING SPACE** in the directory name. `dataset/train/` nests one
   extra level (`7396/`); `test/` and `val ` hold files directly.
4. **The drive is the bottleneck.** Measured 49 img/s (20 ms/img); training needs ~48 img/s,
   so it WILL be I/O-bound. In Phase 0, copy `dataset/` to the internal SSD
   (~2 GB, 14 GB free) and train from there. Keep checkpoints/reports on the external drive.
   The drive may also be unplugged mid-run — checkpoint often, resume must work.

## HARDWARE — hard constraint
MacBook Air M1, **8 GB unified memory** (CPU+GPU shared), 7-core GPU, **fanless**, 14 GB free SSD.
- PyTorch **MPS**, never CUDA. Device order mps -> cpu. Set PYTORCH_ENABLE_MPS_FALLBACK=1.
- **fp32 only** — MPS bf16/fp16 autocast is unreliable and the model is small enough.
- **Peak memory budget 4 GB.** Exceeding it? Cut batch size, never model correctness.
- Fanless: sustained load throttles 30-40%. Assume every run WILL be interrupted.

## DATASET FACTS — established by audit, do NOT re-derive
- 10,569 JPEG, **1024x1024, 8-bit grayscale (PIL mode `L`)**, uniform. 0 corrupt. ~2 GB.
- Delivered split is 70/15/15 BUT IS LEAKED — 245 exact + 284 near-duplicate groups straddle
  splits; 319 held-out images (10.1%) have a train twin. **You must re-split (Phase 1).**
- Dataset is ~6% redundant: ~9,930 visually unique of 10,569.
- Magnification spans **10x .. 500,000x**; 51% of filenames carry no magnification token.
- 2,750 files are TIFF->JPEG re-encodes (~5:1); JPEG artifacts are baked in and unavoidable.
- ~4,356 images carry a bottom SEM databar, ~684 a top banner. Median bar 47 px, max ~127 px.
- Content is heterogeneous: steel microstructure, fracture surfaces, cast iron, coating
  cross-sections with measurement text printed on them.

## FIXED DECISIONS — do not re-debate
- Patch-train small, infer at full 1024x1024. Fully convolutional, no size assumptions.
- **MSE-only loss is FORBIDDEN** — it is exactly what made the previous attempt blurry.
- Decoder uses nearest-neighbour upsample + conv. **No transposed convolution** (checkerboard).
- Never sample patches overlapping a databar region. Never re-split after Phase 1.
- Seed 1337 everywhere. Normalize to [-1, 1].
- Success target: microstructure descriptor discrepancy <5% vs originals on held-out test.

## ARCHITECTURE (sized for this machine)
f=8, 3 downsample stages, base 32 channels (32/64/128), ~3-5M params.
Latent: KL 4-channel default | VQ codebook 512x64 (config flag).
Self-attention only at lowest resolution; drop it if it costs >15% step time on MPS.
Train patch 128x128, batch from Phase 0. Infer at 1024, `--tile 512 --overlap 64` fallback.

## LOSSES
`L1 + 1.0 * LPIPS(net='alex') + 0.25 * (1 - MS_SSIM)`
LPIPS must be **AlexNet** — VGG is too heavy for 8 GB. Replicate grayscale to 3ch for LPIPS.
PatchGAN is Phase 7 only, off by default.

## CHAINING RULE
Every phase writes `reports/phase<N>_<name>.json`. The next phase READS it and uses those
measured numbers. Never estimate what a previous phase measured. Never skip a gate.
