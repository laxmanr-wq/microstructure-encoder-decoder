# Architecture

**Convolutional autoencoder · 3,772,841 parameters · KL latent (4 ch) · f = 8**

The design follows the LDM/autoencoder family (Rombach et al., *High-Resolution Image
Synthesis with Latent Diffusion Models*) scaled deliberately down for a fanless M1 with
8 GB of unified memory. Patch-train small, infer at full 1024×1024, the network is
fully convolutional and makes no size assumptions.

## Overview

```
input 1×1024×1024
   │  conv 3×3 → 32 ch
   ▼
[ResBlock ×2 → downsample] ×3 stages        1024 → 512 → 256 → 128
   channels 32 → 64 → 128
   ▼
[mid block: ResBlock → attention → ResBlock]  at 128×128 (lowest resolution)
   ▼
conv 3×3 → 8 ch   (mean ‖ logvar, double_z)
   ▼
DiagonalGaussian → z ∈ ℝ⁴ˣ¹²⁸ˣ¹²⁸           latent, 8:1 at fp16
   ▼
[conv 3×3 → 128 ch → mid block]
[upsample (nearest + conv) → ResBlock ×3] ×3 stages
   channels 128 → 64 → 32
   ▼
conv 3×3 → 1 ch, unbounded → clamp at inference
```

## Components

### Residual blocks
`GroupNorm(8) + SiLU + 3×3 conv`, two convs per block, 1×1 skip conv when the channel
count changes. The final conv of each block is **zero-initialized** (the LDM/ADM
residual init), so every block starts as identity and training begins stably.

### Self-attention
A single attention block (`Conv 1×1 QKV` → `F.scaled_dot_product_attention` → 1×1
projection) sits in the mid block only, at the lowest resolution. The brief mandated
dropping it if it cost >15% of step time on MPS; the Phase 4 measurement was **3%**, so
it stayed.

### Latent
Two interchangeable heads behind one config flag (`model.latent`):

| | KL (default) | VQ |
|---|---|---|
| Channels | 4 (encoder outputs 8 = mean ‖ logvar) | 64 |
| Regularizer | `kl_weight 1e-6` · closed-form KL | codebook 512×64, commitment β = 0.25 |
| Sampling | mean at inference (`sample=False`) | nearest-code + straight-through |
| Notes | `logvar` clamped to [−30, 20]; bias init 0.0 | uniform init in ±1/512 |

The `logvar` bias initializes to **0.0**, not −6: Phase 3 measured faster convergence
with early latent noise (23.6 vs 21.9 dB at 200 overfit steps), and the noise collapses
on its own by ~step 250.

### Decoder
Upsampling is **nearest-neighbour interpolation followed by a 3×3 conv**: never
transposed convolution, which produces checkerboard artifacts. Three decoder ResBlocks
per stage (one more than the encoder). The output activation is **`none`**: `tanh`
saturates on the 14.6% of micrograph pixels beyond |x| > 0.9 and kills their gradient,
so the model outputs unbounded values that are clamped to [−1, 1] at inference only.

## Loss function

```
L = L1  +  1.0 · LPIPS(AlexNet)  +  0.25 · (1 − MS-SSIM)
```

- **MSE-only is forbidden**: it is exactly what made the previous attempt blurry.
- **LPIPS uses AlexNet**, not VGG: VGG is too heavy for the 8 GB budget. The grayscale
  input is replicated to 3 channels.
- **MS-SSIM** (pytorch_msssim) uses an 11-px window at ≥161 px and a 7-px window on the
  128-px training patches, because pytorch_msssim always performs 4 downsamplings and
  `(win−1)·16` must stay below the input size.
- The KL term enters the total loss with weight `1e-6`, a light nudge toward a smooth
  latent, not a strong prior.

## Inference strategy

- **Whole-image mode** for images up to ~1024², encoder `sample=False` (latent mean).
- **Tiled mode** (`--tile 512 --overlap 64`) for larger inputs or constrained memory:
  512-px tiles with 64-px overlap, blended with a linear-ramp window on both axes.
  This is the app's default; whole-image mode peaks 5.07 GB on this machine, over the
  4 GB budget (macOS swaps silently instead of raising, so the tiled fallback cannot
  self-trigger on MPS; see [operations](operations.md#the-memory-trap)).
- Output clamped to [−1, 1], then mapped back to 8-bit grayscale.

## Design decisions and why

| Decision | Rationale (measured) |
|---|---|
| f=8, base 32 ch | ~3.8 M params fit the 4 GB peak budget with batch 8 and fp32 |
| KL 4-ch over VQ | default; VQ kept as a config flag for future compression work |
| `out_act: none` | Phase 3 ablation, tanh saturation on 14.6% of pixels slows convergence |
| `logvar_init: 0.0` | Phase 3 ablation, 23.6 vs 21.9 dB at 200 steps vs −6.0 |
| Attention only at low res | 3% step-time cost measured on MPS; full-res attention unaffordable |
| Nearest+conv upsample | transposed conv banned (checkerboard artifacts) |
| LPIPS AlexNet | VGG backbone exceeds the memory budget |
| fp32, never bf16/fp16 training | MPS autocast is unreliable on this stack |

Source: `src/model.py`, `src/losses.py`, `configs/base.yaml`.
