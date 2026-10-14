# Roadmap

**Where the project stands, what is open, and the plan to close the gap.**

## Where we are

The prototype is **shippable today as a visual / compression tool**: upload → encode →
decode at 8:1, PSNR ~26.6 dB on held-out test data, working Gradio app, all inside a
4 GB memory budget on a laptop.

It is **not yet validated for quantitative microstructure measurement**: the descriptor
discrepancy is 9.00% against a <5% target. Spatial correlation descriptors (S2, lineal
path) degrade more than phase proportions (volume fraction), which points at missing
high-frequency texture.

## Priorities, in order of expected value

### 1. Train longer
Phase 5 stopped at step 11,796 of a 30,797-step budget, **only 38% consumed**: and the
descriptor trend (23.51% → 9.00%) had not flattened. The cheapest credible win.

```bash
python src/train.py --resume checkpoints/full_last.pt
```

### 2. Phase 8: PatchGAN fine-tune
`src/train_gan.py` adds a hinge-loss PatchGAN discriminator with the
taming-transformers adaptive generator weight, fine-tuning from the EMA weights. The
adversarial term restores high-frequency texture, exactly what the S2/lineal-path
errors indicate is missing.

The keep/discard rule is strict and mechanical (`scripts/phase8_compare.py`): keep only
if **LPIPS and descriptors both improve**. If LPIPS improves but descriptors worsen, the
GAN is hallucinating microstructure and is discarded.

### 3. Descriptor-aware loss term
Add a differentiable penalty on the failing descriptors (e.g. an S2 correlation loss on
training patches). Directly targets the failing metric instead of hoping perceptual loss
carries it.

### 4. Investigate band 0
Coarsest scale is the worst on every metric (17.58% discrepancy, PSNR 22.6 dB). Two
hypotheses to separate: band-0 patches are under-sampled during training, or f=8 is
simply too aggressive a compression at coarse scale. Quick checks: per-band sampling
rates in `src/data.py`, and an f=4 ablation on a band-0-only overfit.

## Known defects (not blocking, but tracked)

| Defect | Impact | Fix direction |
|---|---|---|
| Tiled fallback cannot self-trigger on MPS (exceptions never raise; macOS swaps) | eval degrades into thrashing if tiling isn't passed explicitly | choose tiling by input size, not `except RuntimeError` |
| Whole-image 1024² eval peaks 5.07 GB | over the 4 GB budget on this machine | keep tiled default in app/eval |
| Phase 3 gate recorded FAIL | procedural only, wiring was verified; documented, not re-interpreted | none; gate history is part of the record |

## Explicitly not a priority

- Running the remaining 1,287 test images, 300 already gives stable per-band statistics.
- CUDA / multi-GPU support, out of scope for this machine constraint.
- Re-opening the split, seed 1337 manifest is frozen; all numbers depend on it.

## Longer-horizon ideas (unscheduled)

- VQ latent head (already implemented behind `model.latent: vq`) for a cleaner
  compression story and downstream generative use.
- Descriptor-preserving latent regularization (measure S2 in latent space).
- Band-conditional loss weighting as an alternative to a dedicated band-0 model.
