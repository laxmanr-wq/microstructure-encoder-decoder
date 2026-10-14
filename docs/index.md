# Microstructure Encoder / Decoder

**Documentation set · v2.0 · September 2026**

A faithful image-to-latent-to-image autoencoder for scanning-electron micrographs,
designed, trained and evaluated end-to-end on a fanless MacBook Air M1 with 8 GB of
unified memory. The system compresses a 1024×1024 micrograph to a 128×128×4 latent
(8:1 at fp16) and reconstructs it at native resolution.

**Fidelity is the deliverable, not generation.**

---

## Project at a glance

| | |
|---|---|
| **Task** | Reconstruction of microstructure micrographs (encode → latent → decode) |
| **Model** | Convolutional autoencoder, 3.77 M parameters, KL-regularized 4-channel latent |
| **Loss** | L1 + 1.0·LPIPS(AlexNet) + 0.25·(1 − MS-SSIM) |
| **Dataset** | 10,569 SEM micrographs, 1024×1024 8-bit grayscale, re-split 70/15/15 by cluster |
| **Hardware** | Apple M1 (7-core GPU), PyTorch MPS, fp32 only, 4 GB peak-memory budget |
| **Status** | Phases 0–7 complete · prototype app shipped · quantitative target **not yet met** |

## Current status

| Milestone | Result | Gate |
|---|---|---|
| Phase 0: Environment & throughput | 192.8 img/s from SSD cache (3.6× vs external drive) | PASS |
| Phase 1: De-leaked re-split | 9,172 clusters, zero cross-split duplicates | PASS |
| Phase 2: Dataloader | 1,527 patches/s, 0 databar violations in 2,000 | PASS |
| Phase 3: Overfit gate | 39.72 dB on 8 patches at 500 steps | recorded FAIL¹ |
| Phase 4: Calibration | 1.90 it/s sustained, 2.20 GB peak, 5 h → 30,797-step budget | PASS |
| Phase 5: Full training | 11,796 steps, val LPIPS 0.07698, 2.24 GB peak | PASS |
| Phase 6: Evaluation | PSNR 26.60 dB · MS-SSIM 0.9449 · LPIPS 0.0984 · descriptors 9.00% | metric² |
| Phase 7: Prototype app | Gradio app, verified on user upload at 29.91 dB | shipped |

¹ Wiring verified sound (35 dB crossed at step 825; 39.72 dB at 1,500). See
[evaluation → gates](evaluation.md#gate-discipline).
² **9.00% vs the <5% target**: the success criterion is not met. Only scale band 4
(finest) passes. See [results](results.md#success-criterion).

## Where to look

| I want to… | Go to |
|---|---|
| Understand the model and design decisions | [Architecture](architecture.md) |
| Understand the dataset, the leak, and the re-split | [Data pipeline](data.md) |
| Reproduce training / resume a run | [Training](training.md) |
| Read the measured results | [Results](results.md) |
| Run inference or the app | [Usage](usage.md) |
| Operate the machine safely (memory, throttle, resume) | [Operations](operations.md) |
| See what is open and what we would do next | [Roadmap](roadmap.md) |
| Trace every number to its source | [Reports index](reports.md) |
| Check the papers we build on (with DOIs) | [References](references.md) |

## Source map

```
prep/build_manifest.py     Phase 1: duplicate clustering + de-leaked split
scripts/probe.py           Phase 0: environment, I/O and batch-size probe
scripts/phase2_check.py    Phase 2: dataloader speed + databar compliance
scripts/phase3_overfit.py  Phase 3: single-batch overfit gate
src/train.py               Phase 4/5: trainer (AdamW, EMA, cosine schedule)
scripts/phase5_run.py      Phase 5: orchestration incl. deliberate kill/resume test
src/eval.py                Phase 6: native-res evaluation + tiling
scripts/descriptors.py     Phase 6: MCRpy descriptors + NumPy cross-check
src/infer.py               Phase 7: CLI inference
src/app.py                 Phase 7: Gradio prototype app
src/train_gan.py, src/gan.py  Phase 8: optional PatchGAN (off by default)
```

Every phase writes `reports/phase<N>_<name>.json`; the next phase reads those measured
numbers and never re-estimates them. This documentation quotes only measured values.
