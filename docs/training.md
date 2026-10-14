# Training

**AdamW · fp32 · EMA 0.999 · cosine schedule with 200-step warmup · batch 8 · seed 1337**

## Configuration

Full config in [`configs/base.yaml`](../configs/base.yaml). Key values:

| Group | Setting |
|---|---|
| Optimizer | AdamW, lr 2e-4, betas (0.9, 0.95), weight decay 0.01, grad clip 1.0 |
| Schedule | 200-step linear warmup → cosine decay to 5% floor |
| EMA | decay 0.999, float32 shadow; EMA weights are what evaluation loads |
| Batch | 8 (see below) |
| Precision | fp32 only, MPS bf16/fp16 autocast is unreliable |
| Checkpoints | best + last only (disk is tight), atomic writes via `.tmp` + rename |

## Why batch 8, not 16

Phase 0's probe model suggested batch 16, but the real model plus LPIPS changed the
picture. `reports/phase4_memory_diag.json` measured batch 16 at **0.008 it/s with
+2.8 GB of swap**: thrashing, while batch 8 runs **1.16–1.9 it/s at 2.2 GB peak**.
The project rule applies: *cut batch size, never model correctness.*

`num_workers: 0` for the same reason: each spawned data-loader worker re-imports torch
(~400 MB), and the trainer only needs ~10 patches/s.

## The two runs

### Phase 4: calibration (2,000 steps, ~18 min)

Purpose: measure sustained throughput on the real workload and derive the step budget.
Sustained 1.9 → 1.8 it/s as the fanless machine throttled (16.8%); peak 2.20 GB.
At the recommended 0.95 overhead margin this yields **30,797 steps per 5-hour budget**,
which Phase 5 adopts verbatim, the chain reads measured numbers, never re-estimates.
Final val: PSNR 22.59 dB · MS-SSIM 0.8714 · LPIPS 0.1888. Loss fell 0.509 → 0.226
(first vs last quarter).

### Phase 5: full training

- Resumes from the calibration checkpoint (step 2,000), no warm restart of the
  schedule values, the run continues to the 30,797-step budget.
- **Resume is proven, not assumed:** 150 steps in, the orchestrator
  (`scripts/phase5_run.py`) deliberately SIGTERMs the run and restarts it from the
  checkpoint. Segment A: killed at step 2,150, resumed cleanly from 2,151.
- Checkpoints every 500 steps, the external drive may be unplugged mid-run.
- **Stopped at step 11,796** when the *time* budget expired (38% of the step budget).
  `checkpoints/full_best.pt` holds step **11,500**, best val LPIPS **0.07788**.

Final validation (EMA weights, `reports/phase5_segB.json`):

| Metric | Value |
|---|---|
| PSNR | 26.223 dB |
| MS-SSIM | 0.9295 |
| LPIPS (AlexNet) | 0.07698 |
| L1 | 0.0804 |
| Peak memory | 2.235 GB (budget 4 GB) |
| Wall clock | 2.364 h, sustained 1.083 it/s (29.4% thermal throttle) |

## Monitoring built into the trainer

- **Wall-clock it/s every 50 steps**: on a fanless machine this is the thermal-throttle
  gauge. Healthy: ~1.6–1.9 it/s early, ~1.1 once throttled. Below ~0.5 it/s is memory
  thrashing, not training (see [operations](operations.md)).
- **Peak MPS memory** logged per window via `torch.mps.driver_allocated_memory()`.
- **Val every 500 steps** on 24×8 = 192 fresh patches.
- **Clean SIGTERM handling**: checkpoint, flush, exit; resume picks up at step + 1.

## Reproducing

```bash
# calibration
python src/train.py --steps 2000 --tag calib --report reports/phase4_calib.json

# full run (resumes from calib_last, budget from phase4_calib.json, proves resume)
python scripts/phase5_run.py

# continue training from the last checkpoint (recommended next step, see roadmap)
python src/train.py --resume checkpoints/full_last.pt
```

Source: `src/train.py`, `scripts/phase5_run.py`, `reports/phase4_calib.json`,
`reports/phase5_segA.json`, `reports/phase5_segB.json`.
