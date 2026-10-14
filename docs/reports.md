# Reports Index

Every phase writes `reports/phase<N>_<name>.json`; the next phase reads those measured
numbers and never re-estimates them. This page maps the audit trail.

## Phase reports

| File | Phase | Contents |
|---|---|---|
| `phase0_env.json` | 0: Environment | platform (macOS 26.5, arm64, Python 3.13.9, torch 2.14.0, MPS), batch benchmarks, I/O probe (192.8 vs 53.5 img/s → 3.6×), gate PASS |
| `phase1_data.json` | 1: Data & split | 10,569 images, 9,172 clusters, 1,397 redundant, databar counts, band edges, split counts, leak check (0 cross-split dupes), gate PASS |
| `phase2_data.json` | 2: Dataloader | patch throughput by worker count, 0 databar violations in 2,000 patches, gate PASS |
| `phase3_overfit.json` | 3: Overfit gate | 8-patch overfit history, 39.72 dB final, 35 dB at step 825, gate recorded FAIL (see below) |
| `phase4_calib.json` | 4: Calibration | 2,000-step profile, sustained 1.801 it/s, 2.204 GB peak, 30,797-step 5-hour budget derivation, gate PASS |
| `phase4_memory_diag.json` | 4: Memory | batch 16 vs 8 diagnosis (0.008 vs 1.16 it/s, +2.8 GB swap) |
| `phase5_segA.json` | 5: Resume proof | SIGTERM at step 2,150, resume from 2,151, 100-step segment |
| `phase5_segB.json` | 5: Full run | 11,796 steps total, val PSNR 26.223 / MS-SSIM 0.9295 / LPIPS 0.07698, 2.235 GB peak, 29.4% throttle, gate PASS |
| `phase6_eval.json` | 6: Evaluation | 300 test images, per-band PSNR/MS-SSIM/LPIPS + per-image records, tiled 512/64, 3.07 GB peak |
| `phase6_eval.json` (descriptors) | 6: Success criterion | MCRpy VF/S2/lineal-path per band, overall 9.00% vs <5% target, NOT MET |
| `phase6_desc_sample.npz` | 6: Descriptor sample | raw descriptor curves for the 80 sampled pairs (NumPy cross-check stored) |
| `gap_phase5_chain.log` | 5→6 | chain log proving phase 6 read phase 5's measured budget |

## Artifacts

| File | Contents |
|---|---|
| `phase2_patches.png` | sample training patches (databar-free, dihedral-augmented) |
| `phase3_overfit.png` | overfit loss curve |
| `phase6_grid.png` | original vs reconstruction grid on test samples |
| `phase4_calib.log`, `phase5_segA/B.log`, `phase6_eval*.log`, `phase6_desc.log`, `app_run.log` | raw run logs |
| `eval_grid*.png` | per-eval reconstruction grids |

## Gate history

| Phase | Gate | Verdict |
|---|---|---|
| 0 | MPS available · batch < 4 GB · SSD faster than external · 10,569 images | PASS |
| 1 | zero cross-split duplicate groups | PASS |
| 2 | no databar overlap · ≥ 200 patches/s | PASS |
| 3 | ≥ 35 dB at 500 overfit steps | **FAIL** (31.49 dB; wiring verified sound: 35 dB at step 825, 39.72 dB at 1,500; documented and proceeded) |
| 4 | loss decreasing · peak < 4 GB | PASS |
| 5 | resume proven · val targets reached within budget | PASS |
| 6 | descriptor discrepancy < 5% | **FAIL** (9.00%), open, see [roadmap](roadmap.md) |

Gates are recorded exactly as measured. A FAIL is never re-interpreted after the fact;
the project either documents why it proceeds (phase 3) or keeps the gate open as the
active work item (phase 6).

## Checkpoints

| File | Step | Note |
|---|---|---|
| `checkpoints/calib_best.pt` / `calib_last.pt` | 2,000 | Phase 4 calibration endpoint |
| `checkpoints/full_best.pt` | 11,500 | best val LPIPS 0.07788, the evaluated/shipped weights |
| `checkpoints/full_last.pt` | 11,796 | resume point for continued training |
