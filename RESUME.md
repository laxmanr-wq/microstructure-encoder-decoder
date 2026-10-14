# Resume point

Updated 17 Sep 2026. **Phases 0–6 are complete.** The prototype app works.
**Phase 6 measured the success criterion and it is NOT met: 9.00% vs the <5% target.**

## Phase 6 RESULT — measured 17 Sep on 300 test images (`reports/phase6_eval.json`)

Run with `--limit 300 --tile 512 --overlap 64`, ckpt `full_best.pt` step 11,500.
Peak **3.07 GB** (under the 4 GB budget), 3,236 s, 0.09 img/s.

Image fidelity:

| band | n | PSNR | MS-SSIM | LPIPS |
|---|---|---|---|---|
| 0 | 42 | 22.60 | 0.8858 | 0.1485 |
| 1 | 63 | 25.47 | 0.9426 | 0.1011 |
| 2 | 54 | 26.87 | 0.9527 | 0.0960 |
| 3 | 71 | 27.59 | 0.9615 | 0.0820 |
| 4 | 70 | 28.82 | 0.9597 | 0.0845 |
| **all** | **300** | **26.60** | **0.9449** | **0.0984** |

MCRpy descriptor discrepancy (16 pairs/band, 256 px crops, numpy cross-check stored):

| band | VF | S2 | LP | discrepancy | <5%? |
|---|---|---|---|---|---|
| 0 | 12.88% | 20.21% | 19.64% | **17.58%** | NO |
| 1 | 5.02% | 9.64% | 9.38% | **8.01%** | NO |
| 2 | 3.71% | 6.00% | 6.17% | **5.29%** | NO |
| 3 | 4.51% | 11.54% | 12.99% | **9.68%** | NO |
| 4 | 3.43% | 4.80% | 5.12% | **4.45%** | **YES** |
| **overall** | | | | **9.00%** | **NO** |

Up from 23.51% at the calibration checkpoint, so training helped a lot — but the target is
still missed by ~1.8x. **Only band 4 (finest scale) passes.** Band 0 is the worst on both
fidelity and descriptors. Volume fraction is consistently the best-preserved descriptor;
**S2 and lineal path are the failing ones**, i.e. spatial correlation structure is lost
more than phase proportion.

## Prototype app — WORKING

```bash
source .venv/bin/activate
export PYTORCH_ENABLE_MPS_FALLBACK=1 PYTHONUNBUFFERED=1
python -u src/app.py     # -> http://127.0.0.1:7860
```

Upload -> encode -> decode -> side-by-side + metrics + PNG/latent download. Verified on a
user-uploaded fracture micrograph: PSNR 29.91 dB, MS-SSIM 0.9627, LPIPS 0.0849, 8:1.

**`src/app.py` default inference mode was changed to `tile 512 / overlap 64`** (was
`whole image`). Do not revert on this machine: whole-image peaks 5.07 GB, over the 4 GB
budget, and thrashes.

## KNOWN BUG — tiled fallback cannot self-trigger on MPS

`src/eval.py: recon_full()` falls back to tiling only inside `except RuntimeError`. On MPS
the allocation never raises — macOS silently swaps instead — so the fallback never fires and
the run degrades into thrashing (observed: 2 h 40 m, <300 images, decelerating, would have
needed 13–17 h). **Always pass `--tile 512 --overlap 64` explicitly.** A real fix would be to
choose tiling by input size rather than by exception.

## Current state — measured, do not re-derive

Phase 5 finished on 13 Sep and its gate **PASSED**
(`reports/phase5_segB.json`, `reports/gap_phase5_chain.log`):

- Trained to **step 11,796** of the 30,797-step budget, then stopped early on the time budget.
- `checkpoints/full_best.pt` is **step 11,500**, best val LPIPS **0.07788**.
- Final val: **PSNR 26.223 · MS-SSIM 0.9295 · LPIPS 0.07698 · L1 0.0804**.
- Peak memory **2.235 GB** (budget 4 GB), 2.364 h wall clock, sustained 1.083 it/s
  (29.4% thermal throttle), batch 8, 3,772,841 params.
- Resume-after-kill was proved in segment A (SIGTERM at step 2150, resumed from step 2151).

The Phase 4 smoke tests all passed, but note the descriptor result **from the weak
calibration checkpoint**: overall discrepancy **23.51%** vs the <5% target. That number is
from `calib`, NOT from `full_best.pt`. Phase 6 must re-measure it — this is the open question.

## Next action — decide the goal first

The app is shippable as a **visual/compression** prototype today. It is **not** yet
validated for quantitative microstructure measurement (9.00% > 5%).

To close the descriptor gap, in rough order of expected value:

1. **Train longer.** Phase 5 stopped at step 11,796 of a 30,797 budget — only 38%. The
   trend from 23.51% -> 9.00% has not flattened. Cheapest credible win. Resume from
   `checkpoints/full_last.pt`.
2. **Phase 7 PatchGAN** (`src/train_gan.py`). Adversarial loss restores high-frequency
   texture, which is exactly what S2/lineal-path errors indicate is missing. README rule:
   keep only if LPIPS **and** descriptors both improve.
3. **Add a descriptor-aware loss term** (e.g. S2 correlation penalty). Directly targets the
   failing metric instead of hoping perceptual loss carries it.
4. **Investigate band 0.** Worst on every metric. Check whether band-0 patches are
   under-sampled during training or whether f=8 is simply too aggressive at coarse scale.

Running the remaining 1,102 test images is **not** a priority — 300 already gives stable
per-band statistics.

## Memory operating rules (learned the hard way 17 Sep)

- Need **≥40% free** before a long run. Check `memory_pressure` and `sysctl vm.swapusage`.
- Quitting Chrome (~806 MB / 24 procs) and Antigravity IDE (~447 MB / 20 procs) is usually
  enough; reboot only if swap is already saturated.
- Always `-u` / `PYTHONUNBUFFERED=1`. Without it stdout is block-buffered until the
  100-image flush, which makes a thrashing run look identical to a healthy one.
- `src/eval.py` writes its JSON only at the end — interrupting costs time, never data.

### Resume-after-reboot shortcut (added 17 Sep)

On 17 Sep the user chose **reboot**. After restarting:

1. Reopen only VS Code (+ terminal). No Chrome, no Brave, no Antigravity IDE.
2. Mount the external drive, then run the read-only gate check:
   `bash scripts/preflight.sh`
   It checks memory ≥40%, swap, and the 10,569-image cache, and prints the
   Phase 6 commands on PASS. It modifies and deletes nothing.
3. On PASS, run the Phase 6 block above.

## Decisions already made — don't re-debate

- **Phase 3 gate recorded as FAIL** — 31.49 dB at 500 steps. The wiring was verified sound
  (35 dB crossed at step 825, 39.72 dB at 1,500), so the project proceeds.
- **Batch 8, not 16.** Batch 16 thrashes on this machine (0.008 it/s);
  `reports/phase4_memory_diag.json`.
- **`num_workers: 0`** for train and val: each spawned worker re-imports torch (~400 MB).
- **Model:** output activation `none`, `logvar_init: 0.0` — both from Phase 3 ablations.
- The manifest stores **absolute paths into `~/ms_cache/dataset`** (the fast SSD cache), so
  training and eval never read the slow external drive. Checkpoints/reports stay external.

## What healthy looks like

- Bare model, batch 8, synthetic data: **~2.4 it/s**.
- Full training, batch 8: **~1.6–1.9 it/s** early, ~1.1 it/s once throttled; peak 2.2 GB.
- Below ~0.5 it/s it is memory thrashing: check `sysctl vm.swapusage` and
  `memory_pressure`, and close what else is open.
