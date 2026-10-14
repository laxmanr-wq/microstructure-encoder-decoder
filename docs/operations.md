# Operations

**Running a memory-constrained ML workload on a fanless MacBook Air M1 (8 GB unified)**

This page is the runbook for keeping long jobs alive on the target machine. Everything
here was learned by measurement, mostly the hard way on 17 Sep 2026, and encoded into
`scripts/preflight.sh`.

## The memory trap

> **On MPS, allocation failures never raise.** macOS silently swaps instead. Code that
> falls back to tiled inference inside `except RuntimeError`, as `src/eval.py`'s
> `recon_full()` originally did, never triggers, and the run degrades into thrashing:
> observed 2 h 40 m for <300 images, decelerating, on track for 13–17 h.

Consequences:

- **Always pass `--tile 512 --overlap 64` explicitly** for eval and anything ≥ ~1024².
  The app's default is already tiled.
- A real fix (open, see [roadmap](roadmap.md)) is to choose tiling by input size, not by
  exception.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` must be set for ops MPS doesn't implement.

## Before any long run

```bash
bash scripts/preflight.sh     # read-only; modifies nothing
```

It checks and enforces:

- **≥ 40% free memory** (hard gate, blocks with a top-offender list),
- swap usage (`sysctl vm.swapusage`),
- the 10,569-image SSD cache is present,
- prints the Phase 6 command block on PASS.

Freeing memory is usually a matter of quitting Chrome (~806 MB / 24 procs) and the
Antigravity IDE (~447 MB / 20 procs). Reboot only if swap is already saturated.

## What healthy looks like

| Signal | Healthy | Trouble |
|---|---|---|
| Bare model, batch 8, synthetic data | ~2.4 it/s |, |
| Full training, batch 8, early | ~1.6–1.9 it/s |, |
| After thermal throttle | ~1.1 it/s |, |
| Any sustained phase |, | **< 0.5 it/s = thrashing** |

At < 0.5 it/s: check `sysctl vm.swapusage` and `memory_pressure`, close other apps,
and switch to tiled inference. Do not "wait it out", thrashing does not recover.

## Thermal throttling

The M1 is fanless: sustained load throttles 30–40% and *every* long run must be planned
around interruption. The trainer logs wall-clock it/s every 50 steps precisely so the
throttle is visible in the log (Phase 4 measured a 16.8% sag, Phase 5 29.4%).

## Uninterruptible-run discipline

- **Checkpoint every 500 steps**: the external drive may be unplugged mid-run. Only
  best + last are kept (disk is tight).
- **SIGTERM is handled**: checkpoint, flush, exit. Resume continues at step + 1. This
  was *proven* during Phase 5 by deliberately killing the run at step 2,150 and
  resuming from 2,151.
- **Always run with `-u` / `PYTHONUNBUFFERED=1`.** Without it, stdout is block-buffered
  until a 100-image flush, which makes a thrashing run look identical to a healthy one.
- **`src/eval.py` writes its JSON only at the end.** Interrupting costs time, never data.
- **Never re-split the dataset**: checkpoints are only meaningful against the Phase 1
  manifest (seed 1337).

## Resume after reboot

1. Reopen only VS Code + terminal. No Chrome, no Brave, no Antigravity IDE.
2. Mount the external drive, then run the read-only gate: `bash scripts/preflight.sh`.
3. On PASS, resume training (`src/train.py --resume checkpoints/full_last.pt`) or run
   the eval block it prints.

## Memory profile of the shipped components

| Component | Peak | Within 4 GB? |
|---|---|---|
| Training (batch 8) | 2.235 GB | yes |
| Eval, tiled 512/64 | 3.07 GB | yes |
| Eval, whole-image 1024² | 5.07 GB | **no, swaps** |
| Gradio app, tiled default | ≤ ~3 GB | yes |

## Storage layout

| Location | Content | Why |
|---|---|---|
| `~/ms_cache/dataset/` (internal SSD) | dataset copy, 10,569 images | 192.8 vs 53.5 img/s reads, 3.6× |
| External drive `checkpoints/` | `full_best.pt`, `full_last.pt`, `calib_*` | survives laptop loss; best+last only |
| External drive `reports/` | one JSON per phase + logs | the audit trail every number quotes |
| External drive `outputs/` | app/CLI artifacts | user-facing results |
