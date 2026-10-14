# Data Pipeline

**10,569 images · 1024×1024 · 8-bit grayscale · re-split 70/15/15 by duplicate cluster**

## The dataset

Scanning-electron micrographs spanning magnifications from 10× to 500,000×: steel
microstructure, fracture surfaces, cast iron, and coating cross-sections (some with
measurement text printed on them). All files are 1024×1024 8-bit grayscale JPEG, zero
corrupt. Roughly 2 GB. 2,750 files are TIFF→JPEG re-encodes (~5:1), so JPEG artifacts
are baked in and unavoidable.

## Environment traps handled by the pipeline

1. **AppleDouble sidecars.** The external exFAT drive carries 10,573 `._*` shadow files;
   `os.listdir` sees 21,142 entries and PIL raises `UnidentifiedImageError` on every
   sidecar. The manifest builder filters basenames starting with `._` and `.DS_Store`.
2. **A directory named `val ` with a trailing space.** The delivered `train/` also nests
   one extra level (`7396/`). Neither fact is visible in a casual `ls`.
3. **The external drive is the bottleneck.** Measured 53.5 img/s external vs 192.8 img/s
   from the internal-SSD cache, a **3.6× speedup**. The dataset is rsynced to
   `~/ms_cache/dataset/` (excluding sidecars) and all training/eval reads hit the cache.
   Checkpoints and reports stay on the external drive.

## The leak, and the re-split

The delivered 70/15/15 split was audited and found **leaked**: 245 exact-duplicate and
284 near-duplicate groups straddled splits, and 319 held-out images (10.1% of test) had
a train twin. Any metric computed on it would be optimistic.

`prep/build_manifest.py` rebuilds the split:

1. **Enumerate** the 10,569 real images.
2. **Cluster duplicates** with union-find over three signals:
   - **md5**: exact copies;
   - **256-bit dHash**: perceptual hashes, merged at Hamming distance ≤ 10. Skipped
     below a 64×64 pixel std of 6.0, where featureless images all collide;
   - **base-name families**: catches re-encodes of the same source frame.
3. **Detect SEM databars** on both edges (info banners burned into the micrographs) and
   store `crop_top` / `crop_bottom` per image. **Files on disk are never modified**;
   the dataloader crops on read.
4. **Assign a scale band** (0 = coarsest … 4 = finest) from the radially averaged power
   spectrum, with band edges at 12.2 / 21.3 / 34.1 / 56.9 px characteristic size.
5. **Split whole clusters** 70/15/15, stratified by dominant scale band, seed 1337.

### Measured manifest (Phase 1, `reports/phase1_data.json`)

| Signal | Count |
|---|---|
| Images | 10,569 (0 errors) |
| Clusters (after union-find) | 9,172 |
| Redundant images | 1,397 (~13%) |
| md5 duplicate groups | 496 |
| dHash exact groups | 579 (+83 Hamming merges) |
| Base-name families / merges | 9,615 / 685 |
| Databars: bottom / top | 4,103 / 994 (median bar 56 px) |
| Band counts 0–4 | 1,992 / 2,125 / 2,126 / 1,907 / 2,419 |

### Split counts

| Split | All images | Kept (`keep=1`) |
|---|---|---|
| train | 7,392 | 6,387 |
| val | 1,590 | 1,373 |
| test | 1,587 | 1,402 |

`keep=0` marks images excluded from sampling (e.g. flagged suspicious by the audit);
the dataloader reads the manifest's `split` column **as-is and never re-splits**.

### Leak gate

Verified independently of the cluster ids by re-checking raw md5, dHash and Hamming
distance against the final split: **zero cross-split duplicate groups**. Gate PASS.

## Dataloader (Phase 2)

`src/data.py`, 128×128 patches, batch 8, normalised to [−1, 1] as `1×H×W` float32.

- **Augmentation:** 8-fold dihedral (rot90 + flip), micrographs are rotation-invariant.
- **Databar safety:** patches are sampled only from the body region; a 2,000-patch
  audit found **0 violations**.
- **Throughput:** 275 patches/s at `num_workers=0`, 1,527 patches/s at 6 workers.
  Training settled on `num_workers=0` anyway: every spawned worker re-imports torch
  (~400 MB) on this 8 GB machine, and the model only consumes ~10 patches/s.
- **Cache:** manifest paths point at `~/ms_cache/dataset` (internal SSD).

Source: `prep/build_manifest.py`, `src/data.py`, `reports/phase1_data.json`,
`reports/phase2_data.json`.
