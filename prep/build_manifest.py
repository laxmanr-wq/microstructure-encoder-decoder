"""Phase 1 — rebuild the split from scratch. The delivered split is leaked.

Outputs manifests/manifest.csv. Never modifies image files on disk.
"""
import os, re, csv, sys, json, time, hashlib, random
from multiprocessing import Pool, cpu_count

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
SSD_DATA = os.path.expanduser("~/ms_cache/dataset")
EXT_DATA = os.path.join(ROOT, "dataset")
DATA = SSD_DATA if os.path.isdir(SSD_DATA) else EXT_DATA
MANIFEST = os.path.join(ROOT, "manifests", "manifest.csv")
REPORT = os.path.join(ROOT, "reports", "phase1_data.json")

SEED = 1337
EXPECTED = 10569

# --- tunables (calibrated against the reference audit in the brief) ---
DHASH_W, DHASH_H = 17, 16          # 16x17 bilinear, horizontal gradient -> 256 bits
LOWC_STD = 6.0                     # skip dHash below this 64x64 std
HAMMING_T = 10                     # /256 bits for near-duplicate merge
FAMILY_MIN_BASE = 5
FAMILY_MAX_MEMBERS = 20
BAR_SEARCH = 0.20                  # outer 20% of height
BAR_STEP_MIN = 12.0
BAR_MAX_FRAC = 0.15                # cap crop at 15% of height
BAR_MARGIN = 4
N_BANDS = 5
SPLITS = (("train", 0.70), ("val", 0.15), ("test", 0.15))


# ---------------------------------------------------------------- enumerate
def enumerate_images(root):
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if not d.startswith("._")]
        for f in fn:
            if f.startswith("._") or f == ".DS_Store":
                continue
            out.append(os.path.join(dp, f))
    return sorted(out)


def orig_split_of(path):
    rel = os.path.relpath(path, DATA)
    top = rel.split(os.sep)[0]
    return top.strip() or top


# ---------------------------------------------------------------- basename family
UUID_RE = re.compile(r"_[0-9a-fA-F]{8}(?:-?[0-9a-fA-F]{4}){3}-?[0-9a-fA-F]{12}$")
HEXTAIL_RE = re.compile(r"_[0-9a-fA-F]{16,}$")
PARENS_RE = re.compile(r"\(\d+\)$")


def family_base(fname):
    b = fname
    # strip the ['...'] list-literal wrapper
    m = re.match(r"^\[?['\"](.+?)['\"]\]?$", b)
    if m:
        b = m.group(1)
    # strip double extension (.tif.jpg) and the single extension
    b = re.sub(r"\.(tif|tiff|png|jpg|jpeg|bmp)\.(jpg|jpeg|png)$", "", b, flags=re.I)
    b = re.sub(r"\.(jpg|jpeg|png|tif|tiff|bmp)$", "", b, flags=re.I)
    m = re.match(r"^\[?['\"](.+?)['\"]\]?$", b)
    if m:
        b = m.group(1)
    b = b.strip()
    # trailing _<uuid> / long hex token
    b = UUID_RE.sub("", b)
    b = HEXTAIL_RE.sub("", b)
    # (n) copy suffix
    b = PARENS_RE.sub("", b).strip()
    b = re.sub(r"[ _-]copy$", "", b, flags=re.I).strip()
    # trailing "1" collision counter
    if len(b) > FAMILY_MIN_BASE and b.endswith("1"):
        b = b[:-1]
    return b.strip().lower()


# ---------------------------------------------------------------- per-image work
def dhash_hex(img_l):
    a = np.asarray(img_l.resize((DHASH_W, DHASH_H), Image.BILINEAR), dtype=np.float32)
    bits = (a[:, :-1] > a[:, 1:]).flatten()
    packed = np.packbits(bits)
    return packed.tobytes().hex()


def _bar_like(rowmean, rowstd, body_mean, body_std):
    """Per-row bar-likeness: far from body brightness, or far flatter than the body."""
    return (np.abs(rowmean - body_mean) > 20.0) | (rowstd < 0.5 * body_std)


def _strip_test(strip, step, body_mean, body_std):
    if strip.size == 0:
        return False
    s_mean, s_std = float(strip.mean()), float(strip.std())
    return step > BAR_STEP_MIN and (
        s_mean < 90.0 or s_std > 45.0
        or (s_std < 0.5 * body_std and abs(s_mean - body_mean) > 20.0))


def detect_bar(rowmean, rowstd, H, body_mean, body_std, top):
    """Strongest step in row-means within the outer 20%, then extend the boundary
    inward across contiguous bar-like rows so the whole bar is cropped, not just
    the strongest internal transition."""
    n = max(8, int(round(H * BAR_SEARCH)))
    like = _bar_like(rowmean, rowstd, body_mean, body_std)

    if top:
        seg = rowmean[:n]
        g = np.abs(np.diff(seg))
        if g.size == 0:
            return 0, 0.0
        i = int(np.argmax(g))
        step = float(g[i])
        r = i + 1                                   # strip is rows[:r]
        while r < n and like[r]:                    # extend inward
            r += 1
        ok = _strip_test(rowmean[:r], step, body_mean, body_std)
        if not ok:                                  # second chance: outermost bar-like run
            r2 = 0
            while r2 < n and like[r2]:
                r2 += 1
            if r2 >= 8:
                ins = rowmean[r2:r2 + 20]
                step2 = float(abs(rowmean[:r2].mean() - ins.mean())) if ins.size else 0.0
                if _strip_test(rowmean[:r2], step2, body_mean, body_std):
                    r, step, ok = r2, step2, True
        if not ok:
            return 0, step
        crop = r + BAR_MARGIN
    else:
        seg = rowmean[H - n:]
        g = np.abs(np.diff(seg))
        if g.size == 0:
            return 0, 0.0
        i = int(np.argmax(g))
        step = float(g[i])
        r = H - n + i + 1                           # strip is rows[r:]
        while r - 1 >= H - n and like[r - 1]:       # extend inward
            r -= 1
        ok = _strip_test(rowmean[r:], step, body_mean, body_std)
        if not ok:
            r2 = H
            while r2 - 1 >= H - n and like[r2 - 1]:
                r2 -= 1
            if H - r2 >= 8:
                ins = rowmean[max(0, r2 - 20):r2]
                step2 = float(abs(rowmean[r2:].mean() - ins.mean())) if ins.size else 0.0
                if _strip_test(rowmean[r2:], step2, body_mean, body_std):
                    r, step, ok = r2, step2, True
        if not ok:
            return 0, step
        crop = (H - r) + BAR_MARGIN
    crop = int(min(crop, int(H * BAR_MAX_FRAC)))
    return max(crop, 0), step


def radial_char_px(body):
    """Characteristic length in px from the radially averaged power spectrum."""
    h, w = body.shape
    n = 512
    if h < n or w < n:
        pad = np.zeros((max(h, n), max(w, n)), dtype=np.float32)
        pad[:h, :w] = body
        pad[h:, :] = body.mean()
        pad[:, w:] = body.mean()
        body = pad
        h, w = body.shape
    y0, x0 = (h - n) // 2, (w - n) // 2
    c = body[y0:y0 + n, x0:x0 + n].astype(np.float32)
    c = c - c.mean()                                   # DC removed
    win = np.hanning(n).astype(np.float32)
    c = c * win[:, None] * win[None, :]
    F = np.fft.rfft2(c)
    P = (F.real ** 2 + F.imag ** 2)
    fy = np.fft.fftfreq(n)[:, None] * n
    fx = np.arange(P.shape[1])[None, :]
    r = np.sqrt(fy ** 2 + fx ** 2).astype(np.int32)
    nb = n // 2
    ring = np.bincount(r.ravel(), weights=P.ravel(), minlength=nb + 1)[:nb + 1]
    ring[0] = 0.0                                      # DC
    tot = ring.sum()
    if tot <= 0:
        return float("nan")
    cum = np.cumsum(ring)
    k = int(np.searchsorted(cum, 0.5 * tot))
    k = max(k, 1)
    return float(n / k)                                # 1 / (k/n) cycles-per-px


def process(path):
    try:
        with open(path, "rb") as f:
            raw = f.read()
        md5 = hashlib.md5(raw).hexdigest()
        with Image.open(path) as im:
            im = im.convert("L")
            w, h = im.size
            small = np.asarray(im.resize((64, 64), Image.BILINEAR), dtype=np.float32)
            std64 = float(small.std())
            low_contrast = int(std64 < LOWC_STD)
            dh = "" if low_contrast else dhash_hex(im)
            a = np.asarray(im, dtype=np.float32)

        rowmean = a.mean(axis=1)
        rowstd = a.std(axis=1)
        b0, b1 = int(h * 0.2), int(h * 0.8)
        body = a[b0:b1]
        body_mean, body_std = float(body.mean()), float(body.std())
        crop_top, step_t = detect_bar(rowmean, rowstd, h, body_mean, body_std, top=True)
        crop_bot, step_b = detect_bar(rowmean, rowstd, h, body_mean, body_std, top=False)
        if crop_top + crop_bot >= h - 160:
            crop_top = crop_bot = 0

        core = a[crop_top:h - crop_bot]
        sat_frac = float(((core <= 2) | (core >= 253)).mean())
        # flat_frac: fraction of 8x8 blocks (on a 256x256 view) with std < 2
        cs = np.asarray(Image.fromarray(core.astype(np.uint8)).resize((256, 256), Image.BILINEAR),
                        dtype=np.float32)
        blk = cs.reshape(32, 8, 32, 8).transpose(0, 2, 1, 3).reshape(-1, 64)
        flat_frac = float((blk.std(axis=1) < 2.0).mean())
        char_px = radial_char_px(core)

        return dict(path=path, w=w, h=h, md5=md5, dhash=dh, std=round(std64, 3),
                    low_contrast=low_contrast, crop_top=crop_top, crop_bottom=crop_bot,
                    step_top=round(step_t, 2), step_bottom=round(step_b, 2),
                    sat_frac=round(sat_frac, 5), flat_frac=round(flat_frac, 5),
                    char_px=round(char_px, 3), error="")
    except Exception as e:                                    # noqa: BLE001
        return dict(path=path, w=0, h=0, md5="", dhash="", std=0.0, low_contrast=1,
                    crop_top=0, crop_bottom=0, step_top=0.0, step_bottom=0.0,
                    sat_frac=0.0, flat_frac=1.0, char_px=float("nan"), error=str(e))


# ---------------------------------------------------------------- union-find
class UF:
    def __init__(self, n):
        self.p = list(range(n))
        self.r = [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self.r[ra] < self.r[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.r[ra] == self.r[rb]:
            self.r[ra] += 1
        return True


POPCNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def hamming_merge(uf, rows, idxs):
    """Merge near-duplicates by <=HAMMING_T bit distance over 256-bit dHashes."""
    if not idxs:
        return 0
    mats = np.stack([np.frombuffer(bytes.fromhex(rows[i]["dhash"]), dtype=np.uint8)
                     for i in idxs])
    n = len(idxs)
    merged = 0
    CH = 256
    for s in range(0, n, CH):
        e = min(s + CH, n)
        d = POPCNT[np.bitwise_xor(mats[s:e, None, :], mats[None, :, :])].sum(axis=2)
        d[:, :s] = 255                       # keep it upper-triangular
        for a in range(e - s):
            d[a, : s + a + 1] = 255
        ii, jj = np.nonzero(d <= HAMMING_T)
        for a, b in zip(ii, jj):
            if uf.union(idxs[s + a], idxs[b]):
                merged += 1
    return merged


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    random.seed(SEED)
    files = enumerate_images(DATA)
    print(f"enumerated {len(files)} images from {DATA}")
    assert len(files) == EXPECTED, f"expected {EXPECTED}, found {len(files)}"

    nw = max(2, min(8, cpu_count()))
    with Pool(nw) as pool:
        rows = []
        for i, r in enumerate(pool.imap(process, files, chunksize=32)):
            rows.append(r)
            if (i + 1) % 1000 == 0:
                print(f"  scanned {i+1}/{len(files)}  ({time.time()-t0:.0f}s)", flush=True)
    errs = [r for r in rows if r["error"]]
    print(f"scan done in {time.time()-t0:.0f}s, errors={len(errs)}")

    n = len(rows)
    uf = UF(n)

    # (1) exact md5
    by_md5 = {}
    for i, r in enumerate(rows):
        by_md5.setdefault(r["md5"], []).append(i)
    md5_groups = {k: v for k, v in by_md5.items() if len(v) > 1}
    for v in md5_groups.values():
        for j in v[1:]:
            uf.union(v[0], j)

    # (2) dHash near-duplicates (skip low-contrast)
    dh_idx = [i for i, r in enumerate(rows) if r["dhash"]]
    by_dh = {}
    for i in dh_idx:
        by_dh.setdefault(rows[i]["dhash"], []).append(i)
    dhash_groups = {k: v for k, v in by_dh.items() if len(v) > 1}
    for v in dhash_groups.values():
        for j in v[1:]:
            uf.union(v[0], j)
    n_ham = hamming_merge(uf, rows, dh_idx)
    print(f"md5 groups={len(md5_groups)} dhash-exact groups={len(dhash_groups)} "
          f"hamming merges={n_ham}")

    # (3) base-name families
    fam = {}
    for i, r in enumerate(rows):
        fam.setdefault(family_base(os.path.basename(r["path"])), []).append(i)
    fam_merged = 0
    fam_id = {}
    for fid, (base, members) in enumerate(sorted(fam.items())):
        for i in members:
            fam_id[i] = fid
        if len(base) >= FAMILY_MIN_BASE and len(members) <= FAMILY_MAX_MEMBERS:
            for j in members[1:]:
                if uf.union(members[0], j):
                    fam_merged += 1
    print(f"basename families={len(fam)} merges={fam_merged}")

    # clusters
    croot = {}
    for i in range(n):
        croot.setdefault(uf.find(i), []).append(i)
    clusters = list(croot.values())
    cid = {}
    for c, members in enumerate(sorted(clusters, key=lambda m: rows[m[0]]["path"])):
        for i in members:
            cid[i] = c
    n_clusters = len(clusters)
    redundant = n - n_clusters
    print(f"clusters={n_clusters} redundant={redundant}")

    # scale bands from char_px quantiles
    cp = np.array([r["char_px"] for r in rows], dtype=np.float64)
    valid = np.isfinite(cp)
    qs = np.quantile(cp[valid], [i / N_BANDS for i in range(1, N_BANDS)])
    band = np.digitize(np.nan_to_num(cp, nan=float(np.median(cp[valid]))), qs)
    print("band edges (px):", [round(float(q), 2) for q in qs])

    # keep flag: cluster representative + usable
    bars_top = sum(1 for r in rows if r["crop_top"] > 0)
    bars_bot = sum(1 for r in rows if r["crop_bottom"] > 0)
    suspicion = []
    for i, r in enumerate(rows):
        s = 0
        if r["low_contrast"]:
            s += 1
        if r["flat_frac"] > 0.60:
            s += 1
        if r["sat_frac"] > 0.30:
            s += 1
        if (r["h"] - r["crop_top"] - r["crop_bottom"]) < 256 or r["w"] < 256:
            s += 2
        if r["error"]:
            s += 3
        suspicion.append(s)

    # representative of each cluster = lowest suspicion, then largest body, then path
    rep = {}
    for c_members in croot.values():
        best = min(c_members, key=lambda i: (suspicion[i],
                                             -(rows[i]["h"] - rows[i]["crop_top"] - rows[i]["crop_bottom"]),
                                             rows[i]["path"]))
        rep[uf.find(c_members[0])] = best
    keep = [0] * n
    for i in range(n):
        if rep[uf.find(i)] == i and suspicion[i] < 2:
            keep[i] = 1

    # split whole clusters, stratified by dominant scale band, seed 1337
    rng = random.Random(SEED)
    cl_by_band = {}
    for c_members in croot.values():
        bands = [int(band[i]) for i in c_members]
        dom = max(set(bands), key=bands.count)
        cl_by_band.setdefault(dom, []).append(c_members)
    split_of = {}
    tally = {s: 0 for s, _ in SPLITS}
    for b in sorted(cl_by_band):
        cls = cl_by_band[b]
        rng.shuffle(cls)
        tot = sum(len(c) for c in cls)
        got = {s: 0 for s, _ in SPLITS}
        for c_members in cls:
            deficit = [( (frac * tot - got[s]) / max(frac, 1e-9), s) for s, frac in SPLITS]
            s = max(deficit)[1]
            got[s] += len(c_members)
            for i in c_members:
                split_of[i] = s
        for s in got:
            tally[s] += got[s]
    print("split image counts:", tally)

    # write manifest
    os.makedirs(os.path.dirname(MANIFEST), exist_ok=True)
    cols = ["path", "split", "orig_split", "cluster", "scale_band", "char_px",
            "crop_top", "crop_bottom", "std", "low_contrast", "w", "h", "md5", "dhash",
            "flat_frac", "sat_frac", "suspicion", "tile_id", "keep"]
    with open(MANIFEST, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for i, r in enumerate(rows):
            wr.writerow([r["path"], split_of[i], orig_split_of(r["path"]), cid[i],
                         int(band[i]), r["char_px"], r["crop_top"], r["crop_bottom"],
                         r["std"], r["low_contrast"], r["w"], r["h"], r["md5"], r["dhash"],
                         r["flat_frac"], r["sat_frac"], suspicion[i], fam_id[i], keep[i]])
    print("wrote", MANIFEST)

    # ---- GATE: independent leak check, recomputed from raw md5/dhash columns ----
    leaks = {"md5": 0, "dhash": 0, "hamming": 0}
    examples = []
    for k, v in by_md5.items():
        if len(v) > 1 and len({split_of[i] for i in v}) > 1:
            leaks["md5"] += 1
            if len(examples) < 5:
                examples.append({"kind": "md5", "n": len(v),
                                 "paths": [os.path.basename(rows[i]["path"]) for i in v[:3]]})
    for k, v in by_dh.items():
        if len(v) > 1 and len({split_of[i] for i in v}) > 1:
            leaks["dhash"] += 1
            if len(examples) < 5:
                examples.append({"kind": "dhash", "n": len(v),
                                 "paths": [os.path.basename(rows[i]["path"]) for i in v[:3]]})
    # near-duplicate leak check by hamming, independent of cluster ids
    if dh_idx:
        mats = np.stack([np.frombuffer(bytes.fromhex(rows[i]["dhash"]), dtype=np.uint8)
                         for i in dh_idx])
        sp = np.array([{"train": 0, "val": 1, "test": 2}[split_of[i]] for i in dh_idx])
        CH = 256
        for s in range(0, len(dh_idx), CH):
            e = min(s + CH, len(dh_idx))
            d = POPCNT[np.bitwise_xor(mats[s:e, None, :], mats[None, :, :])].sum(axis=2)
            for a in range(e - s):
                d[a, : s + a + 1] = 255
            ii, jj = np.nonzero(d <= HAMMING_T)
            leaks["hamming"] += int((sp[s + ii] != sp[jj]).sum())
    gate_pass = sum(leaks.values()) == 0
    print("LEAK CHECK:", leaks, "->", "PASS" if gate_pass else "FAIL")

    ref = {"clusters": 9458, "redundant": 1111, "bottom_bars": 4356, "top_bars": 684,
           "band_edges": [12.2, 21.3, 34.1, 56.9]}
    rep_json = {
        "phase": 1, "data_root": DATA, "n_images": n, "errors": len(errs),
        "clusters": n_clusters, "redundant": redundant,
        "md5_dup_groups": len(md5_groups), "dhash_exact_groups": len(dhash_groups),
        "hamming_merges": n_ham, "hamming_threshold": HAMMING_T,
        "basename_families": len(fam), "basename_merges": fam_merged,
        "bars_top": bars_top, "bars_bottom": bars_bot,
        "median_bar_px": float(np.median([r["crop_bottom"] for r in rows if r["crop_bottom"] > 0]))
        if bars_bot else 0.0,
        "band_edges_px": [round(float(q), 3) for q in qs],
        "band_counts": {int(b): int((band == b).sum()) for b in range(N_BANDS)},
        "split_counts": tally,
        "split_counts_keep": {s: sum(1 for i in range(n) if split_of[i] == s and keep[i])
                              for s, _ in SPLITS},
        "kept": int(sum(keep)),
        "reference_audit": ref,
        "deviation": {"clusters": n_clusters - ref["clusters"],
                      "redundant": redundant - ref["redundant"],
                      "bottom_bars": bars_bot - ref["bottom_bars"],
                      "top_bars": bars_top - ref["top_bars"]},
        "leak_check": leaks,
        "gate": {"zero_cross_split_duplicates": gate_pass, "PASS": gate_pass},
        "seconds": round(time.time() - t0, 1),
    }
    with open(REPORT, "w") as f:
        json.dump(rep_json, f, indent=2)
    print("wrote", REPORT)
    print("GATE:", "PASS" if gate_pass else "FAIL")


if __name__ == "__main__":
    main()
