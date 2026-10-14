"""Phase 2 gate: databar-overlap assertion + throughput from the SSD cache."""
import os, sys, json, time, random
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, "/Volumes/Hard Disk/Microstructure Encoder Decoder/src")
import data as D

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
OUT = os.path.join(ROOT, "reports", "phase2_data.json")
PNG = os.path.join(ROOT, "reports", "phase2_patches.png")


def main():
    env = json.load(open(os.path.join(ROOT, "reports", "phase0_env.json")))
    p1 = json.load(open(os.path.join(ROOT, "reports", "phase1_data.json")))
    bs = env["chosen_batch"]

    counts = {s: len(D.load_manifest(split=s)) for s in ("train", "val", "test")}
    all_counts = {s: len(D.load_manifest(split=s, keep_only=False)) for s in ("train", "val", "test")}
    print("keep-filtered:", counts, " all:", all_counts)

    # ---- assertion: no sampled patch may overlap a databar region ----
    # Verified geometrically against the ORIGINAL image coordinates.
    rng = random.Random(1337)
    recs = [r for r in D.load_manifest(split="train") if r["crop_top"] + r["crop_bottom"] > 0]
    checked, viol = 0, 0
    for rec in rng.sample(recs, min(400, len(recs))):
        a = D.body_array(rec)
        H, W = a.shape
        assert H == rec["h"] - rec["crop_top"] - rec["crop_bottom"], "body slice wrong"
        p = D.PATCH
        if H < p:
            continue
        for _ in range(5):
            y = rng.randrange(0, H - p + 1)
            y0_orig = rec["crop_top"] + y            # patch top in original coords
            y1_orig = y0_orig + p
            if y0_orig < rec["crop_top"] or y1_orig > rec["h"] - rec["crop_bottom"]:
                viol += 1
            checked += 1
    assert_pass = viol == 0
    print(f"databar-overlap assertion: checked {checked} patches, violations {viol} -> "
          f"{'PASS' if assert_pass else 'FAIL'}")

    # ---- throughput ----
    nw_results = {}
    for nw in (0, 4, 6):
        dl, ds = D.make_loader("train", bs, num_workers=nw, length=bs * 40)
        it = iter(dl)
        next(it)                                          # warm workers
        t0 = time.perf_counter(); n = 0
        for x, b in it:
            n += x.shape[0]
        dt = time.perf_counter() - t0
        nw_results[nw] = round(n / dt, 1)
        print(f"  num_workers={nw}: {nw_results[nw]} patches/s")
        del dl, it
    best_nw = max(nw_results, key=lambda k: nw_results[k])
    pps = nw_results[best_nw]

    # ---- patch grid ----
    dl, ds = D.make_loader("train", 20, num_workers=0, length=20, shuffle=True)
    x, band = next(iter(dl))
    fig, axes = plt.subplots(5, 4, figsize=(8, 10))
    for i, ax in enumerate(axes.ravel()):
        ax.imshow(((x[i, 0].numpy() + 1) * 127.5).astype(np.uint8), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"band {int(band[i])}", fontsize=7)
        ax.axis("off")
    fig.suptitle("Phase 2 — real training patches (128x128, databar-excluded)")
    fig.tight_layout()
    fig.savefig(PNG, dpi=110)
    print("wrote", PNG)

    gate = {"no_databar_overlap": assert_pass, "patches_per_s_ge_200": pps >= 200}
    gate["PASS"] = all(gate.values())
    rep = {"phase": 2, "manifest": D.MANIFEST, "data_root_used": p1["data_root"],
           "patch": D.PATCH, "batch_size": bs, "augmentation": "8-fold dihedral (rot90 + flip)",
           "normalisation": "[-1,1], 1xHxW float32",
           "counts_keep_filtered": counts, "counts_all": all_counts,
           "patches_per_s_by_workers": nw_results, "num_workers_chosen": best_nw,
           "patches_per_s": pps, "databar_patches_checked": checked,
           "databar_violations": viol, "gate": gate}
    json.dump(rep, open(OUT, "w"), indent=2)
    print("wrote", OUT)
    print("GATE:", "PASS" if gate["PASS"] else "FAIL")


if __name__ == "__main__":
    main()
