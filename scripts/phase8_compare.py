"""Phase 8 — keep/discard decision for PatchGAN sharpening.

KEEP ONLY IF descriptor discrepancy AND LPIPS both improve on the held-out test set.
If LPIPS improved (sharper) but descriptors got worse, the GAN is inventing
microstructure that isn't there — hallucination — and it is discarded."""
import os, json

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
REP = os.path.join(ROOT, "reports")
KEYS = ("psnr", "ms_ssim", "lpips")
DKEYS = ("descriptor_discrepancy_pct", "s2_pct_err", "lineal_path_pct_err", "volume_fraction_pct_err")


def main():
    base = json.load(open(os.path.join(REP, "eval.json")))
    gan = json.load(open(os.path.join(REP, "eval_gan.json")))
    for r in (base, gan):
        assert "per_band" in r.get("descriptors", {}), "run scripts/descriptors.py on both evals first"

    rows, halluc_bands = [], []
    for b in sorted(base["per_band"]):
        pb, pg = base["per_band"][b], gan["per_band"].get(b)
        db, dg = base["descriptors"]["per_band"].get(b), gan["descriptors"]["per_band"].get(b)
        if not (pb and pg and db and dg):
            continue
        # descriptors must be measured on the same images for the comparison to mean anything
        assert db["n"] == dg["n"], f"band {b}: descriptor sample sizes differ ({db['n']} vs {dg['n']})"
        row = {"band": b, "n": pb["n"], "desc_n": db["n"]}
        row.update({k: {"before": pb[k], "after": pg[k]} for k in KEYS})
        row.update({k: {"before": db[k], "after": dg[k]} for k in DKEYS})
        row["lpips_improved"] = pg["lpips"] < pb["lpips"]
        row["descriptors_improved"] = dg["descriptor_discrepancy_pct"] < db["descriptor_discrepancy_pct"]
        if row["lpips_improved"] and not row["descriptors_improved"]:
            halluc_bands.append(b)
        rows.append(row)

    ob, og = base["overall"], gan["overall"]
    dob, dog = base["descriptors"]["overall"], gan["descriptors"]["overall"]
    lpips_up = og["lpips"] < ob["lpips"]
    desc_up = dog["descriptor_discrepancy_pct"] < dob["descriptor_discrepancy_pct"]
    keep = lpips_up and desc_up
    if keep:
        verdict = "KEEP: LPIPS and descriptor discrepancy both improved"
    elif lpips_up:
        verdict = ("DISCARD - hallucination: sharper (LPIPS improved) but the microstructure "
                   "descriptors got worse")
    elif desc_up:
        verdict = "DISCARD: descriptors improved but LPIPS got worse"
    else:
        verdict = "DISCARD: neither LPIPS nor descriptor discrepancy improved"

    rep = {"phase": 8, "rule": "keep only if descriptor discrepancy AND LPIPS both improve",
           "overall": {**{k: {"before": ob[k], "after": og[k]} for k in KEYS},
                       **{k: {"before": dob[k], "after": dog[k]} for k in DKEYS}},
           "lpips_improved": lpips_up, "descriptors_improved": desc_up,
           "hallucination_bands": halluc_bands, "keep": keep, "verdict": verdict,
           "per_band": rows,
           "checkpoints": {"before": base.get("ckpt"), "after": gan.get("ckpt")}}
    json.dump(rep, open(os.path.join(REP, "phase8_gan.json"), "w"), indent=2)

    print("| band | PSNR before/after | MS-SSIM before/after | LPIPS before/after | descriptor % before/after |")
    print("|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['band']} | {r['psnr']['before']:.2f} / {r['psnr']['after']:.2f} | "
              f"{r['ms_ssim']['before']:.4f} / {r['ms_ssim']['after']:.4f} | "
              f"{r['lpips']['before']:.4f} / {r['lpips']['after']:.4f} | "
              f"{r['descriptor_discrepancy_pct']['before']:.2f} / {r['descriptor_discrepancy_pct']['after']:.2f} |")
    o = rep["overall"]
    print(f"| **all** | {o['psnr']['before']:.2f} / {o['psnr']['after']:.2f} | "
          f"{o['ms_ssim']['before']:.4f} / {o['ms_ssim']['after']:.4f} | "
          f"{o['lpips']['before']:.4f} / {o['lpips']['after']:.4f} | "
          f"{o['descriptor_discrepancy_pct']['before']:.2f} / {o['descriptor_discrepancy_pct']['after']:.2f} |")
    if halluc_bands:
        print(f"hallucination pattern in bands: {', '.join(halluc_bands)}")
    print(verdict)


if __name__ == "__main__":
    main()
