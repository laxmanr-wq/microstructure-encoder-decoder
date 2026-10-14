"""Phase 7 — write the measured Phase 6 (and Phase 8, if present) tables into
README.md between markers, straight from the report JSONs."""
import os, json, re

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
REP = os.path.join(ROOT, "reports")
START, END = "<!-- EVAL:START -->", "<!-- EVAL:END -->"


def band_ranges():
    edges = json.load(open(os.path.join(REP, "phase1_data.json")))["band_edges_px"]
    lo = [None] + edges
    hi = edges + [None]
    return [f"< {h:.1f}" if l is None else (f"> {l:.1f}" if h is None else f"{l:.1f}–{h:.1f}")
            for l, h in zip(lo, hi)]


def eval_table():
    ev = json.load(open(os.path.join(REP, "eval.json")))
    d = ev.get("descriptors", {})
    dpb = d.get("per_band", {})
    rng = band_ranges()
    out = [f"Full test split at native resolution — {ev['n_images']} images, "
           f"checkpoint step {ev['ckpt_step']}, {'tiled 512/64' if ev['tiled'] else 'whole-image'} inference, "
           f"peak {ev['peak_gb']} GB on {ev['device']}.", "",
           "| band | char. length (px) | images | PSNR (dB) | MS-SSIM | LPIPS | S2 err % | lineal-path err % | vol. frac. err % | descriptor disc. % | < 5 % target |",
           "|---|---|---|---|---|---|---|---|---|---|---|"]
    for b in sorted(ev["per_band"]):
        p, q = ev["per_band"][b], dpb.get(b, {})
        if not p:
            continue
        f = lambda k: f"{q[k]:.2f}" if q.get(k) is not None else "—"
        met = ("yes" if q.get("meets_5pct_target") else "**no**") if q else "—"
        out.append(f"| {b} | {rng[int(b)]} | {p['n']} | {p['psnr']:.2f} | {p['ms_ssim']:.4f} | {p['lpips']:.4f} | "
                   f"{f('s2_pct_err')} | {f('lineal_path_pct_err')} | {f('volume_fraction_pct_err')} | "
                   f"{f('descriptor_discrepancy_pct')} | {met} |")
    o, do = ev["overall"], d.get("overall", {})
    g = lambda k: f"{do[k]:.2f}" if do.get(k) is not None else "—"
    met = ("yes" if do.get("meets_5pct_target") else "**no**") if do else "—"
    out.append(f"| **all** | | {o['n']} | {o['psnr']:.2f} | {o['ms_ssim']:.4f} | {o['lpips']:.4f} | "
               f"{g('s2_pct_err')} | {g('lineal_path_pct_err')} | {g('volume_fraction_pct_err')} | "
               f"{g('descriptor_discrepancy_pct')} | {met} |")
    out += ["", f"Compression ratio: **{o['cr_fp16']:.2f} : 1** with the latent stored as fp16 "
                f"({o['cr_fp32']:.2f} : 1 as fp32), against 8-bit greyscale input."]
    if d:
        out += ["", f"Descriptors: {d.get('engine')} on {d.get('crop_px')} px centre crops, "
                    f"{d.get('pairs_per_band')} pairs per band; Otsu threshold from the original applied "
                    f"to both images."]
    p8 = os.path.join(REP, "phase8_gan.json")
    if os.path.exists(p8):
        g8 = json.load(open(p8))
        ov = g8["overall"]
        out += ["", "### Phase 8 — PatchGAN sharpening", "",
                "| | PSNR | MS-SSIM | LPIPS | descriptor disc. % |", "|---|---|---|---|---|",
                f"| before | {ov['psnr']['before']:.2f} | {ov['ms_ssim']['before']:.4f} | {ov['lpips']['before']:.4f} | {ov['descriptor_discrepancy_pct']['before']:.2f} |",
                f"| after | {ov['psnr']['after']:.2f} | {ov['ms_ssim']['after']:.4f} | {ov['lpips']['after']:.4f} | {ov['descriptor_discrepancy_pct']['after']:.2f} |",
                "", f"**{g8['verdict']}**"]
    return "\n".join(out)


def main():
    path = os.path.join(ROOT, "README.md")
    s = open(path).read()
    block = f"{START}\n{eval_table()}\n{END}"
    if START in s and END in s:
        s = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: block, s, flags=re.S)
    else:
        s = s.replace("See `reports/` — every phase writes its measured numbers there.",
                      block + "\n\nEvery phase writes its measured numbers to `reports/`.")
        assert START in s, "README Results anchor not found"
    open(path, "w").write(s)
    print("README.md results updated")


if __name__ == "__main__":
    main()
