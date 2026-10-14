"""Phase 6 stage 2 — microstructure descriptors via MCRpy (separate process: TensorFlow
and torch must not share this 8 GB machine).

Computes VolumeFractions, TwoPointCorrelations (S2) and LinealPath for original vs
reconstruction on a stratified sample, breaks the discrepancy out per scale band, and
merges the result into reports/eval.json. A NumPy/FFT implementation of the same three
descriptors runs alongside as an independent cross-check."""
import os, sys, json, time, argparse
import numpy as np

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# ---------------------------------------------------------------- binarisation
def otsu(a):
    hist = np.bincount(a.ravel(), minlength=256).astype(np.float64)
    p = hist / hist.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    mt = mu[-1]
    den = omega * (1 - omega)
    den[den == 0] = 1e-12
    return int(np.argmax((mt * omega - mu) ** 2 / den))


# ---------------------------------------------------------------- numpy cross-check
def s2_fft(ind):
    f = np.fft.rfft2(ind.astype(np.float64))
    ac = np.fft.fftshift(np.fft.irfft2(f * np.conj(f), s=ind.shape) / ind.size)
    H, W = ind.shape
    yy, xx = np.ogrid[:H, :W]
    r = np.sqrt((yy - H // 2) ** 2 + (xx - W // 2) ** 2).astype(np.int32)
    nmax = min(H, W) // 4
    prof = np.bincount(r.ravel(), weights=ac.ravel(), minlength=r.max() + 1)
    cnt = np.bincount(r.ravel(), minlength=r.max() + 1)
    return prof[:nmax] / np.maximum(cnt[:nmax], 1)


def lineal_path(ind, nmax=64):
    out = np.zeros(nmax)
    for axis in (0, 1):
        a = (ind if axis == 0 else ind.T).astype(bool)
        acc = a
        out[0] += float(acc.mean())
        for r in range(1, nmax):
            if acc.shape[1] <= 1:
                break
            acc = acc[:, :-1] & a[:, r:]
            out[r] += float(acc.mean()) if acc.size else 0.0
    return out / 2.0


def pct_err(o, r):
    o, r = np.asarray(o, dtype=np.float64).ravel(), np.asarray(r, dtype=np.float64).ravel()
    n = min(o.size, r.size)
    return float(np.abs(o[:n] - r[:n]).sum() / max(np.abs(o[:n]).sum(), 1e-12) * 100)


# ---------------------------------------------------------------- mcrpy
def mcrpy_setup(size):
    import mcrpy
    from mcrpy import characterize, Microstructure, CharacterizationSettings
    st = CharacterizationSettings(
        descriptor_types=["VolumeFractions", "TwoPointCorrelations", "LinealPath"],
        use_multiphase=False, logging_level=40)

    def run(ind):
        d = characterize(Microstructure(ind.astype(np.float64), use_multiphase=False), st)
        return {k: np.asarray(v) for k, v in d.items() if k != "settings"}
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=os.path.join(ROOT, "reports", "desc_sample.npz"))
    ap.add_argument("--eval", default=os.path.join(ROOT, "reports", "eval.json"))
    ap.add_argument("--per-band", type=int, default=16, help="pairs per scale band")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--no-mcrpy", action="store_true")
    a = ap.parse_args()

    z = np.load(a.npz)
    bands = sorted({int(k[1]) for k in z.files if k.startswith("b")})
    run = None
    if not a.no_mcrpy:
        t0 = time.time()
        run = mcrpy_setup(a.size)
        print(f"mcrpy ready ({time.time()-t0:.0f}s)", flush=True)

    per_band, t_start = {}, time.time()
    for b in bands:
        O, R = z[f"b{b}_orig"], z[f"b{b}_recon"]
        n = min(a.per_band, len(O))
        k = a.size
        acc = {"vf": [], "s2": [], "lp": [], "vf_np": [], "s2_np": [], "lp_np": []}
        for i in range(n):
            o_full, r_full = O[i], R[i]
            y0 = (o_full.shape[0] - k) // 2
            x0 = (o_full.shape[1] - k) // 2
            o = o_full[y0:y0 + k, x0:x0 + k]
            r = r_full[y0:y0 + k, x0:x0 + k]
            t = otsu(o)
            io = (o > t).astype(np.uint8)
            ir = (r > t).astype(np.uint8)          # SAME threshold -> measures phase drift
            acc["vf_np"].append(abs(ir.mean() - io.mean()) / max(io.mean(), 1e-9) * 100)
            acc["s2_np"].append(pct_err(s2_fft(io), s2_fft(ir)))
            acc["lp_np"].append(pct_err(lineal_path(io), lineal_path(ir)))
            if run is not None:
                do, dr = run(io), run(ir)
                acc["vf"].append(pct_err(do["VolumeFractions"], dr["VolumeFractions"]))
                acc["s2"].append(pct_err(do["TwoPointCorrelations"], dr["TwoPointCorrelations"]))
                acc["lp"].append(pct_err(do["LinealPath"], dr["LinealPath"]))
            if (i + 1) % 4 == 0:
                print(f"   band {b}: {i+1}/{n}  ({time.time()-t_start:.0f}s)", flush=True)
        m = {k2: (round(float(np.mean(v)), 4) if v else None) for k2, v in acc.items()}
        src = "mcrpy" if run is not None else "numpy"
        vf, s2, lp = (m["vf"], m["s2"], m["lp"]) if run is not None else (m["vf_np"], m["s2_np"], m["lp_np"])
        disc = round(float(np.mean([vf, s2, lp])), 3)
        per_band[str(b)] = {"n": n, "source": src,
                            "volume_fraction_pct_err": vf, "s2_pct_err": s2,
                            "lineal_path_pct_err": lp,
                            "descriptor_discrepancy_pct": disc,
                            "meets_5pct_target": bool(disc < 5.0),
                            "numpy_crosscheck": {"volume_fraction_pct_err": m["vf_np"],
                                                 "s2_pct_err": m["s2_np"],
                                                 "lineal_path_pct_err": m["lp_np"]}}
        print(f"  band {b}: VF {vf:.2f}%  S2 {s2:.2f}%  LP {lp:.2f}%  -> {disc:.2f}% "
              f"({'MEETS' if disc < 5 else 'MISSES'} <5% target)", flush=True)

    keys = ("volume_fraction_pct_err", "s2_pct_err", "lineal_path_pct_err",
            "descriptor_discrepancy_pct")
    overall = {k: round(float(np.mean([per_band[b][k] for b in per_band])), 3) for k in keys}
    overall["meets_5pct_target"] = bool(overall["descriptor_discrepancy_pct"] < 5.0)

    rep = json.load(open(a.eval)) if os.path.exists(a.eval) else {}
    rep["descriptors"] = {
        "engine": "MCRpy" if run is not None else "numpy-only",
        "crop_px": a.size, "pairs_per_band": a.per_band,
        "per_band": per_band, "overall": overall,
        "seconds": round(time.time() - t_start, 1),
        "definitions": {
            "binarisation": "Otsu on the original; the SAME threshold is applied to the "
                            "reconstruction, so the comparison measures phase drift",
            "descriptors": "mcrpy VolumeFractions / TwoPointCorrelations / LinealPath",
            "error": "sum|orig - recon| / sum|orig| x 100, over the full descriptor vector"}}
    json.dump(rep, open(a.eval, "w"), indent=2)
    print(f"\noverall descriptor discrepancy {overall['descriptor_discrepancy_pct']:.2f}% "
          f"(target <5%) -> {'MET' if overall['meets_5pct_target'] else 'NOT MET'}")
    print("wrote", a.eval)


if __name__ == "__main__":
    main()
