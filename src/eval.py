"""Phase 6 — evaluation at native resolution on the FULL test split.
Stage 1 (this file, torch only): PSNR / MS-SSIM / LPIPS / compression, per scale_band,
plus eval_grid.png and a descriptor sample dumped for stage 2 (MCRpy, separate process
because TensorFlow and torch must not share this 8 GB machine)."""
import os, sys, json, time, argparse
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.environ.get("MSAE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import data as D, model as M, losses as L


def load_net(ckpt, dev, use_ema=True):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    net = M.build(cfg)
    net.load_state_dict(ck["model"])
    if use_ema and "ema" in ck:
        sd = net.state_dict()
        merged = {k: (ck["ema"][k] if k in ck["ema"] else sd[k]) for k in sd}
        net.load_state_dict(merged)
    return net.to(dev).eval(), cfg, ck.get("step", 0)


@torch.no_grad()
def recon_full(net, x, tile=0, overlap=64):
    """Whole-image reconstruction; tiled fallback if allocation fails."""
    if tile <= 0:
        return net(x, sample=False)[0]
    B, C, H, W = x.shape
    out = torch.zeros_like(x)
    wsum = torch.zeros_like(x)
    step = tile - overlap
    ramp = torch.ones(tile, device=x.device)
    if overlap > 0:
        r = torch.linspace(0, 1, overlap, device=x.device)
        ramp[:overlap] = r; ramp[-overlap:] = r.flip(0)
    win = (ramp[:, None] * ramp[None, :])[None, None]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    if ys[-1] + tile < H: ys.append(H - tile)
    if xs[-1] + tile < W: xs.append(W - tile)
    for y in ys:
        for xx in xs:
            p = x[:, :, y:y + tile, xx:xx + tile]
            r = net(p, sample=False)[0]
            out[:, :, y:y + tile, xx:xx + tile] += r * win
            wsum[:, :, y:y + tile, xx:xx + tile] += win
    return out / wsum.clamp_min(1e-8)


@torch.no_grad()
def latent_shape(net, x):
    z, _ = net.encode(x, sample=False)
    return tuple(z.shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "checkpoints", "full_best.pt"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--desc-per-band", type=int, default=40)
    ap.add_argument("--out", default=os.path.join(ROOT, "reports", "eval.json"))
    ap.add_argument("--desc-npz", default=os.path.join(ROOT, "reports", "desc_sample.npz"))
    ap.add_argument("--grid", default=os.path.join(ROOT, "reports", "eval_grid.png"))
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    net, cfg, step = load_net(a.ckpt, dev)
    lp = L.get_lpips(dev)
    ds = D.FullImageDataset(a.split)
    n = len(ds) if a.limit <= 0 else min(a.limit, len(ds))
    print(f"eval {n} {a.split} images at native resolution on {dev} (ckpt step {step})")

    rows = []
    tile = a.tile
    peak = 0.0
    t0 = time.perf_counter()
    desc_pool = {}
    for i in range(n):
        x, band, path = ds[i]
        x = x.unsqueeze(0).to(dev)
        try:
            y = recon_full(net, x, tile)
        except RuntimeError as e:
            if tile == 0:
                print(f"  allocation failed at full res ({e}); falling back to tile 512/overlap 64")
                tile = 512
                torch.mps.empty_cache()
                y = recon_full(net, x, tile, a.overlap)
            else:
                raise
        y = y.clamp(-1, 1)
        zs = latent_shape(net, x)
        orig_bytes = x.shape[-2] * x.shape[-1]                       # 8-bit grayscale
        lat_fp16 = int(np.prod(zs[1:])) * 2
        m = {"path": path, "band": int(band),
             "h": int(x.shape[-2]), "w": int(x.shape[-1]),
             "psnr": float(L.psnr(y, x)),
             "ms_ssim": float(L.ms_ssim_value(y, x)),
             "lpips": float(lp(L.to3(y), L.to3(x)).mean()),
             "cr_fp16": round(orig_bytes / lat_fp16, 3),
             "cr_fp32": round(orig_bytes / (lat_fp16 * 2), 3),
             "tiled": tile > 0}
        rows.append(m)
        try:
            peak = max(peak, torch.mps.driver_allocated_memory() / 1024 ** 3)
        except Exception:
            pass
        b = int(band)
        if len(desc_pool.get(b, [])) < a.desc_per_band:
            o = ((x[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            r = ((y[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
            k = 512
            oy, ox = (o.shape[0] - k) // 2, (o.shape[1] - k) // 2
            if o.shape[0] >= k and o.shape[1] >= k:
                desc_pool.setdefault(b, []).append((o[oy:oy + k, ox:ox + k],
                                                    r[oy:oy + k, ox:ox + k], path))
        if (i + 1) % 100 == 0:
            el = time.perf_counter() - t0
            print(f"  {i+1}/{n}  {el:.0f}s  {(i+1)/el:.2f} img/s  peak {peak:.2f} GB", flush=True)
    wall = time.perf_counter() - t0

    # per-band + overall table
    def agg(sel):
        if not sel:
            return None
        return {k: round(float(np.mean([s[k] for s in sel])), 4)
                for k in ("psnr", "ms_ssim", "lpips", "cr_fp16", "cr_fp32")} | {"n": len(sel)}
    per_band = {str(b): agg([r for r in rows if r["band"] == b]) for b in range(5)}
    overall = agg(rows)
    print("\nband |    n |  PSNR |  MS-SSIM |  LPIPS |  CR(fp16)")
    for b in range(5):
        v = per_band[str(b)]
        if v:
            print(f"  {b}  | {v['n']:4d} | {v['psnr']:5.2f} | {v['ms_ssim']:8.4f} | "
                  f"{v['lpips']:6.4f} | {v['cr_fp16']:6.2f}")
    print(f" all | {overall['n']:4d} | {overall['psnr']:5.2f} | {overall['ms_ssim']:8.4f} | "
          f"{overall['lpips']:6.4f} | {overall['cr_fp16']:6.2f}")

    # grid: original | reconstruction | error, one row per scale band
    fig, axes = plt.subplots(5, 3, figsize=(10.5, 17))
    for b in range(5):
        pool = desc_pool.get(b)
        if not pool:
            for c in range(3): axes[b, c].axis("off")
            continue
        o, r, p = pool[0]
        err = np.abs(o.astype(np.int16) - r.astype(np.int16)).astype(np.uint8)
        for c, (img, ttl, cm) in enumerate(((o, "original", "gray"), (r, "reconstruction", "gray"),
                                            (err, "|error|", "inferno"))):
            axes[b, c].imshow(img, cmap=cm, vmin=0, vmax=255 if c < 2 else 64)
            axes[b, c].set_title(f"band {b} — {ttl}", fontsize=9)
            axes[b, c].axis("off")
    fig.suptitle(f"Phase 6 — test set, native resolution (512 px centre crops shown)")
    fig.tight_layout(); fig.savefig(a.grid, dpi=100)

    np.savez_compressed(a.desc_npz,
                        **{f"b{b}_orig": np.stack([q[0] for q in v]) for b, v in desc_pool.items()},
                        **{f"b{b}_recon": np.stack([q[1] for q in v]) for b, v in desc_pool.items()})
    rep = {"phase": 6, "ckpt": a.ckpt, "ckpt_step": step, "split": a.split,
           "n_images": n, "device": dev, "tiled": tile > 0, "tile": tile,
           "peak_gb": round(peak, 3), "wall_clock_s": round(wall, 1),
           "img_s": round(n / wall, 3), "latent_shape_example": None,
           "per_band": per_band, "overall": overall, "per_image": rows,
           "descriptor_sample_npz": a.desc_npz,
           "descriptors": "pending — run scripts/descriptors.py (MCRpy)"}
    json.dump(rep, open(a.out, "w"), indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
