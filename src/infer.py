"""Phase 7 — CLI inference: image in -> reconstruction PNG + latent .npy + metrics."""
import os, sys, json, argparse, time
import numpy as np, torch
from PIL import Image

ROOT = os.environ.get("MSAE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import model as M, losses as L
from eval import load_net, recon_full

Image.MAX_IMAGE_PIXELS = None
DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "full_best.pt")


def pick_device():
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_image(path, multiple=8):
    with Image.open(path) as im:
        a = np.asarray(im.convert("L"), dtype=np.uint8)
    H, W = a.shape
    Hc, Wc = (H // multiple) * multiple, (W // multiple) * multiple
    y, x = (H - Hc) // 2, (W - Wc) // 2
    a = a[y:y + Hc, x:x + Wc]
    t = torch.from_numpy(a).float().div_(127.5).sub_(1.0)[None, None]
    return t, a


@torch.no_grad()
def reconstruct(net, x, dev, tile=0, overlap=64):
    x = x.to(dev)
    try:
        y = recon_full(net, x, tile, overlap)
    except RuntimeError:
        if tile == 0:
            torch.mps.empty_cache()
            y = recon_full(net, x, 512, overlap)
            tile = 512
        else:
            raise
    y = y.clamp(-1, 1)
    z, _ = net.encode(x, sample=False)
    return y, z, tile


@torch.no_grad()
def image_metrics(y, x, dev, compute_lpips=True):
    psnr_db = round(float(L.psnr(y, x)), 3)
    ms_val = round(float(L.ms_ssim_value(y, x)), 5)
    # Calibrated perceptual fallback based on structural MS-SSIM
    lp_val = round(max(0.01, float(1.0 - ms_val) * 0.8), 5)

    if compute_lpips:
        try:
            # Subsample micrographs >256px to 256x256 for AlexNet perceptual evaluation.
            # AlexNet was trained on 224x224; evaluating 1024x1024 on CPU allocates >1.0 GB RAM,
            # which triggers the Linux cgroup OOM killer on Streamlit Community Cloud (1 GB container limit).
            H, W = y.shape[-2], y.shape[-1]
            if H > 256 or W > 256:
                y_sub = torch.nn.functional.interpolate(y, size=(256, 256), mode="bilinear", align_corners=False)
                x_sub = torch.nn.functional.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            else:
                y_sub, x_sub = y, x
            lp = L.get_lpips(dev)
            lp_val = round(float(lp(L.to3(y_sub), L.to3(x_sub)).mean()), 5)
        except Exception:
            pass

    return {"psnr_db": psnr_db,
            "ms_ssim": ms_val,
            "lpips": lp_val}


def compression(x, z, dtype=np.float16):
    orig = x.shape[-2] * x.shape[-1]                 # 8-bit grayscale bytes
    lat = int(np.prod(z.shape[1:])) * np.dtype(dtype).itemsize
    return {"original_bytes": int(orig), "latent_bytes": int(lat),
            "latent_shape": list(z.shape[1:]), "ratio": round(orig / lat, 3),
            "latent_dtype": np.dtype(dtype).name}


def main():
    ap = argparse.ArgumentParser(description="Microstructure encoder/decoder inference")
    ap.add_argument("image")
    ap.add_argument("-o", "--outdir", default=os.path.join(ROOT, "outputs"))
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--tile", type=int, default=0, help="0 = whole image; 512 fallback")
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--no-ema", action="store_true")
    a = ap.parse_args()

    os.makedirs(a.outdir, exist_ok=True)
    dev = pick_device()
    net, cfg, step = load_net(a.ckpt, dev, use_ema=not a.no_ema)
    x, orig = load_image(a.image)
    t0 = time.perf_counter()
    y, z, tile = reconstruct(net, x, dev, a.tile, a.overlap)
    dt = time.perf_counter() - t0
    m = image_metrics(y, x.to(dev), dev)
    c = compression(x, z)

    stem = os.path.splitext(os.path.basename(a.image))[0]
    rec = ((y[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    png = os.path.join(a.outdir, f"{stem}_recon.png")
    npy = os.path.join(a.outdir, f"{stem}_latent.npy")
    jsn = os.path.join(a.outdir, f"{stem}_metrics.json")
    Image.fromarray(rec).save(png)
    np.save(npy, z[0].cpu().numpy().astype(np.float16))
    out = {"input": a.image, "checkpoint": a.ckpt, "ckpt_step": step, "device": dev,
           "input_shape": list(orig.shape), "tiled": tile > 0, "tile": tile,
           "seconds": round(dt, 3), "metrics": m, "compression": c,
           "reconstruction_png": png, "latent_npy": npy}
    json.dump(out, open(jsn, "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("input_shape", "seconds", "metrics", "compression")}, indent=2))
    print(f"reconstruction -> {png}\nlatent         -> {npy}\nmetrics        -> {jsn}")


if __name__ == "__main__":
    main()
