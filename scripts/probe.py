"""Phase 0 — environment + throughput probe. Measures everything on THIS machine."""
import os, json, time, gc, platform, random
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import torch, torch.nn as nn
from PIL import Image

SEED = 1337
random.seed(SEED); torch.manual_seed(SEED)

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
EXT_DATA = os.path.join(ROOT, "dataset")
SSD_DATA = os.path.expanduser("~/ms_cache/dataset")
OUT = os.path.join(ROOT, "reports", "phase0_env.json")


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class ProbeAE(nn.Module):
    """3-5M param conv autoencoder, f=8, base 32 (32/64/128) — matches BRIEF sizing."""
    def __init__(self, base=32, zc=4):
        super().__init__()
        c1, c2, c3 = base, base * 2, base * 4
        def blk(i, o, stride=1):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, stride, 1), nn.GroupNorm(8, o), nn.SiLU(),
                nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())
        self.enc = nn.Sequential(
            nn.Conv2d(1, c1, 3, 1, 1),
            blk(c1, c1), nn.Conv2d(c1, c1, 3, 2, 1),
            blk(c1, c2), nn.Conv2d(c2, c2, 3, 2, 1),
            blk(c2, c3), nn.Conv2d(c3, c3, 3, 2, 1),
            blk(c3, c3), nn.Conv2d(c3, zc, 3, 1, 1))
        self.dec = nn.Sequential(
            nn.Conv2d(zc, c3, 3, 1, 1), blk(c3, c3),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c3, c3, 3, 1, 1), blk(c3, c2),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c2, c2, 3, 1, 1), blk(c2, c1),
            nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(c1, c1, 3, 1, 1), blk(c1, c1),
            nn.Conv2d(c1, 1, 3, 1, 1), nn.Tanh())

    def forward(self, x):
        return self.dec(self.enc(x))


def mps_reset():
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
        try:
            torch.mps.reset_peak_memory_stats()
        except Exception:
            pass
    gc.collect()


def peak_gb():
    if torch.backends.mps.is_available():
        try:
            return torch.mps.driver_allocated_memory() / 1024 ** 3
        except Exception:
            return float("nan")
    return float("nan")


def bench_batch(device, bs, steps=12, warmup=4, res=128):
    mps_reset()
    model = ProbeAE().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    x = torch.randn(bs, 1, res, res, device=device)
    lossf = nn.L1Loss()
    peak = 0.0
    for i in range(warmup + steps):
        if i == warmup:
            if device == "mps":
                torch.mps.synchronize()
            t0 = time.perf_counter()
        opt.zero_grad(set_to_none=True)
        y = model(x)
        loss = lossf(y, x)
        loss.backward()
        opt.step()
        if device == "mps":
            torch.mps.synchronize()
        peak = max(peak, peak_gb())
    dt = time.perf_counter() - t0
    its = steps / dt
    n_params = sum(p.numel() for p in model.parameters())
    del model, opt, x, y, loss
    mps_reset()
    return {"batch": bs, "it_s": round(its, 3), "img_s": round(its * bs, 2),
            "peak_gb": round(peak, 3), "params": n_params}


def list_images(root):
    out = []
    for dp, dn, fn in os.walk(root):
        for f in fn:
            if f.startswith("._") or f == ".DS_Store":
                continue
            out.append(os.path.join(dp, f))
    return sorted(out)


def read_throughput(files, n=120, tag=""):
    sel = files[:: max(1, len(files) // n)][:n]
    t0 = time.perf_counter()
    px = 0
    for p in sel:
        with Image.open(p) as im:
            im = im.convert("L")
            px += im.size[0] * im.size[1]
    dt = time.perf_counter() - t0
    return {"source": tag, "n_images": len(sel), "seconds": round(dt, 3),
            "img_s": round(len(sel) / dt, 2), "mpix_s": round(px / dt / 1e6, 2)}


def main():
    dev = pick_device()
    print(f"device={dev} torch={torch.__version__}")
    res = {
        "phase": 0,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": dev,
        "memory_budget_gb": 4.0,
        "seed": SEED,
    }

    benches = []
    for bs in (8, 16, 32):
        b = bench_batch(dev, bs)
        print(f"  batch {bs:>2}: {b['it_s']:.2f} it/s  {b['img_s']:.1f} img/s  peak {b['peak_gb']:.2f} GB")
        benches.append(b)
    res["benchmarks"] = benches
    res["probe_params"] = benches[0]["params"]

    under = [b for b in benches if b["peak_gb"] < 4.0]
    largest = max([b["batch"] for b in under], default=8)
    # choose the largest batch under budget that also maximises img/s
    best = max(under, key=lambda b: b["img_s"]) if under else benches[0]
    res["largest_batch_under_4gb"] = largest
    res["chosen_batch"] = best["batch"]
    res["chosen_it_s"] = best["it_s"]
    res["chosen_img_s"] = best["img_s"]
    res["chosen_peak_gb"] = best["peak_gb"]

    ext = list_images(EXT_DATA)
    ssd = list_images(SSD_DATA) if os.path.isdir(SSD_DATA) else []
    res["n_images_external"] = len(ext)
    res["n_images_ssd_cache"] = len(ssd)
    res["ssd_cache_path"] = SSD_DATA

    io = []
    if ssd:
        io.append(read_throughput(ssd, tag="ssd_cache"))
    io.append(read_throughput(ext, tag="external"))
    res["io"] = io
    s = next((r["img_s"] for r in io if r["source"] == "ssd_cache"), None)
    e = next((r["img_s"] for r in io if r["source"] == "external"), None)
    res["cache_speedup"] = round(s / e, 2) if (s and e) else None
    for r in io:
        print(f"  read {r['source']:>10}: {r['img_s']:.1f} img/s")

    gate = {
        "mps_available": res["mps_available"],
        "batch_under_4gb": bool(under),
        "ssd_faster_than_external": bool(s and e and s > e),
        "image_count_10569": len(ext) == 10569,
    }
    gate["PASS"] = all(gate.values())
    res["gate"] = gate

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps(gate, indent=2))
    print("GATE:", "PASS" if gate["PASS"] else "FAIL")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
