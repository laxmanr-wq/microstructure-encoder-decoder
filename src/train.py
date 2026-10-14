"""Phase 4/5 — training. AdamW, fp32, EMA, cosine+warmup, best+last checkpoints only,
clean resume, val every N steps, wall-clock it/s every 50 steps (thermal throttling)."""
import os, sys, json, time, math, copy, argparse, signal
import numpy as np, torch, yaml

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import data as D, model as M, losses as L


def pick_device(want="mps"):
    if want == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def peak_gb():
    try:
        return torch.mps.driver_allocated_memory() / 1024 ** 3
    except Exception:
        return float("nan")


class EMA:
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(self.decay).add_(v.detach().float(), alpha=1 - self.decay)

    def copy_to(self, model):
        sd = model.state_dict()
        merged = {k: (self.shadow[k] if k in self.shadow else sd[k]) for k in sd}
        model.load_state_dict(merged)

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, sd):
        self.shadow = {k: v.clone() for k, v in sd.items()}


def lr_at(step, total, base, warmup, min_frac):
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    t = min(max(t, 0.0), 1.0)
    return base * (min_frac + (1 - min_frac) * 0.5 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def validate(net, loader, crit, dev, max_batches=24):
    net.eval()
    acc = {"psnr": 0.0, "ms_ssim": 0.0, "lpips": 0.0, "l1": 0.0}
    n = 0
    for i, (x, _) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(dev)
        y, _, _, _ = net(x, sample=False)
        acc["psnr"] += float(L.psnr(y, x))
        acc["ms_ssim"] += float(L.ms_ssim_value(y, x))
        acc["lpips"] += float(crit.lpips(L.to3(y), L.to3(x)).mean())
        acc["l1"] += float(torch.nn.functional.l1_loss(y, x))
        n += 1
    net.train()
    return {k: v / max(n, 1) for k, v in acc.items()}


def save_ckpt(path, net, ema, opt, step, best, cfg, hist):
    tmp = path + ".tmp"
    torch.save({"model": net.state_dict(), "ema": ema.state_dict(),
                "opt": opt.state_dict(), "step": step, "best": best,
                "cfg": cfg, "hist": hist}, tmp)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "base.yaml"))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--tag", default="calib")
    ap.add_argument("--report", default=None)
    ap.add_argument("--max-hours", type=float, default=None)
    ap.add_argument("--ckpt-every", type=int, default=None)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    if a.steps:
        cfg["train"]["steps"] = a.steps
    if a.ckpt_every:
        cfg["train"]["ckpt_every"] = a.ckpt_every
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    dev = pick_device(cfg.get("device", "mps"))
    out_dir = cfg["train"]["out_dir"]; os.makedirs(out_dir, exist_ok=True)
    bs = cfg["train"]["batch_size"]
    total = cfg["train"]["steps"]

    net = M.build(cfg).to(dev)
    crit = L.ReconLoss(cfg["loss"]["w_l1"], cfg["loss"]["w_lpips"],
                       cfg["loss"]["w_msssim"], device=dev)
    o = cfg["optim"]
    opt = torch.optim.AdamW(net.parameters(), lr=o["lr"], betas=tuple(o["betas"]),
                            weight_decay=o["weight_decay"])
    ema = EMA(net, o["ema_decay"])

    step0, best, hist = 0, float("inf"), []
    if a.resume and os.path.exists(a.resume):
        ck = torch.load(a.resume, map_location=dev, weights_only=False)
        net.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"]); step0 = ck["step"]; best = ck["best"]
        hist = ck.get("hist", [])
        print(f"resumed {a.resume} at step {step0} (best val lpips {best:.4f})")

    train_loader, train_ds = D.make_loader("train", bs, patch=cfg["data"]["patch"],
                                           num_workers=cfg["data"]["num_workers"],
                                           length=bs * (total + 16),
                                           # a restart must not replay the batches it already saw
                                           seed=cfg["seed"] + step0)
    val_loader, _ = D.make_loader("val", bs, patch=cfg["data"]["patch"],
                                  num_workers=0, shuffle=False,   # each spawn worker re-imports torch (~400 MB)
                                  length=bs * cfg["data"]["val_batches"])
    print(f"device={dev} params={M.count_params(net):,} batch={bs} steps={step0}->{total} "
          f"train_imgs={len(train_ds.recs)}")

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))

    it = iter(train_loader)
    t_start = time.perf_counter()
    t_win = t_start
    peak, throttle = 0.0, []
    step = step0
    last_path = os.path.join(out_dir, f"{a.tag}_last.pt")
    best_path = os.path.join(out_dir, f"{a.tag}_best.pt")
    deadline = t_start + a.max_hours * 3600 if a.max_hours else None

    net.train()
    while step < total:
        try:
            x, _ = next(it)
        except StopIteration:
            train_ds.set_epoch(train_ds.epoch + 1)
            it = iter(train_loader); x, _ = next(it)
        x = x.to(dev)
        lr = lr_at(step, total, o["lr"], o["warmup_steps"], o["min_lr_frac"])
        for g in opt.param_groups:
            g["lr"] = lr
        y, z, reg, _ = net(x)
        loss, parts = crit(y, x)
        tot = loss + reg
        opt.zero_grad(set_to_none=True)
        tot.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), o["grad_clip"])
        opt.step(); ema.update(net)
        step += 1
        peak = max(peak, peak_gb())

        if step % cfg["train"]["log_every"] == 0:
            now = time.perf_counter()
            its = cfg["train"]["log_every"] / (now - t_win); t_win = now
            rec = {"step": step, "loss": float(tot), "l1": float(parts["l1"]),
                   "lpips": float(parts["lpips"]), "ms_ssim": float(parts["ms_ssim"]),
                   "lr": lr, "it_s": round(its, 3), "peak_gb": round(peak, 3),
                   "elapsed_s": round(now - t_start, 1)}
            throttle.append({"step": step, "it_s": round(its, 3),
                             "elapsed_s": round(now - t_start, 1)})
            hist.append(rec)
            print(f"step {step:6d}/{total} loss {float(tot):.4f} L1 {float(parts['l1']):.4f} "
                  f"LPIPS {float(parts['lpips']):.4f} MS-SSIM {float(parts['ms_ssim']):.4f} "
                  f"lr {lr:.2e} {its:.2f} it/s peak {peak:.2f}GB", flush=True)

        if step % cfg["train"]["val_every"] == 0 or step == total:
            ev = copy.deepcopy(net)
            ema.copy_to(ev)
            vm = validate(ev, val_loader, crit, dev, cfg["data"]["val_batches"])
            del ev
            vm["step"] = step
            hist.append({"val": vm})
            print(f"  VAL step {step}: PSNR {vm['psnr']:.3f} MS-SSIM {vm['ms_ssim']:.4f} "
                  f"LPIPS {vm['lpips']:.4f}", flush=True)
            if vm["lpips"] < best:
                best = vm["lpips"]
                save_ckpt(best_path, net, ema, opt, step, best, cfg, hist)
                print(f"  new best -> {best_path}", flush=True)

        if step % cfg["train"]["ckpt_every"] == 0 or step == total:
            save_ckpt(last_path, net, ema, opt, step, best, cfg, hist)
        if stop["flag"] or (deadline and time.perf_counter() > deadline):
            save_ckpt(last_path, net, ema, opt, step, best, cfg, hist)
            print(f"stopping early at step {step} (checkpoint saved)")
            break

    save_ckpt(last_path, net, ema, opt, step, best, cfg, hist)
    wall = time.perf_counter() - t_start
    ev = copy.deepcopy(net); ema.copy_to(ev)
    vm = validate(ev, val_loader, crit, dev, cfg["data"]["val_batches"]); del ev

    its_all = [h["it_s"] for h in throttle]
    rep = {"phase": 4 if a.tag == "calib" else 5, "tag": a.tag, "device": dev,
           "params": M.count_params(net), "batch_size": bs,
           "steps_start": step0, "steps_done": step, "steps_target": total,
           "wall_clock_s": round(wall, 1), "wall_clock_h": round(wall / 3600, 3),
           "peak_gb": round(peak, 3),
           "it_s_mean": round(float(np.mean(its_all)), 3) if its_all else None,
           "it_s_first10": round(float(np.mean(its_all[:10])), 3) if len(its_all) >= 10 else None,
           "it_s_last10": round(float(np.mean(its_all[-10:])), 3) if len(its_all) >= 10 else None,
           "throttle_profile": throttle, "val_final": vm, "best_val_lpips": best,
           "history": hist, "config": cfg,
           "checkpoints": {"best": best_path, "last": last_path}}
    if its_all:
        f = rep["it_s_first10"] or its_all[0]
        l = rep["it_s_last10"] or its_all[-1]
        rep["throttle_pct"] = round(100 * (1 - l / f), 1)
        # budgets from MEASURED throttled throughput, not the fresh-start rate
        sustained = min(l, float(np.median(its_all)))
        rep["sustained_it_s"] = round(sustained, 3)
        rep["recommended_budget"] = {
            "basis": "sustained it/s = min(mean of last 10 windows, median of all windows)",
            "sustained_it_s": round(sustained, 3),
            "steps_5h": int(sustained * 5 * 3600 * 0.95),      # 5% margin for val/ckpt
            "steps_10h": int(sustained * 10 * 3600 * 0.95),
            "val_ckpt_overhead_margin": 0.95}
    losses_seq = [h["loss"] for h in hist if "loss" in h]
    if len(losses_seq) >= 4:
        q = max(2, len(losses_seq) // 4)
        first_q, last_q = float(np.mean(losses_seq[:q])), float(np.mean(losses_seq[-q:]))
        rep["loss_first_quarter"] = round(first_q, 5)
        rep["loss_last_quarter"] = round(last_q, 5)
        rep["gate"] = {"loss_decreasing": last_q < first_q,
                       "peak_under_4gb": peak < 4.0}
        rep["gate"]["PASS"] = all(v for k, v in rep["gate"].items() if k != "PASS")
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=2)
        print("wrote", a.report)
    print(f"done: {step} steps in {wall/60:.1f} min, peak {peak:.2f} GB, "
          f"val PSNR {vm['psnr']:.3f} MS-SSIM {vm['ms_ssim']:.4f} LPIPS {vm['lpips']:.4f}")
    if "recommended_budget" in rep:
        rb = rep["recommended_budget"]
        print(f"sustained {rb['sustained_it_s']:.3f} it/s (throttle {rep['throttle_pct']}%)  "
              f"-> 5h budget {rb['steps_5h']} steps, 10h budget {rb['steps_10h']} steps")
    if "gate" in rep:
        print("GATE:", "PASS" if rep["gate"]["PASS"] else "FAIL", rep["gate"])


if __name__ == "__main__":
    main()
