"""Phase 8 — fine-tune the Phase 5 autoencoder with a PatchGAN.

The reconstruction loss is unchanged (L1 + LPIPS + 0.25(1 - MS-SSIM)); a hinge
adversarial term is added with the adaptive weight. The discriminator trains alone
for the first --d-warmup steps so the generator never chases a random critic.
Starts from the EMA weights Phase 6 evaluated; resumable like train.py."""
import os, sys, json, time, copy, argparse, signal
import numpy as np, torch

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import data as D, model as M, losses as L, gan as G
from train import EMA, lr_at, validate, pick_device, peak_gb

CK = os.path.join(ROOT, "checkpoints")


def set_requires_grad(module, flag):
    for p in module.parameters():
        p.requires_grad_(flag)


def save(path, net, ema, disc, opt_g, opt_d, step, best, cfg, hist):
    tmp = path + ".tmp"
    torch.save({"model": net.state_dict(), "ema": ema.state_dict(), "disc": disc.state_dict(),
                "opt": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "step": step,
                "best": best, "cfg": cfg, "hist": hist}, tmp)
    os.replace(tmp, path)


def ema_validate(net, ema, loader, crit, dev, n):
    ev = copy.deepcopy(net)
    ema.copy_to(ev)
    vm = validate(ev, loader, crit, dev, n)
    del ev
    return vm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=os.path.join(CK, "full_best.pt"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--d-lr", type=float, default=1e-4)
    ap.add_argument("--disc-weight", type=float, default=0.5)
    ap.add_argument("--d-warmup", type=int, default=500)
    ap.add_argument("--ndf", type=int, default=32)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--report", default=os.path.join(ROOT, "reports", "phase8_train.json"))
    a = ap.parse_args()

    dev = pick_device()
    resuming = bool(a.resume and os.path.exists(a.resume))
    ck = torch.load(a.resume if resuming else a.init, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])

    net = M.build(cfg)
    if resuming:
        net.load_state_dict(ck["model"])
    else:                                   # start from the EMA weights Phase 6 evaluated
        sd, e = ck["model"], ck.get("ema", {})
        net.load_state_dict({k: (e[k] if k in e else sd[k]) for k in sd})
    net = net.to(dev)
    ema = EMA(net, cfg["optim"]["ema_decay"])
    disc = G.NLayerDiscriminator(1, a.ndf, 3).to(dev)
    opt_g = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.5, 0.9), weight_decay=0.0)
    opt_d = torch.optim.AdamW(disc.parameters(), lr=a.d_lr, betas=(0.5, 0.9), weight_decay=0.0)
    step0, best, hist = 0, float("inf"), []
    if resuming:
        ema.load_state_dict({k: v.to(dev) for k, v in ck["ema"].items()})
        disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt"]); opt_d.load_state_dict(ck["opt_d"])
        step0, best, hist = ck["step"], ck["best"], ck.get("hist", [])
        print(f"resumed {a.resume} at step {step0}", flush=True)

    crit = L.ReconLoss(cfg["loss"]["w_l1"], cfg["loss"]["w_lpips"], cfg["loss"]["w_msssim"], device=dev)
    bs = a.batch or cfg["train"]["batch_size"]
    train_loader, _ = D.make_loader("train", bs, patch=cfg["data"]["patch"], num_workers=0,
                                    length=bs * (a.steps + 16), seed=cfg["seed"] + 8000 + step0)
    val_loader, _ = D.make_loader("val", bs, patch=cfg["data"]["patch"], num_workers=0,
                                  shuffle=False, length=bs * cfg["data"]["val_batches"])
    nval = cfg["data"]["val_batches"]
    print(f"device={dev} G={M.count_params(net):,} D={M.count_params(disc):,} batch={bs} "
          f"steps={step0}->{a.steps} d_warmup={a.d_warmup} disc_weight={a.disc_weight}", flush=True)

    val_before = None
    if not resuming:
        val_before = ema_validate(net, ema, val_loader, crit, dev, nval)
        print(f"  VAL before GAN: PSNR {val_before['psnr']:.3f} MS-SSIM {val_before['ms_ssim']:.4f} "
              f"LPIPS {val_before['lpips']:.4f}", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("flag", True))
    last_p, best_p = os.path.join(CK, "gan_last.pt"), os.path.join(CK, "gan_best.pt")
    it = iter(train_loader)
    t0 = t_win = time.perf_counter(); peak = 0.0; step = step0; throttle = []

    while step < a.steps:
        try:
            x, _ = next(it)
        except StopIteration:
            it = iter(train_loader); x, _ = next(it)
        x = x.to(dev)
        lr = lr_at(step, a.steps, a.lr, 100, 0.2)
        for g_ in opt_g.param_groups:
            g_["lr"] = lr

        # generator: reconstruction + adaptively weighted adversarial term
        set_requires_grad(disc, False)
        y, z, reg, _ = net(x)
        rec, parts = crit(y, x)
        adv = torch.zeros((), device=dev)
        w = torch.zeros((), device=dev)
        if step >= a.d_warmup:
            adv = G.hinge_g_loss(disc(y))
            w = G.adaptive_weight(rec, adv, net.decoder.conv_out.weight) * a.disc_weight
        loss_g = rec + reg + w * adv
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), cfg["optim"]["grad_clip"])
        opt_g.step(); ema.update(net)

        # discriminator: hinge on real patches vs detached reconstructions
        set_requires_grad(disc, True)
        loss_d = G.hinge_d_loss(disc(x), disc(y.detach()))
        opt_d.zero_grad(set_to_none=True)
        loss_d.backward()
        opt_d.step()
        step += 1
        peak = max(peak, peak_gb())

        if step % 50 == 0:
            now = time.perf_counter(); its = 50 / (now - t_win); t_win = now
            rec_ = {"step": step, "loss_g": float(loss_g), "rec": float(rec), "adv": float(adv),
                    "adv_weight": float(w), "loss_d": float(loss_d), "l1": float(parts["l1"]),
                    "lpips": float(parts["lpips"]), "ms_ssim": float(parts["ms_ssim"]),
                    "it_s": round(its, 3), "peak_gb": round(peak, 3)}
            hist.append(rec_); throttle.append({"step": step, "it_s": round(its, 3)})
            print(f"step {step:6d}/{a.steps} loss_g {float(loss_g):.4f} rec {float(rec):.4f} "
                  f"adv {float(adv):+.4f} w {float(w):.3f} loss_d {float(loss_d):.4f} "
                  f"LPIPS {float(parts['lpips']):.4f} {its:.2f} it/s peak {peak:.2f}GB", flush=True)

        if step % a.val_every == 0 or step == a.steps:
            vm = ema_validate(net, ema, val_loader, crit, dev, nval)
            vm["step"] = step
            hist.append({"val": vm})
            print(f"  VAL step {step}: PSNR {vm['psnr']:.3f} MS-SSIM {vm['ms_ssim']:.4f} "
                  f"LPIPS {vm['lpips']:.4f}", flush=True)
            if vm["lpips"] < best:
                best = vm["lpips"]
                save(best_p, net, ema, disc, opt_g, opt_d, step, best, cfg, hist)
                print(f"  new best -> {best_p}", flush=True)
            save(last_p, net, ema, disc, opt_g, opt_d, step, best, cfg, hist)
        if stop["flag"]:
            save(last_p, net, ema, disc, opt_g, opt_d, step, best, cfg, hist)
            print(f"stopping early at step {step} (checkpoint saved)", flush=True)
            break

    save(last_p, net, ema, disc, opt_g, opt_d, step, best, cfg, hist)
    wall = time.perf_counter() - t0
    vm = ema_validate(net, ema, val_loader, crit, dev, nval)
    rep = {"phase": 8, "init": a.init, "steps_start": step0, "steps_done": step, "steps_target": a.steps,
           "batch_size": bs, "ndf": a.ndf, "disc_weight": a.disc_weight, "d_warmup": a.d_warmup,
           "lr": a.lr, "d_lr": a.d_lr, "wall_clock_s": round(wall, 1), "peak_gb": round(peak, 3),
           "val_before_gan": val_before, "val_final": vm, "best_val_lpips": best,
           "throttle_profile": throttle, "history": hist,
           "checkpoints": {"best": best_p, "last": last_p}}
    json.dump(rep, open(a.report, "w"), indent=2)
    print(f"done: {step} steps in {wall/60:.1f} min, peak {peak:.2f} GB, val PSNR {vm['psnr']:.3f} "
          f"MS-SSIM {vm['ms_ssim']:.4f} LPIPS {vm['lpips']:.4f}", flush=True)


if __name__ == "__main__":
    main()
