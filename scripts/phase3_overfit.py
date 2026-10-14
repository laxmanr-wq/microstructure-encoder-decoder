"""Phase 3 gate — overfit 8 fixed patches for 500 steps, augmentation OFF.
GATE: PSNR > 35 dB. A failure here means the architecture or loss wiring is broken."""
import os, sys, json, time, argparse
import numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
sys.path.insert(0, os.path.join(ROOT, "src"))
import data as D, model as M, losses as L

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
SEED = 1337


def fixed_patches(n=8, patch=128):
    """8 fixed train patches, deterministic centre crops (augmentation OFF)."""
    ds = D.PatchDataset("train", patch=patch, train=False)
    idx = np.linspace(0, len(ds.recs) - 1, n).astype(int)
    xs = [ds[int(i)][0] for i in idx]
    return torch.stack(xs), [ds.recs[int(i)]["path"] for i in idx]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--diag-steps", type=int, default=1500,
                    help="continue past the gate to record steps-to-35dB")
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--latent", default="kl")
    ap.add_argument("--attn", type=int, default=1)
    a = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    env = json.load(open(f"{ROOT}/reports/phase0_env.json"))

    x, paths = fixed_patches()
    x = x.to(dev)
    net = M.AutoEncoder(latent=a.latent, attn=bool(a.attn),
                        out_act="none", logvar_init=0.0).to(dev)
    nparam = M.count_params(net)
    print(f"device={dev} params={nparam:,} batch={x.shape}")

    crit = L.ReconLoss(device=dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    total_steps = max(a.steps, a.diag_steps)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=a.lr * 0.05)

    hist = []
    gate_psnr, step_35 = None, None
    t0 = time.perf_counter()
    for step in range(1, total_steps + 1):
        net.train()
        y, z, reg, _ = net(x)
        loss, parts = crit(y, x)
        total = loss + reg
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % 25 == 0 or step == 1 or step == a.steps:
            net.eval()
            with torch.no_grad():
                yv, _, _, _ = net(x, sample=False)       # deterministic (mode), as at inference
                p = float(L.psnr(yv.clamp(-1, 1), x))
            if step == a.steps:
                gate_psnr = p
            if step_35 is None and p > 35.0:
                step_35 = step
            hist.append({"step": step, "loss": float(total), "psnr": p,
                         "l1": float(parts["l1"]), "lpips": float(parts["lpips"]),
                         "ms_ssim": float(parts["ms_ssim"])})
            print(f"  step {step:4d} loss {float(total):.4f} PSNR {p:6.2f} dB "
                  f"L1 {float(parts['l1']):.4f} LPIPS {float(parts['lpips']):.4f} "
                  f"MS-SSIM {float(parts['ms_ssim']):.4f}", flush=True)
    dt = time.perf_counter() - t0

    net.eval()
    with torch.no_grad():
        y, _, _, _ = net(x, sample=False)
        y = y.clamp(-1, 1)
        final_psnr = float(L.psnr(y, x))
        final_ms = float(L.ms_ssim_value(y, x))
        final_lp = float(crit.lpips(L.to3(y), L.to3(x)).mean())

    fig, axes = plt.subplots(2, 8, figsize=(20, 5.4))
    for i in range(8):
        for r, img in enumerate((x, y)):
            axes[r, i].imshow(((img[i, 0].detach().cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8),
                              cmap="gray", vmin=0, vmax=255)
            axes[r, i].axis("off")
        axes[0, i].set_title(os.path.basename(paths[i])[:16], fontsize=6)
    axes[0, 0].set_ylabel("original"); axes[1, 0].set_ylabel("recon")
    fig.suptitle(f"Phase 3 overfit — 8 patches, {a.steps} steps, PSNR {final_psnr:.2f} dB "
                 f"(top: original, bottom: reconstruction)")
    fig.tight_layout(); fig.savefig(f"{ROOT}/reports/phase3_overfit.png", dpi=110)

    gate = {"gate_steps": a.steps, "gate_psnr_db": round(gate_psnr, 3),
            "psnr_gt_35": gate_psnr > 35.0, "PASS": gate_psnr > 35.0,
            "steps_to_35db": step_35, "diag_steps": total_steps,
            "psnr_at_diag_steps": round(final_psnr, 3)}
    rep = {"phase": 3, "device": dev, "params": nparam, "latent": a.latent,
           "attn": bool(a.attn), "steps": a.steps, "lr": a.lr, "n_patches": 8,
           "augmentation": "OFF", "final_psnr_db": round(final_psnr, 3), "gate_psnr_db": round(gate_psnr, 3),
           "steps_to_35db": step_35,
           "final_ms_ssim": round(final_ms, 5), "final_lpips": round(final_lp, 5),
           "seconds": round(dt, 1), "it_s": round(a.steps / dt, 3),
           "loss_spec": "L1 + 1.0*LPIPS(alex) + 0.25*(1-MS_SSIM)",
           "history": hist, "patch_paths": paths,
           "phase0_batch": env["chosen_batch"], "gate": gate}
    json.dump(rep, open(f"{ROOT}/reports/phase3_overfit.json", "w"), indent=2)
    print(f"\nGATE @{a.steps} steps: {gate_psnr:.2f} dB")
    print(f"diagnostic @{total_steps} steps: {final_psnr:.2f} dB  MS-SSIM {final_ms:.4f}  "
          f"LPIPS {final_lp:.4f}   steps to cross 35 dB: {step_35}   ({dt:.0f}s)")
    print("GATE:", "PASS" if gate["PASS"] else f"FAIL ({gate_psnr:.2f} dB < 35 dB at {a.steps} steps)")


if __name__ == "__main__":
    main()
