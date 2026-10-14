"""Validate a checkpoint on the training-time val set with EMA and with raw weights.

Early in training the EMA still carries the random init (decay 0.999: 0.999^2000 = 13.5%
at step 2000), so a step-2000 EMA score understates the model. Phase 5 is judged against
the better of the two so its gate cannot pass on a weakened baseline."""
import os, sys, json, argparse
import torch

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import data as D, losses as L
from eval import load_net
from train import validate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    a = ap.parse_args()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    out, crit = {}, None
    for tag, use_ema in (("ema", True), ("raw", False)):
        net, cfg, step = load_net(a.ckpt, dev, use_ema=use_ema)
        if crit is None:
            crit = L.ReconLoss(cfg["loss"]["w_l1"], cfg["loss"]["w_lpips"], cfg["loss"]["w_msssim"], device=dev)
        bs, nval = cfg["train"]["batch_size"], cfg["data"]["val_batches"]
        loader, _ = D.make_loader("val", bs, patch=cfg["data"]["patch"], num_workers=0,
                                  shuffle=False, length=bs * nval)
        out[tag] = {k: round(float(v), 5) for k, v in validate(net, loader, crit, dev, nval).items()}
        out["step"] = step
        del net
    print("VALJSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
