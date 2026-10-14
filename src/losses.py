"""Phase 3 — losses. BRIEF spec: L1 + 1.0*LPIPS(alex) + 0.25*(1 - MS_SSIM).
MSE-only is FORBIDDEN. LPIPS must be AlexNet. Grayscale replicated to 3ch for LPIPS."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim, ssim

_LPIPS = None


def get_lpips(device):
    global _LPIPS
    if _LPIPS is None:
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
            import lpips as _l
            net = _l.LPIPS(net="alex", verbose=False)
            net.eval()
            for p in net.parameters():
                p.requires_grad_(False)
            _LPIPS = net
    return _LPIPS.to(device)


def to3(x):
    return x.repeat(1, 3, 1, 1) if x.shape[1] == 1 else x


def msssim_win(size):
    """pytorch_msssim always uses 4 downsamplings, so the window must satisfy
    (win-1)*16 < size. 11 for >=161 px (standard), 7 for the 128 px patches."""
    return 11 if size > 160 else 7


def ms_ssim_value(a, b, data_range=2.0):
    """a,b in [-1,1]. Window adapts to input size; win=7 for 128 px patches."""
    size = min(a.shape[-2], a.shape[-1])
    win = msssim_win(size)
    if (win - 1) * 16 >= size:
        return ssim(a, b, data_range=data_range, size_average=True, win_size=min(win, 7))
    return ms_ssim(a, b, data_range=data_range, size_average=True, win_size=win)


class ReconLoss(nn.Module):
    def __init__(self, w_l1=1.0, w_lpips=1.0, w_msssim=0.25, device="cpu"):
        super().__init__()
        self.w_l1, self.w_lpips, self.w_msssim = w_l1, w_lpips, w_msssim
        self.lpips = get_lpips(device) if w_lpips > 0 else None

    def forward(self, y, x):
        l1 = F.l1_loss(y, x)
        out = {"l1": l1.detach()}
        total = self.w_l1 * l1
        if self.lpips is not None:
            lp = self.lpips(to3(y), to3(x)).mean()          # inputs already in [-1,1]
            total = total + self.w_lpips * lp
            out["lpips"] = lp.detach()
        if self.w_msssim > 0:
            ms = ms_ssim_value(y, x)
            total = total + self.w_msssim * (1.0 - ms)
            out["ms_ssim"] = ms.detach()
        out["recon"] = total.detach()
        return total, out


@torch.no_grad()
def psnr(y, x, data_range=2.0):
    """y,x in [-1,1]; data_range=2 covers the full [-1,1] span."""
    mse = F.mse_loss(y, x, reduction="none").flatten(1).mean(1)
    return (10.0 * torch.log10(data_range ** 2 / mse.clamp_min(1e-12))).mean()


@torch.no_grad()
def metrics(y, x, lpips_net=None):
    m = {"psnr": float(psnr(y, x)), "ms_ssim": float(ms_ssim_value(y, x))}
    if lpips_net is not None:
        m["lpips"] = float(lpips_net(to3(y), to3(x)).mean())
    return m


if __name__ == "__main__":
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    x = torch.rand(2, 1, 128, 128, device=dev) * 2 - 1
    y = (x + 0.05 * torch.randn_like(x)).clamp(-1, 1)
    L = ReconLoss(device=dev)
    t, parts = L(y, x)
    print("win@128:", msssim_win(128), "win@1024:", msssim_win(1024))
    print("loss", float(t), {k: round(float(v), 4) for k, v in parts.items()})
    print("psnr", float(psnr(y, x)))
