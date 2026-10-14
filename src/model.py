"""Phase 3 — model. BRIEF spec: f=8, 3 downsample stages, base 32 (32/64/128),
KL 4-ch latent (default) or VQ 512x64, attention only at the lowest resolution,
decoder = nearest-neighbour upsample + conv (NEVER transposed conv)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

GROUPS = 8


def norm(c):
    return nn.GroupNorm(GROUPS, c, eps=1e-6, affine=True)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, zero_init=False):
        super().__init__()
        self.n1, self.c1 = norm(cin), nn.Conv2d(cin, cout, 3, 1, 1)
        self.n2, self.c2 = norm(cout), nn.Conv2d(cout, cout, 3, 1, 1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        if zero_init:
            # each block starts as identity: the standard LDM/ADM residual init.
            nn.init.zeros_(self.c2.weight); nn.init.zeros_(self.c2.bias)

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Self-attention, lowest resolution only."""
    def __init__(self, c):
        super().__init__()
        self.n = norm(c)
        self.qkv = nn.Conv2d(c, c * 3, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.c = c

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.n(x)).reshape(B, 3, C, H * W).unbind(1)
        q, k, v = (t.transpose(1, 2).unsqueeze(1) for t in (q, k, v))   # B,1,HW,C
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.squeeze(1).transpose(1, 2).reshape(B, C, H, W)
        return x + self.proj(h)


class Downsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c, 3, 2, 1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    """Nearest-neighbour + conv. No transposed convolution (checkerboard)."""
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, 1, 1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class Encoder(nn.Module):
    def __init__(self, chans=(32, 64, 128), zc=4, n_res=2, attn=True, double_z=True):
        super().__init__()
        self.conv_in = nn.Conv2d(1, chans[0], 3, 1, 1)
        blocks = []
        cprev = chans[0]
        for i, c in enumerate(chans):
            for _ in range(n_res):
                blocks.append(ResBlock(cprev, c)); cprev = c
            blocks.append(Downsample(c))                      # 3 downsamples -> f=8
        self.blocks = nn.Sequential(*blocks)
        cmid = chans[-1]
        mid = [ResBlock(cmid, cmid)]
        if attn:
            mid.append(AttnBlock(cmid))
        mid.append(ResBlock(cmid, cmid))
        self.mid = nn.Sequential(*mid)
        self.norm_out = norm(cmid)
        self.conv_out = nn.Conv2d(cmid, zc * 2 if double_z else zc, 3, 1, 1)

    def forward(self, x):
        h = self.mid(self.blocks(self.conv_in(x)))
        return self.conv_out(F.silu(self.norm_out(h)))


class Decoder(nn.Module):
    def __init__(self, chans=(32, 64, 128), zc=4, n_res=2, attn=True, out_act="none"):
        super().__init__()
        self.out_act = out_act
        cmid = chans[-1]
        self.conv_in = nn.Conv2d(zc, cmid, 3, 1, 1)
        mid = [ResBlock(cmid, cmid)]
        if attn:
            mid.append(AttnBlock(cmid))
        mid.append(ResBlock(cmid, cmid))
        self.mid = nn.Sequential(*mid)
        blocks = []
        cprev = cmid
        for c in reversed(chans):
            blocks.append(Upsample(cprev))
            for _ in range(n_res + 1):
                blocks.append(ResBlock(cprev, c)); cprev = c
        self.blocks = nn.Sequential(*blocks)
        self.norm_out = norm(cprev)
        self.conv_out = nn.Conv2d(cprev, 1, 3, 1, 1)

    def forward(self, z):
        h = self.blocks(self.mid(self.conv_in(z)))
        y = self.conv_out(F.silu(self.norm_out(h)))
        # tanh saturates on the 14.6% of micrograph pixels beyond |x|>0.9 and kills
        # their gradient; unbounded output + clamp at inference converges far faster.
        return torch.tanh(y) if self.out_act == "tanh" else y


class DiagonalGaussian:
    def __init__(self, params):
        self.mean, self.logvar = params.chunk(2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.std = torch.exp(0.5 * self.logvar)

    def sample(self):
        return self.mean + self.std * torch.randn_like(self.mean)

    def mode(self):
        return self.mean

    def kl(self):
        return 0.5 * torch.mean(self.mean.pow(2) + self.logvar.exp() - 1.0 - self.logvar)


class VectorQuantizer(nn.Module):
    """VQ codebook 512 x 64 (config flag)."""
    def __init__(self, n_e=512, e_dim=64, beta=0.25):
        super().__init__()
        self.n_e, self.e_dim, self.beta = n_e, e_dim, beta
        self.embedding = nn.Embedding(n_e, e_dim)
        self.embedding.weight.data.uniform_(-1.0 / n_e, 1.0 / n_e)

    def forward(self, z):
        B, C, H, W = z.shape
        zf = z.permute(0, 2, 3, 1).reshape(-1, C)
        d = (zf.pow(2).sum(1, keepdim=True) - 2 * zf @ self.embedding.weight.t()
             + self.embedding.weight.pow(2).sum(1))
        idx = d.argmin(1)
        zq = self.embedding(idx).view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        loss = F.mse_loss(zq.detach(), z) + self.beta * F.mse_loss(zq, z.detach())
        zq = z + (zq - z).detach()
        return zq, loss, idx.view(B, H, W)


class AutoEncoder(nn.Module):
    def __init__(self, chans=(32, 64, 128), zc=4, n_res=2, attn=True,
                 latent="kl", kl_weight=1e-6, vq_n=512, vq_dim=64,
                 out_act="none", logvar_init=0.0):
        super().__init__()
        self.latent = latent
        self.kl_weight = kl_weight
        zdim = zc if latent == "kl" else vq_dim
        self.encoder = Encoder(chans, zdim, n_res, attn, double_z=(latent == "kl"))
        self.decoder = Decoder(chans, zdim, n_res, attn, out_act=out_act)
        self.zc = zdim
        if latent == "vq":
            self.quant = VectorQuantizer(vq_n, vq_dim)
        else:
            self.quant = None
            # Phase 3 measured logvar_init=0 converging faster than -6 (23.6 vs 21.9 dB
            # at 200 overfit steps): the early latent noise helps, and it collapses on
            # its own by ~step 250.
            with torch.no_grad():
                self.encoder.conv_out.bias[zc:].fill_(logvar_init)

    def encode(self, x, sample=True):
        h = self.encoder(x)
        if self.latent == "kl":
            p = DiagonalGaussian(h)
            z = p.sample() if sample else p.mode()
            return z, {"kl": p.kl(), "mean": p.mean, "logvar": p.logvar}
        zq, vqloss, idx = self.quant(h)
        return zq, {"vq": vqloss, "indices": idx}

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x, sample=True):
        z, aux = self.encode(x, sample=sample)
        y = self.decode(z)
        reg = self.kl_weight * aux["kl"] if self.latent == "kl" else aux["vq"]
        return y, z, reg, aux


def build(cfg):
    m = cfg.get("model", {})
    return AutoEncoder(chans=tuple(m.get("chans", (32, 64, 128))),
                       zc=m.get("z_channels", 4), n_res=m.get("n_res", 2),
                       attn=m.get("attn", True), latent=m.get("latent", "kl"),
                       kl_weight=m.get("kl_weight", 1e-6),
                       vq_n=m.get("vq_n", 512), vq_dim=m.get("vq_dim", 64),
                       out_act=m.get("out_act", "none"),
                       logvar_init=m.get("logvar_init", 0.0))


def count_params(m):
    return sum(p.numel() for p in m.parameters())


if __name__ == "__main__":
    for attn in (True, False):
        m = AutoEncoder(attn=attn)
        x = torch.randn(2, 1, 128, 128)
        y, z, reg, _ = m(x)
        print(f"attn={attn} params={count_params(m):,} x{tuple(x.shape)} -> z{tuple(z.shape)} y{tuple(y.shape)}")
