"""Phase 8 — optional PatchGAN sharpening. Off by default (BRIEF).
NLayerDiscriminator + hinge loss + the adaptive generator weight from
taming-transformers (Esser et al.)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def weights_init(m):
    name = m.__class__.__name__
    if "Conv" in name:
        nn.init.normal_(m.weight, 0.0, 0.02)
    elif "BatchNorm" in name:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)


class NLayerDiscriminator(nn.Module):
    """PatchGAN: each output logit judges one ~70 px receptive field (n_layers=3).
    ndf defaults to 32, not 64, to stay inside this machine's memory headroom."""
    def __init__(self, input_nc=1, ndf=32, n_layers=3):
        super().__init__()
        kw, pad = 4, 1
        seq = [nn.Conv2d(input_nc, ndf, kw, 2, pad), nn.LeakyReLU(0.2, True)]
        mult = 1
        for n in range(1, n_layers):
            prev, mult = mult, min(2 ** n, 8)
            seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 2, pad, bias=False),
                    nn.BatchNorm2d(ndf * mult), nn.LeakyReLU(0.2, True)]
        prev, mult = mult, min(2 ** n_layers, 8)
        seq += [nn.Conv2d(ndf * prev, ndf * mult, kw, 1, pad, bias=False),
                nn.BatchNorm2d(ndf * mult), nn.LeakyReLU(0.2, True),
                nn.Conv2d(ndf * mult, 1, kw, 1, pad)]
        self.main = nn.Sequential(*seq)
        self.apply(weights_init)

    def forward(self, x):
        return self.main(x)


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def hinge_g_loss(logits_fake):
    return -logits_fake.mean()


def adaptive_weight(rec_loss, g_loss, last_layer, max_w=1e4):
    """Scale the adversarial gradient to the reconstruction gradient's size at the
    decoder's last layer, so neither term swamps the other."""
    rec_g = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
    adv_g = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
    return torch.clamp(rec_g.norm() / (adv_g.norm() + 1e-4), 0.0, max_w).detach()
