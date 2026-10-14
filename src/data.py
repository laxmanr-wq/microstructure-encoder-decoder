"""Phase 2 — dataloader. Uses manifest.csv `split` column AS-IS. Never re-splits."""
import os, csv, random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
MANIFEST = os.path.join(ROOT, "manifests", "manifest.csv")
SEED = 1337
PATCH = 128


def load_manifest(manifest=MANIFEST, split=None, keep_only=True):
    rows = []
    with open(manifest) as f:
        for r in csv.DictReader(f):
            if keep_only and int(r["keep"]) != 1:
                continue
            if split is not None and r["split"] != split:
                continue
            rows.append({
                "path": r["path"], "split": r["split"],
                "crop_top": int(r["crop_top"]), "crop_bottom": int(r["crop_bottom"]),
                "scale_band": int(r["scale_band"]), "cluster": int(r["cluster"]),
                "w": int(r["w"]), "h": int(r["h"]),
            })
    return rows


def dihedral(a, k):
    """8-fold dihedral: micrographs are rotation-invariant."""
    if k >= 4:
        a = np.fliplr(a)
    return np.rot90(a, k % 4)


def body_array(rec):
    """Read image, drop the databar strips. Never touches the file on disk."""
    with Image.open(rec["path"]) as im:
        a = np.asarray(im.convert("L"), dtype=np.uint8)
    h = a.shape[0]
    return a[rec["crop_top"]: h - rec["crop_bottom"]]


def to_tensor(patch):
    x = torch.from_numpy(np.ascontiguousarray(patch)).float().div_(127.5).sub_(1.0)
    return x.unsqueeze(0)                                # 1xHxW in [-1,1]


class PatchDataset(Dataset):
    """train: random crop + 8-fold dihedral. val/test: deterministic centre crop."""

    def __init__(self, split, patch=PATCH, train=None, manifest=MANIFEST,
                 length=None, keep_only=True):
        self.recs = load_manifest(manifest, split, keep_only)
        if not self.recs:
            raise RuntimeError(f"no rows for split={split!r}")
        self.patch = patch
        self.train = (split == "train") if train is None else train
        self.split = split
        self.length = length or len(self.recs)
        self.epoch = 0

    def __len__(self):
        return self.length

    def set_epoch(self, e):
        self.epoch = int(e)

    def _rng(self, idx):
        return random.Random((SEED * 1000003) ^ (self.epoch * 7919) ^ (idx * 104729))

    def __getitem__(self, idx):
        rec = self.recs[idx % len(self.recs)]
        a = body_array(rec)
        H, W = a.shape
        p = self.patch
        if H < p or W < p:
            pad = np.full((max(H, p), max(W, p)), int(a.mean()), dtype=np.uint8)
            pad[:H, :W] = a
            a = pad
            H, W = a.shape
        if self.train:
            rng = self._rng(idx)
            y = rng.randrange(0, H - p + 1)
            x = rng.randrange(0, W - p + 1)
            patch = a[y:y + p, x:x + p]
            patch = dihedral(patch, rng.randrange(8))
        else:
            y = (H - p) // 2
            x = (W - p) // 2
            patch = a[y:y + p, x:x + p]
        return to_tensor(patch), rec["scale_band"]


class FullImageDataset(Dataset):
    """Native-resolution body images for evaluation (Phase 6)."""

    def __init__(self, split, manifest=MANIFEST, keep_only=True, multiple=8):
        self.recs = load_manifest(manifest, split, keep_only)
        self.multiple = multiple

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        rec = self.recs[i]
        a = body_array(rec)
        m = self.multiple
        H, W = a.shape
        Hc, Wc = (H // m) * m, (W // m) * m
        y, x = (H - Hc) // 2, (W - Wc) // 2
        a = a[y:y + Hc, x:x + Wc]
        return to_tensor(a), rec["scale_band"], rec["path"]


def make_loader(split, batch_size, patch=PATCH, num_workers=4, length=None,
                shuffle=None, manifest=MANIFEST, seed=SEED):
    ds = PatchDataset(split, patch=patch, manifest=manifest, length=length)
    if shuffle is None:
        shuffle = (split == "train")
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=False, drop_last=(split == "train"),
                      persistent_workers=num_workers > 0, prefetch_factor=2 if num_workers else None,
                      generator=g), ds
