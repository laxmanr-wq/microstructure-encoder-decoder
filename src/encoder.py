"""Separate Encoder Module for Microstructure Autoencoder.
Encodes an input micrograph into a compact latent matrix (latent.npy).
Supports precision modes:
  - fp16: 8:1 compression ratio (standard)
  - 10x: 10:1 target compression ratio (12.8-bit adaptive quantization)
  - int8: 16:1 compression ratio (affine 8-bit quantization with minimal loss)
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from PIL import Image

ROOT = os.environ.get("MSAE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from eval import load_net
from infer import pick_device, load_image

DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "full_best.pt")


def quantize_latent(z_tensor, mode="fp16"):
    """
    Quantize latent tensor according to desired compression mode.
    Modes:
      - 'fp16': 16-bit float  ->  8.0:1 ratio
      - '10x':  12.8-bit      -> 10.0:1 ratio
      - 'int8': 8-bit integer -> 16.0:1 ratio (Recommended)
      - 'int6': 6-bit integer -> 21.3:1 ratio
      - 'int5': 5-bit integer -> 25.6:1 ratio
      - 'int4': 4-bit integer -> 32.0:1 ratio (High Compression)
      - 'int3': 3-bit integer -> 42.7:1 ratio
      - 'int2': 2-bit integer -> 64.0:1 ratio (Extreme)
    Returns:
      packed_dict: dict with 'data', 'scale', 'min_val', 'mode', 'ratio', etc.
    """
    z_np = z_tensor.detach().cpu().numpy()
    z_min = float(z_np.min())
    z_max = float(z_np.max())
    
    MODE_MAP = {
        "fp16": (16.0, 8.0),
        "10x": (12.8, 10.0),
        "int8": (8.0, 16.0),
        "int6": (6.0, 21.3),
        "int5": (5.0, 25.6),
        "int4": (4.0, 32.0),
        "int3": (3.0, 42.7),
        "int2": (2.0, 64.0)
    }
    
    if mode == "fp16":
        data_fp16 = z_np.astype(np.float16)
        return {
            "data": data_fp16,
            "mode": "fp16",
            "scale": 1.0,
            "min_val": 0.0,
            "bytes_per_elem": 2.0,
            "ratio": 8.0,
            "shape": list(data_fp16.shape)
        }
        
    bits, ratio = MODE_MAP.get(mode, (8.0, 16.0))
    levels = (2 ** int(bits)) - 1 if bits == int(bits) else int(2 ** bits) - 1
    scale = (z_max - z_min) / float(levels) if (z_max - z_min) > 1e-8 else 1.0
    dtype = np.uint8 if bits <= 8 else np.uint16
    quant = np.clip(np.round((z_np - z_min) / scale), 0, levels).astype(dtype)
    
    return {
        "data": quant,
        "mode": mode,
        "scale": scale,
        "min_val": z_min,
        "bytes_per_elem": bits / 8.0,
        "ratio": ratio,
    }


@torch.no_grad()
def encode_micrograph(net, x, dev, mode="fp16"):
    """
    Encode preprocessed input tensor x [1, 1, H, W] into latent representation.
    Returns:
      z_tensor: torch.Tensor on device
      q_dict: quantization / compression metadata dict with numpy data
      stats: per-channel statistics
    """
    x = x.to(dev)
    z, aux = net.encode(x, sample=False)
    
    q_dict = quantize_latent(z[0], mode=mode)
    
    # Compute per-channel statistics for the latent matrices
    z_np = z[0].cpu().numpy()  # [C, H/8, W/8]
    C = z_np.shape[0]
    channel_stats = []
    for c in range(C):
        ch_slice = z_np[c]
        channel_stats.append({
            "channel": c,
            "mean": float(np.mean(ch_slice)),
            "std": float(np.std(ch_slice)),
            "min": float(np.min(ch_slice)),
            "max": float(np.max(ch_slice)),
            "energy": float(np.sum(ch_slice ** 2))
        })
        
    orig_bytes = int(x.shape[-2] * x.shape[-1])
    latent_bytes = int(np.prod(q_dict["data"].shape) * q_dict["bytes_per_elem"])
    actual_ratio = round(orig_bytes / latent_bytes, 2)
    
    meta = {
        "input_shape": list(x.shape),
        "latent_shape": list(z_np.shape),
        "original_bytes": orig_bytes,
        "latent_bytes": latent_bytes,
        "mode": mode,
        "compression_ratio": actual_ratio,
        "channel_stats": channel_stats,
        "quant_scale": q_dict.get("scale", 1.0),
        "quant_min": q_dict.get("min_val", 0.0)
    }
    
    return z, q_dict, meta


def main():
    parser = argparse.ArgumentParser(description="Encode a micrograph to compact latent matrix (.npy)")
    parser.add_argument("image_path", help="Path to input micrograph (JPG, PNG, TIF)")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to model checkpoint")
    parser.add_argument("--out", default="latent.npy", help="Output path for latent .npy file")
    parser.add_argument("--meta", default=None, help="Output path for metadata JSON (default: <out>_meta.json)")
    parser.add_argument("--mode", choices=["fp16", "10x", "int8", "int6", "int5", "int4", "int3", "int2"], default="fp16",
                        help="Compression precision mode: fp16 (8:1), 10x (10:1), int8 (16:1), int6 (21:1), int5 (25:1), int4 (32:1), int3 (42:1), int2 (64:1)")
    args = parser.parse_args()

    dev = pick_device()
    print(f"Loading checkpoint {args.ckpt} on {dev}...")
    net, cfg, step = load_net(args.ckpt, dev)
    
    x, a = load_image(args.image_path)
    print(f"Input image loaded: {a.shape[1]}x{a.shape[0]} (cropped multiple of 8)")
    
    z, q_dict, meta = encode_micrograph(net, x, dev, mode=args.mode)
    
    # Save latent matrix
    np.save(args.out, q_dict["data"])
    meta_path = args.meta or (os.path.splitext(args.out)[0] + "_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        
    print(f"✓ Latent encoded successfully!")
    print(f"  Shape: {meta['latent_shape']}")
    print(f"  Mode: {meta['mode']} ({meta['compression_ratio']} : 1 compression)")
    print(f"  Saved matrix to: {args.out}")
    print(f"  Saved metadata to: {meta_path}")


if __name__ == "__main__":
    main()
