"""Separate Decoder Module for Microstructure Autoencoder.
Decodes a latent matrix (latent.npy, either original or user-edited) back into a high-fidelity micrograph.
Supports float16, float32, and quantized (uint8/uint16) latent representations.
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
from infer import pick_device, load_image, image_metrics

DEFAULT_CKPT = os.path.join(ROOT, "checkpoints", "full_best.pt")


def dequantize_latent(data_np, meta=None):
    """
    Convert a numpy array (which could be float16, float32, uint8, or uint16)
    into a float32 torch tensor suitable for the decoder.
    """
    if data_np.dtype in (np.float32, np.float64):
        z_float = data_np.astype(np.float32)
    elif data_np.dtype == np.float16:
        z_float = data_np.astype(np.float32)
    elif data_np.dtype in (np.uint8, np.int8, np.uint16, np.int16):
        # Quantized array: retrieve scale and min_val from metadata if available
        scale = 1.0
        min_val = 0.0
        if meta and isinstance(meta, dict):
            scale = meta.get("scale") if "scale" in meta else meta.get("quant_scale", 1.0)
            min_val = meta.get("min_val") if "min_val" in meta else meta.get("quant_min", 0.0)
        else:
            # Fallback heuristic: assume standard range [-4, 4]
            max_int = 255.0 if data_np.dtype in (np.uint8, np.int8) else 4095.0
            scale = 8.0 / max_int
            min_val = -4.0
        z_float = (data_np.astype(np.float32) * scale) + min_val
    else:
        z_float = data_np.astype(np.float32)
        
    return z_float


@torch.no_grad()
def decode_latent(net, z_input, dev, meta=None):
    """
    Decodes a latent matrix (numpy array or torch.Tensor) into an 8-bit grayscale image.
    Args:
      net: AutoEncoder model (or Decoder module)
      z_input: numpy array of shape [C, H, W] or [1, C, H, W], or torch.Tensor
      dev: compute device ('mps' or 'cpu')
      meta: optional metadata dict
    Returns:
      recon_pil: PIL.Image (Grayscale 'L')
      recon_np: uint8 numpy array [H*8, W*8]
      y_tensor: torch.Tensor [1, 1, H*8, W*8] in [-1, 1]
    """
    if isinstance(z_input, np.ndarray):
        z_float = dequantize_latent(z_input, meta)
        z_t = torch.from_numpy(z_float).float()
    else:
        z_t = z_input.float()

    if z_t.ndim == 3:
        z_t = z_t.unsqueeze(0)  # Add batch dim: [1, C, H, W]
        
    z_t = z_t.to(dev)
    
    # Decode through the network
    y = net.decode(z_t).clamp(-1.0, 1.0)
    
    # Scale to [0, 255] uint8
    rec_np = ((y[0, 0].cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    recon_pil = Image.fromarray(rec_np, mode="L")
    
    return recon_pil, rec_np, y


def main():
    parser = argparse.ArgumentParser(description="Decode a latent matrix (.npy) to a reconstructed micrograph")
    parser.add_argument("latent_path", help="Path to latent .npy file")
    parser.add_argument("--ckpt", default=DEFAULT_CKPT, help="Path to model checkpoint")
    parser.add_argument("--out", default="recon.png", help="Output path for reconstructed image (PNG)")
    parser.add_argument("--meta", default=None, help="Optional path to metadata JSON")
    parser.add_argument("--ref", default=None, help="Optional path to original micrograph to compute fidelity metrics")
    args = parser.parse_args()

    dev = pick_device()
    print(f"Loading checkpoint {args.ckpt} on {dev}...")
    net, cfg, step = load_net(args.ckpt, dev)
    
    # Load latent matrix
    z_data = np.load(args.latent_path)
    print(f"Loaded latent matrix: shape={z_data.shape}, dtype={z_data.dtype}")
    
    # Load metadata if present
    meta = None
    meta_path = args.meta or (os.path.splitext(args.latent_path)[0] + "_meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            print(f"Loaded metadata from: {meta_path}")
            
    recon_pil, recon_np, y = decode_latent(net, z_data, dev, meta)
    recon_pil.save(args.out)
    print(f"✓ Reconstructed image saved to: {args.out} ({recon_pil.width}x{recon_pil.height})")
    
    # Compute metrics if reference provided
    if args.ref and os.path.exists(args.ref):
        x_ref, _ = load_image(args.ref)
        m = image_metrics(y, x_ref.to(dev), dev)
        print("Fidelity metrics against reference:")
        print(f"  PSNR:    {m['psnr_db']:.2f} dB")
        print(f"  MS-SSIM: {m['ms_ssim']:.4f}")
        print(f"  LPIPS:   {m['lpips']:.4f}")


if __name__ == "__main__":
    main()
