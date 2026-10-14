"""Phase 7 — local Gradio app. Upload a micrograph, see original vs reconstruction,
metrics and compression ratio, download the reconstruction and the latent."""
import os, sys, json, tempfile
import numpy as np, torch
from PIL import Image
import gradio as gr

ROOT = os.environ.get("MSAE_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.join(ROOT, "src"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
import losses as L
from eval import load_net
from infer import reconstruct, image_metrics, compression, pick_device

Image.MAX_IMAGE_PIXELS = None
CKPT = os.environ.get("MSAE_CKPT", os.path.join(ROOT, "checkpoints", "full_best.pt"))
DEV = pick_device()
NET, CFG, STEP = load_net(CKPT, DEV)
OUT = os.path.join(ROOT, "outputs"); os.makedirs(OUT, exist_ok=True)
print(f"loaded {CKPT} (step {STEP}) on {DEV}")


def prep(img, multiple=8):
    a = np.asarray(img.convert("L"), dtype=np.uint8)
    H, W = a.shape
    Hc, Wc = (H // multiple) * multiple, (W // multiple) * multiple
    y, x = (H - Hc) // 2, (W - Wc) // 2
    a = a[y:y + Hc, x:x + Wc]
    return torch.from_numpy(a).float().div_(127.5).sub_(1.0)[None, None], a


def run(image, tile_mode):
    if image is None:
        return None, None, "Upload a micrograph first.", None, None
    x, a = prep(image)
    tile = 512 if tile_mode == "tile 512 / overlap 64" else 0
    y, z, used_tile = reconstruct(NET, x, DEV, tile=tile, overlap=64)
    m = image_metrics(y, x.to(DEV), DEV)
    c = compression(x, z)
    rec = ((y[0, 0].cpu().numpy() + 1) * 127.5).clip(0, 255).astype(np.uint8)
    stem = "upload"
    png = os.path.join(OUT, f"{stem}_recon.png")
    npy = os.path.join(OUT, f"{stem}_latent.npy")
    Image.fromarray(rec).save(png)
    np.save(npy, z[0].cpu().numpy().astype(np.float16))
    md = (
        f"| metric | value |\n|---|---|\n"
        f"| PSNR | **{m['psnr_db']:.2f} dB** |\n"
        f"| MS-SSIM | **{m['ms_ssim']:.4f}** |\n"
        f"| LPIPS (AlexNet) | **{m['lpips']:.4f}** |\n"
        f"| input | {a.shape[1]}x{a.shape[0]} (8-bit grey) |\n"
        f"| latent | {c['latent_shape']} fp16 |\n"
        f"| compression ratio | **{c['ratio']:.2f} : 1** |\n"
        f"| {c['original_bytes']:,} B | -> {c['latent_bytes']:,} B |\n"
        f"| mode | {'tiled 512/64' if used_tile else 'whole image'} |\n"
        f"| checkpoint | step {STEP} on {DEV} |\n")
    return Image.fromarray(a), Image.fromarray(rec), md, png, npy


with gr.Blocks(title="Microstructure Encoder/Decoder") as demo:
    gr.Markdown("# Microstructure Encoder / Decoder\n"
                "Upload a micrograph — it is encoded to a compact latent and decoded back. "
                "Fidelity is the deliverable.")
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.Image(type="pil", label="micrograph", image_mode="L")
            mode = gr.Radio(["whole image", "tile 512 / overlap 64"],
                            value="tile 512 / overlap 64", label="inference mode")
            btn = gr.Button("Encode -> Decode", variant="primary")
        with gr.Column(scale=2):
            with gr.Row():
                o_img = gr.Image(label="original", image_mode="L")
                r_img = gr.Image(label="reconstruction", image_mode="L")
            info = gr.Markdown()
            with gr.Row():
                d_png = gr.File(label="download reconstruction (PNG)")
                d_npy = gr.File(label="download latent (.npy)")
    btn.click(run, [inp, mode], [o_img, r_img, info, d_png, d_npy])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=int(os.environ.get("PORT", 7860)),
                share=False, inbrowser=False)
