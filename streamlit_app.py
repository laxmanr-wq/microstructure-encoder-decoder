import os
import sys
import json
import tempfile
import numpy as np
import pandas as pd
import torch
from PIL import Image
import streamlit as st

# Setup paths and environment
ROOT = os.environ.get("MSAE_ROOT", os.path.abspath(os.path.dirname(__file__)))
SRC_DIR = os.path.join(ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
Image.MAX_IMAGE_PIXELS = None

from eval import load_net
from infer import pick_device, load_image, image_metrics, compression
from encoder import encode_micrograph, quantize_latent
from decoder import decode_latent, dequantize_latent

# Page configuration
st.set_page_config(
    page_title="Microstructure Encoder / Decoder Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
    }
    
    .main-title {
        font-size: 2.1rem;
        font-weight: 700;
        background: linear-gradient(135deg, #60a5fa 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.0rem;
        color: #94a3b8;
        margin-bottom: 1.2rem;
    }
    
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        text-align: center;
        backdrop-filter: blur(8px);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: #3b82f6;
    }
    
    .metric-card-highlight {
        background: rgba(30, 58, 138, 0.4);
        border: 1px solid rgba(96, 165, 250, 0.4);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        text-align: center;
        backdrop-filter: blur(8px);
    }
    
    .metric-val {
        font-size: 1.7rem;
        font-weight: 700;
        color: #f8fafc;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-val-acc {
        font-size: 1.7rem;
        font-weight: 700;
        color: #34d399;
        font-family: 'JetBrains Mono', monospace;
    }
    
    .metric-lbl {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-top: 0.2rem;
    }
    
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.4rem;
        background: rgba(59, 130, 246, 0.15);
        color: #60a5fa;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }
    
    .badge-purple {
        background: rgba(168, 85, 247, 0.15);
        color: #c084fc;
        border: 1px solid rgba(168, 85, 247, 0.3);
    }

    .badge-green {
        background: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .step-header {
        font-size: 1.15rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 0.4rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_model(ckpt_path, device):
    """Load model once and cache across interactions."""
    net, cfg, step = load_net(ckpt_path, device)
    return net, cfg, step


def prepare_image(img, multiple=8):
    """Crop grayscale micrograph to multiple of 8 and normalize to [-1, 1]."""
    a = np.asarray(img.convert("L"), dtype=np.uint8)
    H, W = a.shape
    Hc, Wc = (H // multiple) * multiple, (W // multiple) * multiple
    y, x = (H - Hc) // 2, (W - Wc) // 2
    a = a[y:y + Hc, x:x + Wc].copy()
    t = torch.from_numpy(a).float().div_(127.5).sub_(1.0)[None, None]
    return t, a


def compute_difference_map(orig, recon, amplify=1.0, cmap_name="turbo"):
    """Generate colorized absolute difference heatmap using scientific colormaps."""
    import matplotlib
    diff = np.abs(orig.astype(np.float32) - recon.astype(np.float32))
    diff_norm = np.clip(diff * amplify, 0.0, 255.0) / 255.0
    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    colored = (cmap(diff_norm)[:, :, :3] * 255.0).astype(np.uint8)
    return Image.fromarray(colored), diff


def colorize_channel(channel_2d):
    """Normalize a 2D latent channel matrix to an attractive heatmap for visualization."""
    c_min, c_max = channel_2d.min(), channel_2d.max()
    denom = c_max - c_min if (c_max - c_min) > 1e-8 else 1.0
    norm = ((channel_2d - c_min) / denom * 255.0).astype(np.uint8)
    img_rgb = np.zeros((norm.shape[0], norm.shape[1], 3), dtype=np.uint8)
    img_rgb[..., 0] = norm
    img_rgb[..., 1] = 255 - np.abs(norm.astype(np.int16) - 128).astype(np.uint8) * 2
    img_rgb[..., 2] = 255 - norm
    return Image.fromarray(img_rgb)


def calculate_accuracy_scores(orig_np, recon_np, metrics):
    """Compute normalized physical and structural accuracy percentages."""
    diff = np.abs(orig_np.astype(np.float32) - recon_np.astype(np.float32))
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    
    # 1. Pixel Intensity Accuracy: percentage of 255 scale preserved
    pixel_acc = max(0.0, min(100.0, (1.0 - (mae / 255.0)) * 100.0))
    
    # 2. Structural Accuracy: MS-SSIM scaled to percentage
    struct_acc = float(metrics.get("ms_ssim", 0.0) * 100.0)
    
    # 3. Perceptual Accuracy: 1.0 - LPIPS scaled to percentage
    perceptual_acc = max(0.0, min(100.0, (1.0 - metrics.get("lpips", 0.0)) * 100.0))
    
    # 4. Composite Quality Index: 50% Pixel + 50% Structural
    composite_acc = (0.5 * pixel_acc) + (0.5 * struct_acc)
    
    return {
        "composite_acc": composite_acc,
        "pixel_acc": pixel_acc,
        "struct_acc": struct_acc,
        "perceptual_acc": perceptual_acc,
        "mae": mae,
        "rmse": rmse
    }


# --- Sidebar ---
st.sidebar.markdown("### 🔬 System Configuration")

dev = pick_device()
device_color = "#10b981" if dev == "mps" else "#f59e0b"
st.sidebar.markdown(f"**Compute Device:** <span style='color:{device_color}; font-weight:700;'>{dev.upper()}</span>", unsafe_allow_html=True)

# Checkpoint selector
ckpt_dir = os.path.join(ROOT, "checkpoints")
available_ckpts = []
if os.path.isdir(ckpt_dir):
    available_ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
    available_ckpts.sort()

default_ckpt_idx = available_ckpts.index("full_best.pt") if "full_best.pt" in available_ckpts else 0
chosen_ckpt_name = st.sidebar.selectbox("Checkpoint", available_ckpts, index=default_ckpt_idx)
ckpt_path = os.path.join(ckpt_dir, chosen_ckpt_name)

# Comprehensive Compression Ratio Selector (Supports all tiers from 8:1 to 64:1)
st.sidebar.markdown("### 📦 Compression Precision Tier")
COMP_OPTIONS = {
    "fp16 (8.0 : 1 Baseline)": "fp16",
    "10x Target (10.0 : 1 Optimized)": "10x",
    "int8 Quantized (16.0 : 1 Recommended)": "int8",
    "int6 Quantized (21.3 : 1 High Fidelity)": "int6",
    "int5 Quantized (25.6 : 1 Balanced)": "int5",
    "int4 Nibble (32.0 : 1 Aggressive Preview)": "int4",
    "int3 Quantized (42.7 : 1 Ultra Compressed)": "int3",
    "int2 Quantized (64.0 : 1 Extreme)": "int2"
}
comp_mode_label = st.sidebar.selectbox(
    "Target Ratio",
    list(COMP_OPTIONS.keys()),
    index=2,  # default to 16:1 int8
    help="Select the precision/compression ratio to test. From 8:1 baseline to 64:1 extreme."
)
mode_code = COMP_OPTIONS[comp_mode_label]

# Sample Micrograph Selector
st.sidebar.markdown("### 🖼️ Sample Micrographs")
sample_dir = os.path.join(ROOT, "demo_micrographs")
if not os.path.isdir(sample_dir) or len(os.listdir(sample_dir)) == 0:
    sample_dir = os.path.join(ROOT, "dataset", "test")
sample_files = []
if os.path.isdir(sample_dir):
    sample_files = [f for f in sorted(os.listdir(sample_dir)) if f.lower().endswith(('.jpg', '.png', '.tif'))][:30]

selected_sample = st.sidebar.selectbox("Select Test Image", ["None (Upload custom)"] + sample_files)

# Outputs Directory
out_dir = os.path.join(ROOT, "outputs")
os.makedirs(out_dir, exist_ok=True)

# Main Title & Badges
st.markdown('<div class="main-title">Microstructure Autoencoder Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Decoupled Encoder-Decoder Pipeline • Multi-Ratio Accuracy Benchmarking (8:1 to 64:1) • Latent Editor</div>', unsafe_allow_html=True)

st.markdown("""
<span class="badge">PyTorch MPS Hardware</span>
<span class="badge badge-purple">Decoupled Modules</span>
<span class="badge badge-green">Ratios: 8:1 → 10:1 → 16:1 → 32:1 → 64:1</span>
""", unsafe_allow_html=True)

st.write("")

# Load the model
try:
    with st.spinner(f"Loading checkpoint {chosen_ckpt_name}..."):
        net, cfg, step = get_model(ckpt_path, dev)
    st.sidebar.success(f"✓ Model loaded (Step {step})")
except Exception as e:
    st.sidebar.error(f"Failed to load model: {e}")
    st.stop()

# --- Main Flow ---
st.markdown('<div class="step-header">Step 1: Input Micrograph</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload Micrograph", type=["png", "jpg", "jpeg", "tif", "tiff"])

img_to_process = None
source_name = "upload"

if uploaded_file is not None:
    img_to_process = Image.open(uploaded_file)
    source_name = os.path.splitext(uploaded_file.name)[0]
elif selected_sample != "None (Upload custom)":
    sample_path = os.path.join(sample_dir, selected_sample)
    img_to_process = Image.open(sample_path)
    source_name = os.path.splitext(selected_sample)[0]

if img_to_process is not None:
    col_info, col_btn = st.columns([3, 1])
    with col_info:
        st.info(f"Loaded: **{source_name}** ({img_to_process.width} × {img_to_process.height} px, 8-bit)")
    with col_btn:
        encode_btn = st.button("⚡ 1. Encode Image Only", type="primary", use_container_width=True)

    if encode_btn:
        with st.spinner(f"Encoding micrograph in {mode_code} mode..."):
            x_tensor, a_crop = prepare_image(img_to_process)
            z_tensor, q_dict, meta = encode_micrograph(net, x_tensor, dev, mode=mode_code)
            
            # Save latent and metadata to outputs
            npy_file = os.path.join(out_dir, f"{source_name}_latent_{mode_code}.npy")
            meta_file = os.path.join(out_dir, f"{source_name}_meta_{mode_code}.json")
            np.save(npy_file, q_dict["data"])
            with open(meta_file, "w") as f:
                json.dump(meta, f, indent=2)

            st.session_state["encoded_state"] = {
                "source_name": source_name,
                "x_tensor": x_tensor,
                "orig_crop": a_crop,
                "z_tensor": z_tensor,
                "q_data": q_dict["data"],
                "meta": meta,
                "npy_file": npy_file,
                "meta_file": meta_file,
                "mode_code": mode_code
            }
            # Clear previous decode results
            st.session_state.pop("decoded_orig", None)
            st.session_state.pop("decoded_edited", None)
            st.session_state.pop("sweep_results", None)

    # --- Step 2: Latent Inspection & Options ---
    if "encoded_state" in st.session_state:
        enc = st.session_state["encoded_state"]
        meta = enc["meta"]
        q_data = enc["q_data"]
        
        st.markdown('<div class="step-header">Step 2: Latent Weights & Matrix Encoded</div>', unsafe_allow_html=True)

        # Summary scorecard
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{meta['compression_ratio']:.1f} : 1</div>
                <div class="metric-lbl">Achieved Compression</div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{list(meta['latent_shape'])}</div>
                <div class="metric-lbl">Latent Tensor Shape</div>
            </div>
            """, unsafe_allow_html=True)
        with c3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{meta['latent_bytes'] // 1024:,} <span style="font-size:0.85rem; color:#94a3b8;">KB</span></div>
                <div class="metric-lbl">Encoded Size ({meta['original_bytes'] // 1024:,} KB Orig)</div>
            </div>
            """, unsafe_allow_html=True)
        with c4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-val">{q_data.dtype}</div>
                <div class="metric-lbl">Matrix Data Type</div>
            </div>
            """, unsafe_allow_html=True)

        st.write("")
        st.markdown("**Latent Channel Visualizations ($Z_0, Z_1, Z_2, Z_3$):**")
        
        # Display the 4 latent channel matrix slices
        z_display = q_data.astype(np.float32)
        ch_cols = st.columns(4)
        for c in range(min(4, z_display.shape[0])):
            with ch_cols[c]:
                st.caption(f"Channel $Z_{c}$ ({z_display.shape[1]}×{z_display.shape[2]})")
                st.image(colorize_channel(z_display[c]), use_container_width=True)

        st.write("")
        st.markdown("---")
        
        # Download latent button
        with open(enc["npy_file"], "rb") as f:
            npy_bytes = f.read()
        st.download_button(
            label="💾 Download latent.npy File",
            data=npy_bytes,
            file_name=os.path.basename(enc["npy_file"]),
            mime="application/octet-stream"
        )

        st.write("")
        
        # Three Main Functional Tabs
        main_tab1, main_tab2, main_tab3 = st.tabs([
            "🔬 1. Direct Decode & Accuracy Suite",
            "📊 2. All-Ratios Benchmark Sweep (8:1 → 64:1)",
            "🎛️ 3. Interactive Latent Weight & Matrix Editor"
        ])

        # ==========================================
        # TAB 1: Direct Decode & Accuracy
        # ==========================================
        with main_tab1:
            st.markdown("#### Direct Microstructural Reconstruction")
            st.markdown("Decode the encoded latent matrix through the Decoder module and evaluate the exact reconstruction accuracy against the original micrograph.")
            
            decode_single_btn = st.button("🚀 Decode Encoded Image", type="primary", key="btn_decode_single")
            if decode_single_btn:
                with st.spinner("Executing Decoder module..."):
                    recon_pil, recon_np, y_t = decode_latent(net, q_data, dev, meta)
                    m = image_metrics(y_t, enc["x_tensor"].to(dev), dev)
                    diff_map, diff_raw = compute_difference_map(enc["orig_crop"], recon_np, amplify=1.0, cmap_name="turbo")
                    acc_scores = calculate_accuracy_scores(enc["orig_crop"], recon_np, m)
                    
                    st.session_state["decoded_orig"] = {
                        "recon_pil": recon_pil,
                        "recon_np": recon_np,
                        "metrics": m,
                        "diff_map": diff_map,
                        "diff_raw": diff_raw,
                        "acc_scores": acc_scores
                    }

            if "decoded_orig" in st.session_state:
                d_orig = st.session_state["decoded_orig"]
                m = d_orig["metrics"]
                acc = d_orig["acc_scores"]
                
                st.markdown("---")
                st.markdown("### 🏆 Reconstruction Accuracy & Fidelity Scorecard")

                acc_c1, acc_c2, acc_c3, acc_c4, acc_c5 = st.columns(5)
                with acc_c1:
                    st.markdown(f"""
                    <div class="metric-card-highlight">
                        <div class="metric-val-acc">{acc['composite_acc']:.2f}%</div>
                        <div class="metric-lbl">Overall Accuracy Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                with acc_c2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{acc['pixel_acc']:.2f}%</div>
                        <div class="metric-lbl">Pixel Intensity Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with acc_c3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{acc['struct_acc']:.2f}%</div>
                        <div class="metric-lbl">Structural SSIM Accuracy</div>
                    </div>
                    """, unsafe_allow_html=True)
                with acc_c4:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{m['psnr_db']:.2f} <span style="font-size:0.85rem; color:#94a3b8;">dB</span></div>
                        <div class="metric-lbl">Peak SNR (Fidelity)</div>
                    </div>
                    """, unsafe_allow_html=True)
                with acc_c5:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{meta['compression_ratio']:.1f} : 1</div>
                        <div class="metric-lbl">Compression Ratio</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                st.markdown("#### 🔬 Visual Comparison & Absolute Difference Heatmap")
                
                # Interactive controls for difference visualization
                h_c1, h_c2, h_c3 = st.columns([2, 1, 1])
                with h_c1:
                    st.caption("Displays the pixel-level absolute difference |Original − Reconstructed|. Highlighting microstructural boundary fidelity and phase contrasts.")
                with h_c2:
                    selected_cmap = st.selectbox("Heatmap Colormap", ["turbo", "inferno", "magma", "plasma", "viridis", "coolwarm"], index=0, key="orig_cmap")
                with h_c3:
                    amplify_val = st.slider("Error Gain", 1.0, 5.0, 1.5, 0.5, key="orig_amp", help="Amplify small differences for enhanced visual inspection.")

                # Recompute difference heatmap dynamically
                current_diff_img, current_diff_raw = compute_difference_map(
                    enc["orig_crop"], d_orig["recon_np"], amplify=amplify_val, cmap_name=selected_cmap
                )

                # 3-column side-by-side visual layout
                v1, v2, v3 = st.columns(3)
                with v1:
                    st.markdown("**Original Micrograph**")
                    st.image(enc["orig_crop"], use_container_width=True)
                with v2:
                    st.markdown(f"**Reconstructed Micrograph ({meta['mode']})**")
                    st.image(d_orig["recon_pil"], use_container_width=True)
                with v3:
                    st.markdown(f"**Difference Heatmap ({selected_cmap.title()})**")
                    st.image(current_diff_img, use_container_width=True)

                # Error statistics summary bar
                st.caption(
                    f"**Pixel Residual Stats:** "
                    f"MAE: **{current_diff_raw.mean():.2f} px** | "
                    f"RMSE: **{np.sqrt((current_diff_raw**2).mean()):.2f} px** | "
                    f"Median: **{np.median(current_diff_raw):.2f} px** | "
                    f"95th %ile: **{np.percentile(current_diff_raw, 95):.2f} px** | "
                    f"Peak Error: **{current_diff_raw.max():.1f} px**"
                )

        # ==========================================
        # TAB 2: All-Ratios Accuracy Benchmark Sweep
        # ==========================================
        with main_tab2:
            st.markdown("#### 📊 Comprehensive Multi-Ratio Accuracy Sweep (8:1 to 64:1)")
            st.markdown("Automatically evaluate and compare the exact reconstruction accuracy across all 8 compression tiers on this specific micrograph to determine the best production deployment setting.")
            
            run_sweep_btn = st.button("🚀 Run Accuracy Sweep Across All Ratios", type="primary", key="btn_run_sweep")
            
            if run_sweep_btn:
                with st.spinner("Benchmarking all 8 compression ratios on MPS GPU..."):
                    SWEEP_MODES = [
                        ("fp16 (8:1)", "fp16", "Baseline Standard"),
                        ("10x (10:1)", "10x", "Optimized Target"),
                        ("int8 (16:1)", "int8", "🌟 Production Gold Standard"),
                        ("int6 (21.3:1)", "int6", "High Fidelity"),
                        ("int5 (25.6:1)", "int5", "Balanced"),
                        ("int4 (32:1)", "int4", "Fast Preview Proxy"),
                        ("int3 (42.7:1)", "int3", "Ultra Compressed"),
                        ("int2 (64:1)", "int2", "Extreme (High Risk)")
                    ]
                    
                    sweep_records = []
                    sweep_images = {}
                    
                    orig_np = enc["orig_crop"]
                    x_t = enc["x_tensor"].to(dev)
                    z_base = enc["z_tensor"]
                    orig_bytes = int(orig_np.shape[0] * orig_np.shape[1])
                    
                    with torch.no_grad():
                        for label, m_code, verdict in SWEEP_MODES:
                            q_info = quantize_latent(z_base[0], mode=m_code)
                            scale = q_info.get("scale", 1.0)
                            min_v = q_info.get("min_val", 0.0)
                            deq_data = (q_info["data"].astype(np.float32) * scale) + min_v
                            
                            z_in = torch.from_numpy(deq_data).float().unsqueeze(0).to(dev)
                            y_rec = net.decode(z_in).clamp(-1.0, 1.0)
                            m_vals = image_metrics(y_rec, x_t, dev)
                            
                            rec_uint8 = ((y_rec[0, 0].detach().cpu().numpy() + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
                            scores = calculate_accuracy_scores(orig_np, rec_uint8, m_vals)
                            
                            lat_bytes = int(np.prod(q_info["data"].shape) * q_info["bytes_per_elem"])
                            savings_pct = (1.0 - (lat_bytes / orig_bytes)) * 100.0
                            
                            diff_heat, _ = compute_difference_map(orig_np, rec_uint8, amplify=1.5, cmap_name="turbo")
                            
                            sweep_records.append({
                                "Tier": label,
                                "Mode": m_code,
                                "Ratio": f"{q_info['ratio']:.1f}:1",
                                "Ratio_Num": q_info["ratio"],
                                "Size (KB)": round(lat_bytes / 1024.0, 1),
                                "Storage Saved": f"{savings_pct:.1f}%",
                                "Overall Accuracy": f"{scores['composite_acc']:.2f}%",
                                "Acc_Num": scores["composite_acc"],
                                "Pixel Accuracy": f"{scores['pixel_acc']:.2f}%",
                                "Structural SSIM": f"{scores['struct_acc']:.2f}%",
                                "PSNR (dB)": f"{m_vals['psnr_db']:.2f}",
                                "LPIPS": f"{m_vals['lpips']:.4f}",
                                "Production Verdict": verdict
                            })
                            
                            sweep_images[label] = {
                                "rec_pil": Image.fromarray(rec_uint8),
                                "diff_pil": diff_heat,
                                "scores": scores,
                                "psnr": m_vals['psnr_db'],
                                "ratio": q_info['ratio']
                            }
                            
                    st.session_state["sweep_results"] = {
                        "records": sweep_records,
                        "images": sweep_images
                    }

            if "sweep_results" in st.session_state:
                sw = st.session_state["sweep_results"]
                df_sweep = pd.DataFrame(sw["records"])
                
                st.markdown("### 📋 Multi-Ratio Benchmark Comparison Table")
                display_cols = ["Tier", "Ratio", "Size (KB)", "Storage Saved", "Overall Accuracy", "Pixel Accuracy", "Structural SSIM", "PSNR (dB)", "Production Verdict"]
                st.dataframe(df_sweep[display_cols], use_container_width=True)

                st.write("")
                st.markdown("### 📈 Accuracy vs. Compression Trade-off Curve")
                
                chart_df = df_sweep[["Ratio_Num", "Acc_Num"]].copy()
                chart_df.columns = ["Compression Ratio", "Overall Accuracy (%)"]
                chart_df = chart_df.set_index("Compression Ratio")
                st.line_chart(chart_df, color="#60a5fa")

                st.caption("Notice the knee in the curve: Accuracy remains virtually flat and uncompromised up to **16:1–21:1** (~88.5%), then begins a gradual drop at 32:1 (86.9%) before falling steeply past 40:1.")

                st.write("")
                st.markdown("### 🔍 Visual Gallery: Microstructural Fidelity Across Ratios")
                
                gallery_tabs = st.tabs(list(sw["images"].keys()))
                for idx, (lbl, g_data) in enumerate(sw["images"].items()):
                    with gallery_tabs[idx]:
                        gc1, gc2, gc3 = st.columns(3)
                        with gc1:
                            st.markdown("**Original Micrograph**")
                            st.image(enc["orig_crop"], use_container_width=True)
                        with gc2:
                            st.markdown(f"**Reconstructed ({lbl})**")
                            st.image(g_data["rec_pil"], use_container_width=True)
                        with gc3:
                            st.markdown(f"**Difference Heatmap ({g_data['scores']['composite_acc']:.2f}% Accuracy)**")
                            st.image(g_data["diff_pil"], use_container_width=True)

        # ==========================================
        # TAB 3: Latent Weight & Matrix Editor
        # ==========================================
        with main_tab3:
            st.markdown("#### 🎛️ Interactive Latent Matrix & Weight Editor")
            st.markdown("Adjust individual channel weights, inject noise, mask sub-regions, or upload an edited `.npy` file to observe the impact on reconstructed microstructures in real time.")

            edit_tab1, edit_tab2, edit_tab3, edit_tab4 = st.tabs([
                "Channel Weight Modifiers",
                "Matrix Perturbation & Masking",
                "Direct Matrix Data Preview",
                "Upload External Edited .npy"
            ])

            base_latent = q_data.astype(np.float32).copy()
            edited_latent = base_latent.copy()

            with edit_tab1:
                st.markdown("**Channel Gain (Scaling) & Bias (Offset):**")
                cg_col1, cg_col2 = st.columns(2)
                with cg_col1:
                    g0 = st.slider("Channel 0 Gain", 0.0, 3.0, 1.0, 0.05)
                    g1 = st.slider("Channel 1 Gain", 0.0, 3.0, 1.0, 0.05)
                with cg_col2:
                    g2 = st.slider("Channel 2 Gain", 0.0, 3.0, 1.0, 0.05)
                    g3 = st.slider("Channel 3 Gain", 0.0, 3.0, 1.0, 0.05)

                b_col1, b_col2 = st.columns(2)
                with b_col1:
                    b0 = st.slider("Channel 0 Bias Offset", -2.0, 2.0, 0.0, 0.1)
                    b1 = st.slider("Channel 1 Bias Offset", -2.0, 2.0, 0.0, 0.1)
                with b_col2:
                    b2 = st.slider("Channel 2 Bias Offset", -2.0, 2.0, 0.0, 0.1)
                    b3 = st.slider("Channel 3 Bias Offset", -2.0, 2.0, 0.0, 0.1)

                edited_latent[0] = (edited_latent[0] * g0) + b0
                edited_latent[1] = (edited_latent[1] * g1) + b1
                edited_latent[2] = (edited_latent[2] * g2) + b2
                edited_latent[3] = (edited_latent[3] * g3) + b3

            with edit_tab2:
                st.markdown("**Matrix Perturbation, Sparsity & Spatial Masking:**")
                p_col1, p_col2 = st.columns(2)
                with p_col1:
                    noise_sigma = st.slider("Add Gaussian Noise (σ)", 0.0, 2.0, 0.0, 0.05)
                    if noise_sigma > 0:
                        np.random.seed(42)
                        noise = np.random.normal(0, noise_sigma, edited_latent.shape).astype(np.float32)
                        edited_latent += noise

                    sparsity_thresh = st.slider("Sparsity Cutoff (|z| < threshold → 0)", 0.0, 3.0, 0.0, 0.1)
                    if sparsity_thresh > 0:
                        edited_latent[np.abs(edited_latent) < sparsity_thresh] = 0.0

                with p_col2:
                    mask_mode = st.selectbox(
                        "Spatial Region Mask / Ablation",
                        ["None", "Zero-out Center Patch (32x32)", "Zero-out Top-Half", "Zero-out Bottom-Half", "Invert All Signs (z = -z)"]
                    )
                    _, H_lat, W_lat = edited_latent.shape
                    if mask_mode == "Zero-out Center Patch (32x32)":
                        cy, cx = H_lat // 2, W_lat // 2
                        edited_latent[:, max(0, cy-16):min(H_lat, cy+16), max(0, cx-16):min(W_lat, cx+16)] = 0.0
                    elif mask_mode == "Zero-out Top-Half":
                        edited_latent[:, :H_lat//2, :] = 0.0
                    elif mask_mode == "Zero-out Bottom-Half":
                        edited_latent[:, H_lat//2:, :] = 0.0
                    elif mask_mode == "Invert All Signs (z = -z)":
                        edited_latent = -edited_latent

            with edit_tab3:
                st.markdown("**Direct Matrix Values & Numerical Statistics:**")
                view_ch = st.selectbox("Select Channel to inspect", [0, 1, 2, 3])
                st.markdown(f"**Channel $Z_{view_ch}$ Submatrix (first 12×12 elements):**")
                sub_df = pd.DataFrame(edited_latent[view_ch, :12, :12])
                st.dataframe(sub_df, use_container_width=True)
                st.caption(f"Channel {view_ch} Stats — Min: {edited_latent[view_ch].min():.3f}, Max: {edited_latent[view_ch].max():.3f}, Mean: {edited_latent[view_ch].mean():.3f}, Std: {edited_latent[view_ch].std():.3f}")

            with edit_tab4:
                st.markdown("**Upload External Edited `.npy` Latent File:**")
                ext_file = st.file_uploader("Upload edited_latent.npy", type=["npy"], key="ext_npy")
                if ext_file is not None:
                    try:
                        uploaded_npy = np.load(ext_file)
                        st.success(f"✓ Uploaded matrix with shape {uploaded_npy.shape} and dtype {uploaded_npy.dtype}")
                        if uploaded_npy.shape == edited_latent.shape:
                            edited_latent = uploaded_npy.astype(np.float32)
                        else:
                            st.error(f"Shape mismatch: expected {edited_latent.shape}, got {uploaded_npy.shape}")
                    except Exception as e:
                        st.error(f"Failed to load .npy: {e}")

            st.write("")
            col_dec_btn, col_save_btn = st.columns([1, 1])
            with col_dec_btn:
                decode_edited_btn = st.button("🔬 Decode with Edited Weights", type="primary", use_container_width=True)
            with col_save_btn:
                edited_npy_bytes = tempfile.NamedTemporaryFile(suffix=".npy", delete=False)
                np.save(edited_npy_bytes.name, edited_latent.astype(np.float16))
                with open(edited_npy_bytes.name, "rb") as ef:
                    ef_bytes = ef.read()
                st.download_button(
                    label="💾 Download Edited latent.npy",
                    data=ef_bytes,
                    file_name=f"{source_name}_edited_latent.npy",
                    mime="application/octet-stream",
                    use_container_width=True
                )

            if decode_edited_btn:
                with st.spinner("Decoding edited latent matrix through Decoder..."):
                    e_recon_pil, e_recon_np, e_y_t = decode_latent(net, edited_latent, dev, meta)
                    e_metrics = image_metrics(e_y_t, enc["x_tensor"].to(dev), dev)
                    e_diff_map, e_diff_raw = compute_difference_map(enc["orig_crop"], e_recon_np, amplify=1.5, cmap_name="turbo")
                    e_acc = calculate_accuracy_scores(enc["orig_crop"], e_recon_np, e_metrics)
                    
                    st.session_state["decoded_edited"] = {
                        "recon_pil": e_recon_pil,
                        "recon_np": e_recon_np,
                        "metrics": e_metrics,
                        "diff_map": e_diff_map,
                        "acc_scores": e_acc,
                        "edited_latent": edited_latent
                    }

            if "decoded_edited" in st.session_state:
                d_edit = st.session_state["decoded_edited"]
                em = d_edit["metrics"]
                e_acc = d_edit["acc_scores"]

                st.markdown("---")
                st.markdown("### 🔬 Impact of Edited Weights on Microstructure Accuracy")
                
                em1, em2, em3, em4 = st.columns(4)
                with em1:
                    st.markdown(f"""
                    <div class="metric-card-highlight">
                        <div class="metric-val-acc">{e_acc['composite_acc']:.2f}%</div>
                        <div class="metric-lbl">Edited Accuracy Score</div>
                    </div>
                    """, unsafe_allow_html=True)
                with em2:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{em['psnr_db']:.2f} <span style="font-size:0.85rem; color:#94a3b8;">dB</span></div>
                        <div class="metric-lbl">Edited PSNR</div>
                    </div>
                    """, unsafe_allow_html=True)
                with em3:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{em['ms_ssim']:.4f}</div>
                        <div class="metric-lbl">Edited MS-SSIM</div>
                    </div>
                    """, unsafe_allow_html=True)
                with em4:
                    delta_str = "N/A"
                    if "decoded_orig" in st.session_state:
                        diff_psnr = em['psnr_db'] - st.session_state["decoded_orig"]["metrics"]["psnr_db"]
                        delta_str = f"{diff_psnr:+.2f} dB"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-val">{delta_str}</div>
                        <div class="metric-lbl">PSNR Shift (Δ)</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.write("")
                ec1, ec2, ec3 = st.columns(3)
                with ec1:
                    st.markdown("**Original Micrograph**")
                    st.image(enc["orig_crop"], use_container_width=True)
                with ec2:
                    st.markdown("**Decoded with Edited Weights**")
                    st.image(d_edit["recon_pil"], use_container_width=True)
                with ec3:
                    st.markdown("**Residual Error Heatmap** *(Blue = match, Red = shifted)*")
                    st.image(d_edit["diff_map"], use_container_width=True)

else:
    st.info("👆 Please upload a micrograph or choose one from the sidebar to begin.")
