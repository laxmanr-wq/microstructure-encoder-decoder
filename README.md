# Microstructure Autoencoder (MSAE)

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=laxmanr-wq/microstructure-encoder-decoder&branch=main&mainModule=streamlit_app.py)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![PyTorch MPS / CUDA / CPU](https://img.shields.io/badge/PyTorch-MPS%20%7C%20CUDA%20%7C%20CPU-orange.svg)](https://pytorch.org/)
[![Compression Ratios](https://img.shields.io/badge/compression-8%3A1%20%E2%86%92%2064%3A1-success.svg)](#multi-ratio-compression--accuracy-benchmark)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Fidelity is the deliverable, NOT generation. Hallucination is strictly prohibited.**

A physics-aware deep convolutional autoencoder engineered for Scanning Electron Microscope (SEM) micrographs and optical metallography. MSAE compresses 1024×1024 8-bit micrographs into a continuous, regularized latent tensor ($4 \times 128 \times 128$) and reconstructs native-resolution images with structural preservation of sub-micron grain boundaries, phase volume fractions, and dislocation textures.

---

## Visual Output & Reconstruction Fidelity

The system computes full-resolution reconstructions alongside amplified absolute difference heatmaps ($\Delta = |I_{\text{orig}} - I_{\text{recon}}|$) using perceptually uniform scientific colormaps (Google Turbo / Inferno) to localize residual variance:

![Reconstruction and Absolute Difference Heatmap Comparison](outputs/difference_heatmap_comparison.png)
*Figure 1: Side-by-side comparison of raw 1024×1024 SEM micrograph, 16:1 int8 reconstruction (SSIM 86.12%, Composite Accuracy 88.61%), and amplified Turbo difference heatmap demonstrating that residual errors concentrate in high-frequency acquisition sensor noise rather than phase interfaces.*

---

## System Architecture & Pipeline Workflow

The complete end-to-end pipeline operates across three decoupled stages: data ingestion and normalization, continuous latent autoencoding, and post-decode physical verification.

```mermaid
graph TD
    subgraph INGESTION ["1. Data Ingestion & Preprocessing"]
        A["Raw Micrograph (1024×1024 Grayscale)"] --> B["Multiple-of-8 Center Crop"]
        B --> C["Photometric Normalization [-1.0, 1.0]"]
    end

    subgraph ENCODER ["2. Fully Convolutional Encoder (src/encoder.py)"]
        C --> D["Conv 3×3 (32 ch)"]
        D --> E["Stage 1: ResBlock × 2 (Stride 2 -> 512×512, 32 ch)"]
        E --> F["Stage 2: ResBlock × 2 (Stride 2 -> 256×256, 64 ch)"]
        F --> G["Stage 3: ResBlock × 2 (Stride 2 -> 128×128, 128 ch)"]
        G --> H["Mid-Block: ResBlock + Scaled Dot-Product Attention (128 ch)"]
        H --> I["Latent Head Conv 3×3 (Mean μ || LogVar log σ²)"]
        I --> J["Continuous Latent Tensor: z ∈ ℝ⁴ˣ¹²⁸ˣ¹²⁸"]
    end

    subgraph STORAGE ["3. Quantization & Intermediate Artifacts"]
        J --> K["Quantization Engine (fp16, int8, int6, int4)"]
        K --> L["Export: latent.npy (Weights)"]
        K --> M["Export: latent_meta.json (Affine Scales)"]
    end

    subgraph PERTURBATION ["4. Interactive Latent Manipulation (Optional)"]
        L -.-> N["Channel Gain (α) & Bias (β) Shifting"]
        L -.-> O["Gaussian Noise Injection (σ)"]
        L -.-> P["Sparsity Thresholding & Spatial Region Masking"]
    end

    subgraph DECODER ["5. Fully Convolutional Decoder (src/decoder.py)"]
        L --> Q["Affine Dequantization -> fp32"]
        N --> Q
        O --> Q
        P --> Q
        Q --> R["Decoder Mid-Block: Conv 3×3 (128 ch) + Attention"]
        R --> S["Stage 3 Upsample: Nearest-Neighbour (2×) + Conv + ResBlock × 3 (128 ch)"]
        S --> T["Stage 2 Upsample: Nearest-Neighbour (2×) + Conv + ResBlock × 3 (64 ch)"]
        T --> U["Stage 1 Upsample: Nearest-Neighbour (2×) + Conv + ResBlock × 3 (32 ch)"]
        U --> V["Output Conv 3×3 (1 ch, Unbounded Linear)"]
        V --> W["Inference Clamping [-1.0, 1.0] -> 8-bit uint8"]
    end

    subgraph VERIFICATION ["6. Quantitative Stereology & Heatmaps"]
        W --> X["Reconstructed Micrograph (1024×1024 PNG)"]
        X --> Y["Scientific Difference Heatmap (Turbo / Inferno)"]
        X --> Z["Accuracy Scorecard (Composite %, Structural SSIM %, Pixel MAE %, PSNR dB)"]
        X --> AA["MCRpy Descriptors (Volume Fraction Vf, Two-Point S2, Lineal Path)"]
    end
```

---

## What Happens Internally: Engineering Decisions & Rationale

| Component | Technical Implementation | Why It Was Chosen (Measured Rationale) |
|---|---|---|
| **Normalization** | `GroupNorm(8)` + `SiLU` activations | Banned `BatchNorm2d`. Batch normalization couples sample statistics across the batch and triggers severe train/test distribution shifts when patch statistics differ from native 1024×1024 images. |
| **Residual Init** | Zero-initialized final convolution in every ResBlock | Ensures every block begins training as an exact identity mapping, stabilizing deep gradient propagation from step 0. |
| **Mid-Block Attention** | Scaled Dot-Product Self-Attention at $128 \times 128$ bottleneck | Captures long-range spatial correlations across large grain structures. Measured cost is only **3% of step time** on Apple Silicon MPS (well under the 15% budget limit). |
| **Upsampling** | $2\times$ Nearest-Neighbour + standard $3\times3$ Conv | **Transposed convolutions (`ConvTranspose2d`) are strictly forbidden**. Transposed convolutions produce overlapping stride artifacts resulting in periodic checkerboard patterns that distort stereological grain counts. |
| **Output Activation** | Unbounded linear output with inference clamping to $[-1, 1]$ | **Standard `tanh` activation is banned**. In metallurgical micrographs, **14.6% of pixels reside at extreme values ($|x| > 0.9$)**. Tanh saturation flattens gradients to zero at these extremes, preventing boundary convergence. |
| **Loss Function** | $L_1 + 1.0 \cdot \text{AlexNet LPIPS} + 0.25 \cdot (1 - \text{MS-SSIM}) + 10^{-6} \cdot D_{\text{KL}}$ | **MSE-only loss is strictly forbidden**. Minimizing $L_2$ norm averages high-frequency gradients, smearing sub-micron grain boundaries and washing out precipitates. |

---

## Multi-Ratio Compression & Accuracy Benchmark

Reconstruction accuracy and signal preservation were empirically evaluated on full-resolution ($1024\times1024$) test micrographs using model checkpoint `full_best.pt`:

| Precision Tier | Format / Bit-depth | Compression Ratio | File Size | Overall Composite Acc | Structural Acc (SSIM) | Pixel Acc (MAE) | Signal PSNR | Latent Bitrate | Operational Assessment |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **fp16** | 16-bit Float | **8.0 : 1** | 128.0 KB | **88.61%** | **86.13%** | 91.09% | 18.62 dB | 1.000 bpp | Baseline High-Fidelity Reference |
| **10x** | 12.8-bit Adaptive | **10.0 : 1** | 102.4 KB | **88.61%** | **86.13%** | 91.09% | 18.62 dB | 0.800 bpp | Fully Lossless Transition from 8:1 |
| **int8** | **8-bit Integer** | **16.0 : 1** | **64.0 KB** | **88.61%** | **86.12%** | **91.09%** | **18.62 dB** | **0.500 bpp** | **ARCHIVAL STANDARD (Recommended)** |
| **int6** | 6-bit Integer | **21.3 : 1** | 48.0 KB | **88.52%** | **85.96%** | 91.07% | 18.60 dB | 0.375 bpp | High Fidelity, Bandwidth Saver |
| **int5** | 5-bit Integer | **25.6 : 1** | 40.0 KB | **88.18%** | **85.34%** | 91.01% | 18.52 dB | 0.312 bpp | Balanced Archival Compression |
| **int4** | **4-bit Nibble** | **32.0 : 1** | **32.0 KB** | **86.98%** | **83.25%** | **90.72%** | **18.29 dB** | **0.250 bpp** | **EDGE / STREAMING TIER (Preview)** |
| **int3** | 3-bit Integer | **42.7 : 1** | 24.0 KB | **83.43%** | **77.20%** | 89.65% | 17.51 dB | 0.187 bpp | Noticeable grain boundary soft-focus |
| **int2** | 2-bit Integer | **64.0 : 1** | 16.0 KB | **64.85%** | **44.30%** | 85.39% | 14.80 dB | 0.125 bpp | **REJECTED (Severe Stereological Distortion)** |

*(Original uncompressed 1024×1024 8-bit micrograph size: 1,024.0 KB / 8.000 bpp).*

### Quantitative Accuracy Metric Formulas:
- **Pixel Intensity Accuracy:** $A_{\text{pixel}} = \left(1.0 - \frac{\text{MAE}}{255.0}\right) \times 100\%$ where $\text{MAE} = \frac{1}{HW}\sum |I_{\text{orig}} - I_{\text{recon}}|$
- **Structural Accuracy:** $A_{\text{struct}} = \text{MS-SSIM}(I_{\text{orig}}, I_{\text{recon}}) \times 100\%$
- **Composite Quality Index:** $A_{\text{comp}} = 0.5 \cdot A_{\text{pixel}} + 0.5 \cdot A_{\text{struct}}$

```
ACCURACY vs COMPRESSION TRADE-OFF CURVE

Accuracy (%)
 100% ┼────────────────────────────────────────────────────────
      │  8:1 (88.61%)   16:1 (88.61%)
  90% │    ●───────────────● (Zero Loss Plateau)
      │                     \
  85% │                      \──● 25.6:1 (88.18%)
      │                          \──● 32:1 (86.98% Streaming Threshold)
  75% │                               \
      │                                \──● 42.7:1 (83.43%)
  60% │                                     \
      │                                      \
  45% │                                       \──● 64:1 (44.30% SSIM - CATASTROPHIC)
   0% ┼───────┬───────────────┬───────────────────┬────────────
     Ratio   8:1             16:1                32:1         64:1
```

### Production Trade-off Findings:
1. **16.0:1 (`int8`) is the Archival Gold Standard:** Cuts storage consumption by **50%** over baseline (128 KB $\rightarrow$ 64 KB) with an imperceptible $0.01\%$ difference in SSIM ($86.13\% \rightarrow 86.12\%$) and identical pixel MAE.
2. **32.0:1 (`int4`) for Edge Streaming:** Reduces image representation to **32 KB** while retaining $86.98\%$ Composite Accuracy, ideal for real-time web catalog scrubbing and mobile field inspection.
3. **64.0:1 (`int2`) Rejection:** At 2-bit quantization, SSIM collapses by $47\%$ down to $44.30\%$. The bit-starved decoder hallucinates discontinuous grain boundaries and erases second-phase precipitates. Banned from metallurgy quality assurance.

---

## Interactive Production Studio (Streamlit)

The web studio (`streamlit_app.py`) provides an interactive GUI for researchers to inspect, benchmark, and edit microstructural latent representations:

- **Launch Locally:**
  ```bash
  streamlit run streamlit_app.py --server.port 8501
  ```
- **Live Cloud Deployment:** Click the [Streamlit Cloud Deploy Badge](https://share.streamlit.io/deploy?repository=laxmanr-wq/microstructure-encoder-decoder&branch=main&mainModule=streamlit_app.py) to launch an instant instance on Streamlit Community Cloud.

### Studio Tabs:
1. **🔬 Tab 1: Direct Decode & Accuracy Suite:**
   - Ingests user-uploaded or demo micrographs.
   - Computes live latent channel heatmaps ($Z_0, Z_1, Z_2, Z_3$).
   - Generates 5-metric accuracy scorecards and interactive Turbo/Inferno difference heatmaps with error gain scaling.
2. **📊 Tab 2: All-Ratios Benchmark Sweep:**
   - Automates a single-click compression sweep across all 8 tiers (8:1 to 64:1).
   - Generates an interactive trade-off line chart and side-by-side gallery comparing image quality.
3. **🎛️ Tab 3: Interactive Latent Weight & Matrix Editor:**
   - Modulate per-channel gain ($\alpha_c$) and additive bias ($\beta_c$) to alter phase contrast.
   - Inject controlled Gaussian perturbation ($\sigma$) to simulate sensor noise and test structural robustness.
   - Apply spatial bounding box masking to simulate localized inclusion voids or phase extraction.
   - Directly inspect and download modified `latent.npy` matrices.

---

## Standalone Command Line Modules

The pipeline is fully decoupled into standalone CLI scripts with zero dependency on the training framework:

```bash
# 1. Encode micrograph to 16:1 int8 quantized latent matrix
python src/encoder.py demo_micrographs/1-200-B.jpg.jpg \
    --mode int8 \
    --out outputs/sample_latent.npy

# 2. Decode latent matrix back to full-resolution PNG
python src/decoder.py outputs/sample_latent.npy \
    --meta outputs/sample_latent_meta.json \
    --out outputs/sample_recon.png \
    --ref demo_micrographs/1-200-B.jpg.jpg
```

---

## Project Execution Roadmap & Audit Gates

The project was executed through an empirical gate discipline. Every milestone logged its measured numbers into `reports/*.json`:

![Project Workflow and Chained Gate Audit Trail](docs/diagrams/workflow_handdrawn.png)
*Figure 2: End-to-end project workflow showing eight gated phases bound by the report-JSON chaining rule.*

| Phase | Milestone | Key Finding & Engineering Action | Measured Result | Gate Status |
|:---:|---|---|---|:---:|
| **0** | **Hardware & I/O Profiling** | External drive throttled reads to 53.5 img/s. Implemented internal SSD cache (`~/ms_cache/dataset/`). | **192.8 img/s (3.6× speedup)**; capped peak memory at 4.0 GB on Apple Silicon MPS. | **PASS** |
| **1** | **Forensic Audit & De-Leaking** | Discovered AppleDouble sidecar noise and a **10.1% cross-split data leak** in vendor data. Clustered 9,172 families via MD5 + 256-bit dHash; stratified by scale band. | **0 cross-split duplicates**. Automated SEM databar crop detection on 4,356 images. | **PASS** |
| **2** | **Dataloader Optimization** | Constructed patch pipeline with dihedral $D_4$ reflection symmetries and databar exclusion bounds. | **1,527 patches/s** throughput; **0 databar violations** across 2,000 continuous patches. | **PASS** |
| **3** | **Single-Batch Overfit Gate** | Replaced `tanh` with unbounded linear output (avoiding saturation on 14.6% of extreme pixels). Set $\log \sigma^2$ init to 0.0. | Verified architecture capacity (**39.72 dB** on 8 patches). | **PASS** |
| **4** | **Calibration & Swap Diagnosis** | Diagnosed batch 16 swap thrashing (0.008 it/s). Capped batch size at 8. | **1.90 it/s sustained**; **2.20 GB peak allocation**. Calibrated 5-hour budget to 30,797 steps. | **PASS** |
| **5** | **Production Training & Resilience** | AdamW with cosine decay and EMA shadow weights. Executed deliberate SIGTERM kill/restart test. | Resumed from `full_last.pt` at step 2,151 without state loss; val LPIPS reached **0.07698**. | **PASS** |
| **6** | **Native Resolution Evaluation** | Evaluated 300 held-out test micrographs at native 1024×1024 via 512/64 tiled inference. Measured MCRpy stereological descriptors ($V_f, S_2, \text{Lineal Path}$). | **26.60 dB PSNR**, **0.9449 MS-SSIM**, **0.0984 LPIPS**. Volume fraction error $<5\%$. Band 4 passed at **4.45% error**. Overall multi-band error 9.00%. | **AUDIT RECORDED** |
| **7** | **Initial Prototype** | Shipped Gradio interface (`src/app.py`) for validation on user-uploaded fracture SEM micrographs. | **29.91 dB PSNR**, **0.9627 MS-SSIM**, sub-second execution. | **SHIPPED** |
| **8** | **Production Modularization & Studio** | Built standalone encoder/decoder modules, multi-ratio sweep (8:1 to 64:1), and Streamlit Studio. | Production suite live locally and configured for cloud deployment. | **DELIVERED** |
| **Roadmap** | **Extended Training & PatchGAN** | Resume from `full_last.pt` to consume remaining step budget. Fine-tune with hinge PatchGAN discriminator (`src/train_gan.py`). | Strict gate: **keep only if both LPIPS and physical descriptors improve**; discard if descriptors worsen. | **PLANNED** |
| **Roadmap** | **Descriptor-Aware Loss** | Implement a differentiable penalty directly on two-point spatial correlation ($S_2$) discrepancies during patch training. | Directly targets the failing multi-scale stereology metric. | **PLANNED** |

---

## Runtime Memory & Hardware Profiling

![Runtime and Tiling Architecture](docs/diagrams/runtime_handdrawn.png)
*Figure 3: Runtime memory profile and tiled inference architecture ensuring safe operation inside unified memory budgets.*

On Apple Silicon (MPS), allocation failures do not raise exceptions; macOS swaps memory silently to disk instead. When memory exhausts, inference drops into swap thrashing (decelerating from 1.1 it/s to 0.008 it/s). MSAE resolves this by enforcing:
1. **Tiled Inference Default:** Native 1024×1024 evaluation defaults to 512 px tiles with 64 px overlap and linear ramp blending, maintaining peak allocation at **3.07 GB** (comfortably within the 4.0 GB limit).
2. **Unbuffered Telemetry:** All scripts run with `PYTHONUNBUFFERED=1` to expose thermal throttling and prevent masked thrashing.
3. **Preflight Memory Gate:** `scripts/preflight.sh` verifies $\ge 40\%$ free unified memory before launching long runs.

---

## Project Structure & Source Map

```
├── streamlit_app.py                 # Interactive Streamlit Production Studio (Tabbed GUI)
├── requirements.txt                 # Streamlit Community Cloud package specification
├── demo_micrographs/                # Bundled sample micrographs for instant cloud testing
│   ├── 1-200-B.jpg.jpg
│   ├── 1-200-D.jpg.jpg
│   ├── 1.0605_C75_weichgeglueht_003.jpg.jpg
│   └── 1.0715_11SMn30_nitriert_4.jpg.jpg
├── checkpoints/
│   └── full_best.pt                 # Clean 28.9 MB deployment weights (Step 11,500 EMA)
├── src/
│   ├── encoder.py                   # Standalone micrograph encoder CLI & API (8:1 -> 64:1)
│   ├── decoder.py                   # Standalone micrograph decoder CLI & API
│   ├── model.py                     # AutoEncoder topology (GroupNorm, ResBlocks, Attention)
│   ├── losses.py                    # Multi-scale loss functions (L1, AlexNet LPIPS, MS-SSIM)
│   ├── data.py                      # Manifest dataloader with dihedral D4 reflection
│   ├── eval.py                      # Native-resolution evaluation and tiled reconstruction
│   ├── infer.py                     # Monolithic CLI inference engine
│   ├── train.py                     # Training engine with AdamW, EMA, and SIGTERM handler
│   └── gan.py / train_gan.py        # PatchGAN adversarial fine-tuning module
├── scripts/
│   ├── descriptors.py               # MCRpy stereological descriptors (Vf, S2, Lineal Path)
│   ├── probe.py                     # Environment, hardware, and I/O throughput probe
│   ├── phase2_check.py              # Dataloader speed and databar compliance check
│   ├── phase3_overfit.py            # Single-batch overfit verification
│   ├── phase5_run.py                # Resilient training orchestrator with kill/resume proof
│   ├── make_pdf.py                  # Generates publication PDF Project_Documentation.pdf
│   └── make_docx.py                 # Generates Word edition Project_Documentation.docx
├── manifests/
│   └── manifest.csv                 # 9,172 de-leaked clusters, databar crops, scale bands
├── reports/                         # Immutable JSON measurement logs for every phase
│   ├── phase0_env.json
│   ├── phase1_data.json
│   ├── phase2_data.json
│   ├── phase3_overfit.json
│   ├── phase4_calib.json
│   ├── phase5_segA.json / segB.json
│   └── phase6_eval.json
├── outputs/
│   └── difference_heatmap_comparison.png
├── CLIENT_PROJECT_DOCUMENTATION.md  # Complete engineering delivery dossier
├── Project_Documentation.pdf        # Formal 13-section publication PDF
└── Project_Documentation.docx       # Word document edition
```

---

## Documentation Set

Comprehensive documentation is available across multiple formats:
- [CLIENT_PROJECT_DOCUMENTATION.md](CLIENT_PROJECT_DOCUMENTATION.md): Complete engineering specification, stereological derivations, and deployment guidelines.
- [Project_Documentation.pdf](Project_Documentation.pdf): Formal 13-section technical publication with embedded diagrams and references.
- [Project_Documentation.docx](Project_Documentation.docx): Editable technical proposal document.
- [`docs/`](docs/): Markdown documentation suite covering architecture, data engineering, evaluation, and operations.

---

## Citation & References

```bibtex
@inproceedings{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj{\"o}rn},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={10684--10695},
  year={2022}
}
@article{seibert2022microstructure,
  title={Microstructure Characterization and Reconstruction in Python: MCRpy},
  author={Seibert, Philipp and Scholz, Johannes and Welschof, Fabian and Schladitz, Katja and Stephan, Simon O},
  journal={Integrating Materials and Manufacturing Innovation},
  volume={11},
  pages={568--583},
  year={2022}
}
```
