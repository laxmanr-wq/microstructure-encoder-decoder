"""Draw hand-drawn-style (Excalidraw) workflow diagrams: PNG for the docx/PDF and
editable .excalidraw sources. Uses matplotlib xkcd mode + the Handwritten font when
available, matching Excalidraw's aesthetic (rough ink, wobbly arrows, hachure fill)."""
import os, json, warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = "/Volumes/Hard Disk/Microstructure Encoder Decoder"
DIAG = os.path.join(ROOT, "docs", "diagrams")
os.makedirs(DIAG, exist_ok=True)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- handwritten font
def pick_font():
    cands = []
    for d in (os.path.expanduser("~/Library/Fonts"), "/Library/Fonts",
              "/System/Library/Fonts", "/System/Library/Fonts/Supplemental"):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                if "handwritten" in f.lower():
                    cands.append(os.path.join(d, f))
    for p in cands:
        try:
            font_manager.fontManager.addfont(p)
            fam = font_manager.FontProperties(fname=p).get_name()
            print("handwritten font:", fam, "(", os.path.basename(p), ")")
            return fam
        except Exception:
            continue
    for fallback in ("Chalkboard SE", "Comic Sans MS", "Comic Neue", "Marker Felt"):
        if any(fallback.lower() in f.name.lower() for f in font_manager.fontManager.ttflist):
            print("fallback font:", fallback)
            return fallback
    print("no handwritten font found, xkcd default")
    return "xkcd"

FONT = pick_font()
plt.xkcd(scale=1.0, length=100, randomness=2)

INK = "#1e1e1e"
ARROW = "#404040"
BLUE, GREEN, YELLOW, RED, LILAC = "#a5d8ff", "#b2f2bb", "#ffec99", "#ffc9c9", "#d0bfff"
GREY = "#e9ecef"


def fresh(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100); ax.set_ylim(0, 100)
    ax.axis("off")
    return fig, ax


def box(ax, cx, cy, w, h, text, fc=BLUE, fs=9.5, dashed=False, lwidth=1.8):
    b = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.4,rounding_size=1.4",
                       linewidth=lwidth, edgecolor=INK,
                       facecolor=fc, linestyle="--" if dashed else "-",
                       mutation_aspect=1.0)
    ax.add_patch(b)
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontfamily=FONT, color=INK, linespacing=1.35)
    return (cx, cy, w, h)


def diamond(ax, cx, cy, w, h, text, fc=YELLOW, fs=9):
    ax.add_patch(plt.Polygon([(cx - w/2, cy), (cx, cy + h/2), (cx + w/2, cy), (cx, cy - h/2)],
                             closed=True, facecolor=fc, edgecolor=INK, linewidth=1.8))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, fontfamily=FONT)
    return (cx, cy, w, h)


def arrow(ax, p0, p1, label=None, rad=0.06, fs=8, color=ARROW, ls="-"):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                        color=color, linewidth=1.5, linestyle=ls,
                        connectionstyle=f"arc3,rad={rad}", zorder=1)
    ax.add_patch(a)
    if label:
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        ax.text(mx, my + 1.2, label, ha="center", va="bottom", fontsize=fs,
                fontfamily=FONT, color="#555555")


def title(ax, s):
    ax.text(50, 97.5, s, ha="center", va="top", fontsize=14, fontfamily=FONT,
            color=INK, fontweight="bold")


def note(ax, x, y, s, fs=8, color="#555555", ha="left"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=fs, fontfamily=FONT,
            color=color, linespacing=1.4)


def save(fig, name):
    png = os.path.join(DIAG, name + ".png")
    fig.savefig(png, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png)


# ================================================================ 1. BUILD WORKFLOW
fig, ax = fresh((13.5, 8.6))
title(ax, "PROJECT WORKFLOW: 8 gated phases; every phase reads the previous phase's measured report")

R1 = 76
p0 = box(ax, 11, R1, 17, 11,
         "PHASE 0 · Environment\nscripts/probe.py\nMPS · fp32 · I/O probe\nbatch sweep", GREEN)
p1 = box(ax, 36, R1, 19, 11,
         "PHASE 1 · De-leak re-split\nprep/build_manifest.py\nunion-find: md5+dHash+name\ndatabars, scale bands", GREEN)
p2 = box(ax, 62, R1, 17, 11,
         "PHASE 2 · Dataloader\nscripts/phase2_check.py\n128-px patches\ndihedral aug", GREEN)
p3 = box(ax, 87, R1, 16, 11,
         "PHASE 3 · Overfit gate\nscripts/phase3_overfit.py\n8 patches, wiring\nproof", GREEN)

R2 = 58
r0 = box(ax, 11, R2, 17, 6.5, "phase0_env.json\n192.8 img/s · 3.6×", GREY, fs=8)
r1 = box(ax, 36, R2, 19, 6.5, "phase1_data.json\n9,172 clusters · leak 0", GREY, fs=8)
r2 = box(ax, 62, R2, 17, 6.5, "phase2_data.json\n1,527 patches/s · 0 violations", GREY, fs=8)
r3 = box(ax, 87, R2, 16, 6.5, "phase3_overfit.json\n39.72 dB · gate FAIL¹", GREY, fs=8)

R3 = 36
p4 = box(ax, 15, R3, 21, 11,
         "PHASE 4 · Calibration\nsrc/train.py --steps 2000\nAdamW+EMA+cosine\n2.20 GB peak", BLUE)
p5 = box(ax, 47, R3, 22, 11,
         "PHASE 5 · Full training\nscripts/phase5_run.py\nto 5 h budget (30,797 steps)\nkill→resume PROVEN @ step 2,150", BLUE)
p6 = box(ax, 82, R3, 21, 11,
         "PHASE 6 · Evaluation\nsrc/eval.py + descriptors.py\n300 imgs, tiled 512/64\nMCRpy VF/S2/LP", BLUE)

R4 = 15
p7 = box(ax, 30, R4, 24, 10,
         "PHASE 7 · Prototype\nsrc/infer.py · src/app.py\nGradio, 8:1 @ fp16\nSHIPPED", GREEN)
p8 = box(ax, 72, R4, 24, 10,
         "PHASE 8 (optional) · PatchGAN\nsrc/train_gan.py + phase8_compare.py\nhinge + adaptive weight",
         LILAC, dashed=True)

for top, bot in ((p0, r0), (p1, r1), (p2, r2), (p3, r3)):
    arrow(ax, (top[0], top[1] - top[3]/2), (bot[0], bot[1] + bot[3]/2), rad=0)
for b in (r0, r1, r2, r3):
    arrow(ax, (b[0], b[1] - b[3]/2), (p4[0] + 4, p4[1] + p4[3]/2 + 1), rad=0.12)
arrow(ax, (p4[0] + p4[2]/2, p4[1]), (p5[0] - p5[2]/2, p5[1]),
      "reads budget", rad=0)
arrow(ax, (p5[0] + p5[2]/2, p5[1]), (p6[0] - p6[2]/2, p6[1]),
      "step-11,500 EMA weights", rad=0)
arrow(ax, (p6[0], p6[1] - p6[3]/2), (p7[0] + 6, p7[1] + p7[3]/2 + 1),
      "app + metrics", rad=0.1)
arrow(ax, (p5[0] + 6, p5[1] - p5[3]/2), (p8[0] - 8, p8[1] + p8[3]/2 + 1),
      "fine-tune", rad=0.12, ls="--")
arrow(ax, (p6[0] - 2, p6[1] - p6[3]/2), (p8[0] + 4, p8[1] + p8[3]/2 + 1),
      "keep/discard", rad=-0.1, ls="--")

note(ax, 1.5, 8.5,
     "CHAINING RULE: every phase writes reports/phase<N>_<name>.json:\n"
     "the next phase READS it. Never estimate what a phase measured.", fs=8.5)
note(ax, 99, 90, "¹ recorded FAIL, wiring proven (35 dB @ step 825); see gates table", fs=7.5, ha="right")
note(ax, 99, 5, "gate: KEEP only if LPIPS AND descriptors both improve; else discard (hallucination)",
     fs=7.5, ha="right")
save(fig, "workflow_handdrawn")

# ================================================================ 2. RUNTIME FLOW
fig, ax = fresh((12.5, 8.6))
title(ax, "RUNTIME DATA FLOW: one micrograph through the autoencoder")

x = 10
b1 = box(ax, x, 88, 16, 8, "micrograph.jpg\n(any H×W, grayscale)", GREY)
b2 = box(ax, x, 72, 16, 9, "centre-crop to\nmultiple of 8\nnormalize to [−1, 1]", GREY)
enc = box(ax, x, 53, 22, 14,
          "ENCODER  (f = 8)\nconv 3×3 → 32\n[2×ResBlock + down] ×3\n32 → 64 → 128 ch", BLUE)
mid = box(ax, x, 34, 22, 9,
          "mid block\nResBlock → self-attention\n→ ResBlock   @ H/8", BLUE)
z8 = box(ax, x, 19, 22, 8, "conv 3×3 → 8 ch\n(mean ‖ logvar)  @ H/8", BLUE)
z = box(ax, 42, 19, 17, 9, "latent z\n4 × H/8 × W/8\ne.g. 4×128×128", YELLOW)
out = box(ax, 42, 37, 17, 9, "sample = False\nlatent mean\n(deterministic)", GREY, fs=8.5)

dec = box(ax, 74, 34, 22, 14,
          "DECODER\nconv 3×3 → 128\nmid block\n[up + 3×ResBlock] ×3", GREEN)
up = box(ax, 74, 55, 22, 8, "upsample: nearest+conv\nNEVER transposed conv", GREEN, fs=8.5)
y = box(ax, 74, 76, 22, 8, "1-ch output, unbounded\nclamp to [−1, 1]", GREEN)
fin = box(ax, 74, 90, 22, 8, "recon PNG · latent.npy fp16\nmetrics JSON (PSNR/MS-SSIM/LPIPS)", YELLOW)

arrow(ax, (x, 84), (x, 76.5))
arrow(ax, (x, 67.5), (x, 60))
arrow(ax, (x, 46), (x, 38.5))
arrow(ax, (x, 29.5), (x, 23))
arrow(ax, (x + 11, 19), (z[0] - z[2]/2, 19), "encode: mode", rad=0)
arrow(ax, (z[0], 23.5), (out[0] - out[2]/2, 37), rad=-0.15)
arrow(ax, (out[0] + out[2]/2, 37), (dec[0] - dec[2]/2, 37), rad=0)
arrow(ax, (dec[0], 41), (up[0], 51))
arrow(ax, (up[0], 59), (y[0], 72))
arrow(ax, (y[0], 80), (fin[0], 86))
arrow(ax, (z[0], 14.5), (dec[0] - 6, 27), "decode z", rad=0.12)

note(ax, 30.5, 88, "residual zero-init (LDM/ADM): block ≡ identity at start", fs=8)
note(ax, 30.5, 72, "attention kept: measured 3% of step time (< 15% drop threshold)", fs=8)
note(ax, 30.5, 55, "logvar clamp [−30, 20]; bias init 0.0 (23.6 vs 21.9 dB @ 200 steps)", fs=8)
note(ax, 30.5, 8, "tanh is banned on output: it saturates 14.6% of micrograph pixels (|x|>0.9) and kills their gradient", fs=8)
note(ax, 84, 12, "8:1 compression @ fp16", fs=9)
save(fig, "runtime_handdrawn")

# ================================================================ 3. MAPPING / TRACEABILITY
fig, ax = fresh((13.0, 7.6))
title(ax, "PERFECT MAPPING: every artifact, measurement and decision traces end-to-end")

src = box(ax, 12, 80, 18, 12, "DATASET\n10,569 SEM images\n1024×1024 · 8-bit grey\n~2 GB", GREEN)
brief = box(ax, 12, 55, 18, 12, "BRIEF.md\nfixed decisions\nf=8 · KL-4ch · losses\nseed 1337 · 4 GB cap", YELLOW)
man = box(ax, 40, 80, 19, 12, "manifest.csv\nsplit · cluster · band\ncrop_top/bottom · keep", BLUE)
train = box(ax, 40, 55, 19, 12, "training\ntrain.py · phase5_run.py\nbatch 8 · EMA · AdamW", BLUE)
ckpt = box(ax, 68, 55, 17, 12, "checkpoints\ncalib (2,000)\nfull (11,500 / 11,796)", LILAC)
rep = box(ax, 68, 80, 17, 12, "reports/ JSONs\none per phase\ngates + measurements", GREY)
app = box(ax, 40, 25, 19, 11, "PROTOTYPE\ninfer.py · app.py\ntiled 512/64 · 8:1", GREEN)
ev = box(ax, 68, 25, 17, 11, "EVALUATION\nPSNR 26.60 · LPIPS .0984\ndesc 9.00% (<5% target)", RED)
docs = box(ax, 12, 25, 18, 11, "DOCUMENTATION\ndocs/ + .docx/.pdf\nreferences with DOIs", YELLOW)

arrow(ax, (21, 80), (man[0] - man[2]/2, 80), "build_manifest.py: cluster + de-leak", rad=0)
arrow(ax, (man[0], 74), (train[0], 61), "7,392/1,590/1,587 split\n6,387/1,373/1,402 kept", rad=0.08)
arrow(ax, (brief[0] + 9, 55), (train[0] - 9.5, 55), "config + hard rules", rad=0)
arrow(ax, (train[0] + 9.5, 57), (ckpt[0] - 8.5, 57), "checkpoints @500 steps", rad=0)
arrow(ax, (ckpt[0] - 4, 61), (rep[0] - 4, 74), "each run logs\nthroughput + memory", rad=-0.1)
arrow(ax, (man[0] + 9.5, 82), (rep[0] - 8.5, 82), "phase1 leak gate = 0", rad=0)
arrow(ax, (ckpt[0], 49), (ev[0], 30.5), "EMA weights step 11,500", rad=0.1)
arrow(ax, (ckpt[0] - 8.5, 52), (app[0] + 9.5, 28), "full_best.pt", rad=0.15)
arrow(ax, (rep[0] - 8.5, 77), (ev[0] + 6, 30.5), "phase4→6 chained numbers", rad=-0.25)
arrow(ax, (ev[0] - 8.5, 25), (docs[0] + 9, 25), "measured tables + verdicts", rad=0)
arrow(ax, (app[0] - 9.5, 27), (docs[0] + 9, 28), "verified: 29.91 dB upload", rad=0.1)

note(ax, 31, 92, "traps handled: 10,573 AppleDouble sidecars · 'val ' trailing space · exFAT", fs=7.5)
note(ax, 50, 8,
     "READ ACROSS: any number in the documentation ← reports/phase<N>_<name>.json ← the measured run.\n"
     "Nothing is estimated; gates are recorded as measured (two FAILs documented, never re-interpreted).",
     fs=8.5, ha="center")
save(fig, "mapping_handdrawn")

# ================================================================ .excalidraw sources
def ex_rect(x, y, w, h, bg, seed):
    return {"id": f"r{seed}", "type": "rectangle", "x": x, "y": y, "width": w, "height": h,
            "angle": 0, "strokeColor": "#1e1e1e", "backgroundColor": bg,
            "fillStyle": "hachure", "strokeWidth": 2, "strokeStyle": "solid",
            "roughness": 2, "seed": seed, "groupIds": [], "roundness": {"type": 3},
            "boundElements": None, "updated": 1, "link": None, "locked": False}


def ex_arrow(x, y, dx, dy, seed):
    return {"id": f"a{seed}", "type": "arrow", "x": x, "y": y, "width": abs(dx),
            "height": abs(dy), "angle": 0, "strokeColor": "#1e1e1e",
            "backgroundColor": "transparent", "fillStyle": "hachure", "strokeWidth": 2,
            "strokeStyle": "solid", "roughness": 2, "seed": seed, "groupIds": [],
            "roundness": {"type": 2}, "boundElements": None, "updated": 1, "link": None,
            "locked": False, "points": [[0, 0], [dx, dy]], "lastCommittedPoint": None,
            "startBinding": None, "endBinding": None,
            "startArrowhead": None, "endArrowhead": "arrow"}


def ex_text(x, y, text, size=16, seed=0):
    lines = text.split("\n")
    return {"id": f"t{seed}", "type": "text", "x": x, "y": y, "width": 200, "height": size * 1.25 * len(lines),
            "angle": 0, "strokeColor": "#1e1e1e", "backgroundColor": "transparent",
            "fillStyle": "hachure", "strokeWidth": 2, "strokeStyle": "solid", "roughness": 2,
            "seed": seed, "groupIds": [], "roundness": None, "boundElements": None,
            "updated": 1, "link": None, "locked": False,
            "text": text, "fontSize": size, "fontFamily": 1,
            "textAlign": "center", "verticalAlign": "middle",
            "containerId": None, "originalText": text, "lineHeight": 1.25}


def write_excalidraw(name, elements):
    data = {"type": "excalidraw", "version": 2,
            "source": "scripts/make_diagrams.py",
            "elements": elements,
            "appState": {"viewBackgroundColor": "#ffffff", "gridSize": None},
            "files": {}}
    p = os.path.join(DIAG, name + ".excalidraw")
    with open(p, "w") as f:
        json.dump(data, f, indent=1)
    print("wrote", p)


# phase workflow as editable excalidraw (simplified geometry, same content)
els, seed = [], 100
phases = [
    (0, 0, 260, 120, BLUE, "PHASE 0 · probe.py\nMPS · fp32 · I/O\n192.8 img/s · 3.6x"),
    (280, 0, 260, 120, BLUE, "PHASE 1 · build_manifest.py\nunion-find md5+dHash+name\n9,172 clusters · leak 0"),
    (560, 0, 260, 120, BLUE, "PHASE 2 · phase2_check.py\n128-px patches · dihedral\n1,527 patches/s"),
    (840, 0, 260, 120, BLUE, "PHASE 3 · phase3_overfit.py\n8-patch wiring proof\n39.72 dB · gate FAIL*"),
    (140, 180, 260, 120, GREEN, "PHASE 4 · train.py calib\n2,000 steps · 2.20 GB\n-> 30,797-step budget"),
    (420, 180, 280, 120, GREEN, "PHASE 5 · phase5_run.py\nkill->resume PROVEN\n11,796 steps · LPIPS .0770"),
    (720, 180, 280, 120, GREEN, "PHASE 6 · eval + descriptors\n300 imgs tiled 512/64\n26.60 dB · desc 9.00%"),
    (280, 360, 260, 110, YELLOW, "PHASE 7 · infer.py + app.py\nGradio prototype\n8:1 @ fp16 · SHIPPED"),
    (620, 360, 300, 110, LILAC, "PHASE 8 optional · train_gan.py\nPatchGAN hinge + adaptive w\nKEEP if LPIPS AND desc improve"),
]
for i, (x, y, w, h, bg, txt) in enumerate(phases):
    els.append(ex_rect(x, y, w, h, bg, seed)); seed += 1
    els.append(ex_text(x + 10, y + h / 2 - 20, txt, 14, seed)); seed += 1
flows = [(-30, 170, -60, -40), (530, 60, 0, 100), (830, 60, -90, 110),
         (620, 60, -200, 110), (400, 110, 0, 100)]
for dx, dy, px, py in flows:
    els.append(ex_arrow(px, py, dx, dy, seed)); seed += 1
els.append(ex_text(10, 500, "CHAINING RULE: every phase writes reports/phase<N>_<name>.json: the next phase READS it. Never estimate what a phase measured.", 14, seed))
write_excalidraw("workflow_handdrawn", els)

# runtime flow as editable excalidraw
els, seed = [], 300
rt = [
    (0, 0, 200, 80, GREY, "micrograph.jpg\nany H x W grey"),
    (0, 120, 200, 90, GREY, "centre-crop to /8\nnormalize [-1,1]"),
    (0, 250, 220, 110, BLUE, "ENCODER f=8\n[2xResBlock+down] x3\n32/64/128 + attn mid"),
    (0, 400, 220, 90, YELLOW, "mean || logvar -> z\n4 x H/8 x W/8\nsample=False"),
    (260, 250, 220, 110, GREEN, "DECODER\nmid + [up+3xResBlock] x3\nnearest+conv (no tconv)"),
    (260, 400, 220, 90, GREEN, "unbounded -> clamp\n[-1,1]"),
    (260, 530, 240, 90, YELLOW, "PNG + latent.npy fp16\nmetrics JSON"),
]
for x, y, w, h, bg, txt in rt:
    els.append(ex_rect(x, y, w, h, bg, seed)); seed += 1
    els.append(ex_text(x + 8, y + h / 2 - 18, txt, 13, seed)); seed += 1
for (px, py, dx, dy) in [(100, 80, 0, 40), (100, 210, 0, 40), (100, 360, 0, 40),
                         (220, 445, 40, -140), (370, 360, 0, 40), (370, 490, 0, 40)]:
    els.append(ex_arrow(px, py, dx, dy, seed)); seed += 1
els.append(ex_text(0, 660, "tanh banned on output (saturates 14.6% of pixels). 8:1 @ fp16.", 14, seed))
write_excalidraw("runtime_handdrawn", els)

print("done")
