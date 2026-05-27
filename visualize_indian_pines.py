import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# ─────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets", "Indian Pines"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "indian_pines_visualization.png"
)
TRAIN_RATIO = 0.10  # fraction used as training samples
SEED = 42

CLASS_NAMES = [
    "Alfalfa",  # 1
    "Corn-notill",  # 2
    "Corn-mintill",  # 3
    "Corn",  # 4
    "Grass-pasture",  # 5
    "Grass-trees",  # 6
    "Grass-pasture-mowed",  # 7
    "Hay-windrowed",  # 8
    "Oats",  # 9
    "Soybean-notill",  # 10
    "Soybean-mintill",  # 11
    "Soybean-clean",  # 12
    "Wheat",  # 13
    "Woods",  # 14
    "Buildings-grass-trees-drives",  # 15
    "Stone-steel-towers",  # 16
]

# Colors matching the reference figure legend (blue→green→yellow→orange→red→dark)
CLASS_COLORS = [
    "#00008B",  # 1  Alfalfa              - dark blue
    "#00008B",  # placeholder - overridden below with distinct colors
]
CLASS_COLORS = [
    "#00008B",  # 1  Alfalfa              - dark blue
    "#1E3A8A",  # 2  Corn-notill          - navy blue
    "#3B82F6",  # 3  Corn-mintill         - medium blue
    "#06B6D4",  # 4  Corn                 - cyan blue
    "#22D3EE",  # 5  Grass-pasture        - light cyan
    "#A5F3FC",  # 6  Grass-trees          - very light cyan
    "#6EE7B7",  # 7  Grass-pasture-mowed  - mint
    "#22C55E",  # 8  Hay-windrowed        - green
    "#A3E635",  # 9  Oats                 - lime
    "#EAB308",  # 10 Soybean-notill       - yellow
    "#F59E0B",  # 11 Soybean-mintill      - amber
    "#F97316",  # 12 Soybean-clean        - orange
    "#EF4444",  # 13 Wheat                - red-orange
    "#DC2626",  # 14 Woods                - red
    "#7F1D1D",  # 15 Buildings-grass...   - dark red/brown
    "#450A0A",  # 16 Stone-steel-towers   - very dark red
]

BG_COLOR = "#FFFFFF"  # background / unlabeled = white
N_CLASSES = 16

image = np.load(os.path.join(DATA_DIR, "indianpinearray.npy"))  # (H, W, C)
gt = np.load(os.path.join(DATA_DIR, "IPgt.npy"))  # (H, W), values 0–16

H, W, C = image.shape
print(f"  Image shape : {H} x {W} x {C}")
print(f"  GT   shape  : {gt.shape},  unique labels: {np.unique(gt)}")


def make_rgb(img, r=29, g=19, b=9):
    r = min(r, img.shape[2] - 1)
    g = min(g, img.shape[2] - 1)
    b = min(b, img.shape[2] - 1)
    rgb = img[:, :, [r, g, b]].astype(np.float32)
    # per-channel percentile stretch for better contrast
    lo, hi = np.percentile(rgb, (2, 98), axis=(0, 1))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb


rgb_image = make_rgb(image)

all_colors = [BG_COLOR] + CLASS_COLORS  # len = 17
cmap = ListedColormap(all_colors)
norm = BoundaryNorm(boundaries=np.arange(-0.5, 17.5, 1), ncolors=17)

fig = plt.figure(figsize=(12, 6), facecolor="white")

ax_a = fig.add_axes([0.03, 0.22, 0.28, 0.72])  # (a) RGB
ax_b = fig.add_axes([0.36, 0.22, 0.28, 0.72])  # (b) GT
# ax_c = fig.add_axes([0.69, 0.22, 0.28, 0.72])  # (c) Sample

ax_a.imshow(rgb_image, interpolation="nearest")
ax_a.axis("off")

ax_b.imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
ax_b.axis("off")

patches = [
    mpatches.Patch(
        facecolor=CLASS_COLORS[i], edgecolor="#555", linewidth=0.5, label=CLASS_NAMES[i]
    )
    for i in range(N_CLASSES)
]

legend = fig.legend(
    handles=patches,
    loc="lower left",
    bbox_to_anchor=(0.05, 0.19),
    ncol=6,
    fontsize=7.5,
    frameon=True,
    framealpha=0.9,
    edgecolor="#ccc",
    handlelength=1.5,
    handleheight=1.0,
    columnspacing=1.0,
    handletextpad=0.5,
    borderpad=0.6,
)

fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="white")
print(f"\nSaved: {OUTPUT_PATH}")
