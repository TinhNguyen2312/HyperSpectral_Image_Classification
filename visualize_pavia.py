import os
import numpy as np
import matplotlib

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy import io

# ─────────────────────────────────────────────────────────────
# 1. CONFIG
# ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "datasets", "Pavia"
)
OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "pavia_visualization.png"
)
SEED = 42

CLASS_NAMES = [
    "Asphalt",              # 1
    "Meadows",              # 2
    "Gravel",               # 3
    "Trees",                # 4
    "Painted metal sheets", # 5
    "Bare Soil",            # 6
    "Bitumen",              # 7
    "Self-Blocking Bricks", # 8
    "Shadows",              # 9
]

# Standard colors used in Pavia University literature
CLASS_COLORS = [
    "#C0C0C0",  # 1  Asphalt              - gray
    "#00FF00",  # 2  Meadows              - bright green
    "#FF69B4",  # 3  Gravel               - pink
    "#006400",  # 4  Trees                - dark green
    "#00FFFF",  # 5  Painted metal sheets - cyan
    "#8B4513",  # 6  Bare Soil            - brown
    "#000080",  # 7  Bitumen              - dark navy
    "#FF8C00",  # 8  Self-Blocking Bricks - dark orange
    "#00008B",  # 9  Shadows              - dark blue
]

BG_COLOR = "#FFFFFF"  # background / unlabeled = white
N_CLASSES = 9

# ─────────────────────────────────────────────────────────────
# 2. LOAD DATA  (.mat format)
# ─────────────────────────────────────────────────────────────
mat_img = io.loadmat(os.path.join(DATA_DIR, "PaviaU.mat"))
image = mat_img["paviaU"].astype(np.float32)          # (610, 340, 103)

mat_gt = io.loadmat(os.path.join(DATA_DIR, "PaviaU_gt.mat"))
gt = mat_gt["paviaU_gt"].astype(np.int32)             # (610, 340), values 0–9

H, W, C = image.shape
print(f"  Image shape : {H} x {W} x {C}")
print(f"  GT    shape : {gt.shape},  unique labels: {np.unique(gt)}")

unique_labels, counts = np.unique(gt, return_counts=True)
print(f"\n  {'Class':<28} {'Pixels':>8}")
print("  " + "-" * 38)
for lbl, cnt in zip(unique_labels, counts):
    name = CLASS_NAMES[lbl - 1] if lbl > 0 else "Background"
    print(f"  {name:<28} {cnt:>8,}")


# ─────────────────────────────────────────────────────────────
# 3. FALSE-COLOR RGB
#    ROSIS sensor: band 1 ~ 430nm, spacing ~4nm, 103 bands → ~840nm
#    R=55 (~650nm), G=35 (~570nm), B=15 (~490nm)  (0-based)
# ─────────────────────────────────────────────────────────────
def make_rgb(img, r=55, g=35, b=15):
    r = min(r, img.shape[2] - 1)
    g = min(g, img.shape[2] - 1)
    b = min(b, img.shape[2] - 1)
    rgb = img[:, :, [r, g, b]].astype(np.float32)
    lo, hi = np.percentile(rgb, (2, 98), axis=(0, 1))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
    return rgb


rgb_image = make_rgb(image)

# ─────────────────────────────────────────────────────────────
# 4. COLOR MAP  (index 0 = white background, 1–9 = classes)
# ─────────────────────────────────────────────────────────────
all_colors = [BG_COLOR] + CLASS_COLORS   # len = 10
cmap = ListedColormap(all_colors)
norm = BoundaryNorm(boundaries=np.arange(-0.5, 10.5, 1), ncolors=10)

# ─────────────────────────────────────────────────────────────
# 5. PLOT  — 2 panels: (a) RGB  (b) GT  + legend below
# ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(10, 8), facecolor="white")

ax_a = fig.add_axes([0.03, 0.22, 0.44, 0.74])  # (a) RGB
ax_b = fig.add_axes([0.52, 0.22, 0.44, 0.74])  # (b) GT

ax_a.imshow(rgb_image, interpolation="nearest")
ax_a.set_title("(a)", fontsize=13, fontweight="bold", pad=6)
ax_a.axis("off")

ax_b.imshow(gt, cmap=cmap, norm=norm, interpolation="nearest")
ax_b.set_title("(b)", fontsize=13, fontweight="bold", pad=6)
ax_b.axis("off")

patches = [
    mpatches.Patch(
        facecolor=CLASS_COLORS[i], edgecolor="#555", linewidth=0.5,
        label=CLASS_NAMES[i]
    )
    for i in range(N_CLASSES)
]

fig.legend(
    handles=patches,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.0),
    ncol=5,
    fontsize=8.5,
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
