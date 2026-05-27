"""
Phân tích dữ liệu Indian Pines Hyperspectral Dataset
File: indianpinearray.npy  -> ảnh HSI (H x W x C)
File: IPgt.npy             -> ground truth labels (H x W)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import os

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets", "Indian Pines")

image = np.load(os.path.join(DATA_DIR, "indianpinearray.npy"))   # (H, W, C)
gt    = np.load(os.path.join(DATA_DIR, "IPgt.npy"))              # (H, W)

CLASS_NAMES = [
    "Undefined",                    # 0
    "Alfalfa",                      # 1
    "Corn-notill",                  # 2
    "Corn-mintill",                 # 3
    "Corn",                         # 4
    "Grass-pasture",                # 5
    "Grass-trees",                  # 6
    "Grass-pasture-mowed",          # 7
    "Hay-windrowed",                # 8
    "Oats",                         # 9
    "Soybean-notill",               # 10
    "Soybean-mintill",              # 11
    "Soybean-clean",                # 12
    "Wheat",                        # 13
    "Woods",                        # 14
    "Buildings-Grass-Trees-Drives", # 15
    "Stone-Steel-Towers",           # 16
]

# ─────────────────────────────────────────────
# 2. THỐNG KÊ CƠ BẢN
# ─────────────────────────────────────────────
H, W, C = image.shape
print("=" * 60)
print("  INDIAN PINES – THỐNG KÊ CƠ BẢN")
print("=" * 60)
print(f"  Kích thước ảnh  : {H} x {W} pixels")
print(f"  Số kênh phổ     : {C} bands")
print(f"  Dtype ảnh       : {image.dtype}")
print(f"  Giá trị min/max : {image.min():.4f} / {image.max():.4f}")
print(f"  Giá trị trung bình: {image.mean():.4f}")
print(f"  Độ lệch chuẩn   : {image.std():.4f}")
print()

# Phân phối nhãn
unique_labels, counts = np.unique(gt, return_counts=True)
print(f"  Số lớp (bao gồm nền): {len(unique_labels)}")
print()
print(f"  {'Lớp':<5} {'Tên lớp':<35} {'Số pixel':>10} {'%':>8}")
print("  " + "-" * 62)
total_labeled = counts[unique_labels > 0].sum()
for lbl, cnt in zip(unique_labels, counts):
    name = CLASS_NAMES[lbl] if lbl < len(CLASS_NAMES) else f"Class {lbl}"
    pct  = cnt / (H * W) * 100
    print(f"  {lbl:<5} {name:<35} {cnt:>10,} {pct:>7.2f}%")
print(f"\n  Tổng pixel có nhãn: {total_labeled:,} / {H*W:,}")

# ─────────────────────────────────────────────
# 3. VẼ ĐỒ THỊ
# ─────────────────────────────────────────────
fig = plt.figure(figsize=(20, 18))
fig.patch.set_facecolor("#0f1117")
plt.rcParams.update({
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "axes.edgecolor": "#444",
    "axes.facecolor": "#1a1d27",
    "figure.facecolor": "#0f1117",
})

COLORS_16 = [
    "#e6194b","#3cb44b","#ffe119","#4363d8","#f58231",
    "#911eb4","#42d4f4","#f032e6","#bfef45","#fabebe",
    "#469990","#e6beff","#9A6324","#fffac8","#800000","#aaffc3",
]

# ── 3a. False-Color RGB ──────────────────────
ax1 = fig.add_subplot(3, 3, 1)
# Dùng kênh 29, 19, 9 (approximate R-G-B cho AVIRIS)
r_idx, g_idx, b_idx = min(29, C-1), min(19, C-1), min(9, C-1)
rgb = image[:, :, [r_idx, g_idx, b_idx]].astype(np.float32)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
ax1.imshow(rgb)
ax1.set_title(f"False-Color RGB\n(Band {r_idx+1}, {g_idx+1}, {b_idx+1})", color="white", fontsize=11)
ax1.axis("off")

# ── 3b. Ground Truth Map ─────────────────────
ax2 = fig.add_subplot(3, 3, 2)
cmap_gt = ListedColormap(["#111111"] + COLORS_16)
im2 = ax2.imshow(gt, cmap=cmap_gt, vmin=0, vmax=16)
ax2.set_title("Ground Truth Map", color="white", fontsize=11)
ax2.axis("off")
patches = [mpatches.Patch(color=COLORS_16[i-1], label=f"{i}: {CLASS_NAMES[i]}")
           for i in range(1, len(CLASS_NAMES))]
ax2.legend(handles=patches, loc="upper right", bbox_to_anchor=(2.3, 1.02),
           fontsize=7, framealpha=0.4, facecolor="#1a1d27", edgecolor="#444",
           labelcolor="white", ncol=1)

# ── 3c. Phân phối số pixel theo lớp ─────────
ax3 = fig.add_subplot(3, 3, 3)
labeled_mask = unique_labels > 0
lbl_vals = unique_labels[labeled_mask]
lbl_cnts = counts[labeled_mask]
bars = ax3.barh(
    [CLASS_NAMES[l] for l in lbl_vals],
    lbl_cnts,
    color=[COLORS_16[i-1] for i in lbl_vals],
    edgecolor="none",
    height=0.7,
)
ax3.set_xlabel("Số pixel", color="white")
ax3.set_title("Phân phối lớp", color="white", fontsize=11)
ax3.tick_params(axis="y", labelsize=8)
for bar, cnt in zip(bars, lbl_cnts):
    ax3.text(cnt + 30, bar.get_y() + bar.get_height()/2,
             f"{cnt:,}", va="center", fontsize=7, color="white")

# ── 3d. Phổ trung bình theo lớp ──────────────
ax4 = fig.add_subplot(3, 3, 4)
for i, (lbl, color) in enumerate(zip(lbl_vals, COLORS_16)):
    mask = (gt == lbl)
    mean_spectrum = image[mask].mean(axis=0)
    ax4.plot(mean_spectrum, color=color, linewidth=1, alpha=0.85,
             label=CLASS_NAMES[lbl])
ax4.set_xlabel("Band index")
ax4.set_ylabel("Mean reflectance")
ax4.set_title("Phổ trung bình theo lớp", color="white", fontsize=11)
ax4.legend(fontsize=6, loc="upper right", framealpha=0.3,
           facecolor="#1a1d27", edgecolor="#444", labelcolor="white")

# ── 3e. Histogram giá trị pixel (3 kênh đại diện) ─
ax5 = fig.add_subplot(3, 3, 5)
sample_bands = [0, C//4, C//2, 3*C//4, C-1]
sample_colors = ["#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231"]
for band_idx, col in zip(sample_bands, sample_colors):
    ax5.hist(image[:, :, band_idx].ravel(), bins=60, color=col,
             alpha=0.55, label=f"Band {band_idx+1}")
ax5.set_xlabel("Pixel value")
ax5.set_ylabel("Count")
ax5.set_title("Histogram pixel (5 kênh đại diện)", color="white", fontsize=11)
ax5.legend(fontsize=8, framealpha=0.3, facecolor="#1a1d27",
           edgecolor="#444", labelcolor="white")

# ── 3f. Phương sai theo kênh phổ ─────────────
ax6 = fig.add_subplot(3, 3, 6)
band_var = image.reshape(-1, C).var(axis=0)
ax6.fill_between(range(C), band_var, color="#4363d8", alpha=0.6)
ax6.plot(band_var, color="#42d4f4", linewidth=1)
ax6.set_xlabel("Band index")
ax6.set_ylabel("Variance")
ax6.set_title("Phương sai theo kênh phổ", color="white", fontsize=11)

# ── 3g. Heatmap tương quan giữa các band (sample) ─
ax7 = fig.add_subplot(3, 3, 7)
step = max(1, C // 20)
sampled_bands = list(range(0, C, step))[:20]
pixels_flat = image.reshape(-1, C)[:, sampled_bands]
corr_matrix = np.corrcoef(pixels_flat.T)
im7 = ax7.imshow(corr_matrix, cmap="coolwarm", vmin=-1, vmax=1)
ax7.set_title(f"Tương quan band\n(mỗi {step} band lấy 1)", color="white", fontsize=11)
ax7.set_xlabel("Band index (sampled)")
ax7.set_ylabel("Band index (sampled)")
plt.colorbar(im7, ax=ax7, fraction=0.046, pad=0.04)

# ── 3h. Phổ min/max/mean toàn ảnh ────────────
ax8 = fig.add_subplot(3, 3, 8)
flat = image.reshape(-1, C)
band_mean = flat.mean(axis=0)
band_min  = flat.min(axis=0)
band_max  = flat.max(axis=0)
ax8.fill_between(range(C), band_min, band_max, alpha=0.3, color="#3cb44b", label="Min-Max range")
ax8.plot(band_mean, color="#ffe119", linewidth=1.5, label="Mean")
ax8.set_xlabel("Band index")
ax8.set_ylabel("Reflectance")
ax8.set_title("Dải phổ (min/mean/max)", color="white", fontsize=11)
ax8.legend(fontsize=9, framealpha=0.3, facecolor="#1a1d27",
           edgecolor="#444", labelcolor="white")

# ── 3i. Single band grayscale ─────────────────
ax9 = fig.add_subplot(3, 3, 9)
mid_band = C // 2
ax9.imshow(image[:, :, mid_band], cmap="inferno")
ax9.set_title(f"Single Band (Band {mid_band+1})", color="white", fontsize=11)
ax9.axis("off")

plt.suptitle("Indian Pines – Exploratory Data Analysis",
             color="white", fontsize=16, fontweight="bold", y=1.01)
plt.tight_layout()

out_path = os.path.join(os.path.dirname(__file__), "indian_pines_eda.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
print(f"\n  Đã lưu biểu đồ: {out_path}")
plt.show()
