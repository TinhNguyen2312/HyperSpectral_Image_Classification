# -*- coding: utf-8 -*-
"""
Test script for utils/dataset.py
Verifies: load, normalization, GT labels, sample_gt splits, HSIDataset, DataLoader
Usage:
    python tests/test_dataset.py              # test all available datasets
    python tests/test_dataset.py ipnpy pu     # test specific datasets
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from utils.dataset import load_mat_hsi, sample_gt, HSIDataset

# ── ANSI colors ───────────────────────────────────────────────
OK   = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
HEAD = "\033[1;96m"
RST  = "\033[0m"

DATASET_DIR = os.path.join(os.path.dirname(__file__), "..", "datasets")

# Datasets that actually have files present
AVAILABLE = {
    "ipnpy": {"patch_size": 1,  "expected_classes": 16, "expected_bands": 200},
    "pu":    {"patch_size": 7,  "expected_classes": 9,  "expected_bands": 103},
    "sa":    {"patch_size": 7,  "expected_classes": 16, "expected_bands": 204},
    "whulk": {"patch_size": 7,  "expected_classes": 9,  "expected_bands": 270},
}

passed = 0
failed = 0

def ok(msg):
    global passed; passed += 1
    print(f"  {OK} {msg}")

def fail(msg):
    global failed; failed += 1
    print(f"  {FAIL} {msg}")

def warn(msg):
    print(f"  {WARN} {msg}")

def section(title):
    print(f"\n{HEAD}{'─'*60}{RST}")
    print(f"{HEAD}  {title}{RST}")
    print(f"{HEAD}{'─'*60}{RST}")


# ══════════════════════════════════════════════════════════════
# CHECK 1 – load_mat_hsi output types & shapes
# ══════════════════════════════════════════════════════════════
def check_load(name, cfg):
    section(f"Dataset: {name}")

    try:
        image, gt, labels = load_mat_hsi(name, DATASET_DIR)
    except Exception as e:
        fail(f"load_mat_hsi raised: {e}")
        return None, None, None

    H, W, C = image.shape

    # --- types ---
    if image.dtype == np.float32:
        ok(f"image.dtype = float32")
    else:
        fail(f"image.dtype = {image.dtype}  (expected float32)")

    if gt.dtype in (np.int32, np.int64, np.int_):
        ok(f"gt.dtype = {gt.dtype}")
    else:
        fail(f"gt.dtype = {gt.dtype}  (expected int)")

    # --- shape ---
    if image.ndim == 3:
        ok(f"image.shape = {image.shape}  (H×W×C)")
    else:
        fail(f"image.ndim = {image.ndim}  (expected 3)")

    if gt.shape == (H, W):
        ok(f"gt.shape    = {gt.shape}  matches image H×W")
    else:
        fail(f"gt.shape = {gt.shape}  ≠ image ({H},{W})")

    # --- expected bands ---
    exp_c = cfg["expected_bands"]
    if C == exp_c:
        ok(f"bands = {C}  (expected {exp_c})")
    else:
        warn(f"bands = {C}  (expected {exp_c} — may be corrected version)")

    # --- NaN / Inf ---
    if not np.isnan(image).any():
        ok("No NaN in image")
    else:
        fail(f"NaN found in image: {np.isnan(image).sum()} values")

    if not np.isinf(image).any():
        ok("No Inf in image")
    else:
        fail(f"Inf found in image: {np.isinf(image).sum()} values")

    # --- normalization: global [0,1] then mean-subtracted per channel ---
    # After step1 global norm → [0,1]; after step2 mean subtraction → mean≈0 per channel
    per_channel_mean = image.mean(axis=(0, 1))
    max_mean_dev = np.abs(per_channel_mean).max()
    if max_mean_dev < 1e-5:
        ok(f"Per-channel mean ≈ 0  (max dev = {max_mean_dev:.2e})")
    else:
        fail(f"Per-channel mean NOT zero  (max dev = {max_mean_dev:.4f}) — normalization bug?")

    img_min, img_max = image.min(), image.max()
    print(f"       image value range: [{img_min:.4f}, {img_max:.4f}]")

    # --- GT labels ---
    unique_gt = np.unique(gt)
    gt_min, gt_max = int(unique_gt.min()), int(unique_gt.max())

    if gt_min == -1:
        ok(f"GT min = -1  (background correctly mapped to -1)")
    else:
        fail(f"GT min = {gt_min}  (expected -1 for background)")

    exp_cls = cfg["expected_classes"]
    actual_cls = len(labels)
    if actual_cls == exp_cls:
        ok(f"labels list length = {actual_cls}  (expected {exp_cls})")
    else:
        fail(f"labels list length = {actual_cls}  (expected {exp_cls}): {labels}")

    if gt_max == exp_cls - 1:
        ok(f"GT max = {gt_max}  = n_classes-1  ✓ (0-indexed)")
    else:
        fail(f"GT max = {gt_max}  ≠ {exp_cls-1}  (class index mismatch)")

    # --- class pixel counts ---
    print(f"\n  {'Class':>4}  {'Name':<35} {'Pixels':>8}")
    print(f"  {'─'*55}")
    labeled_total = 0
    for cls_idx in range(exp_cls):
        cnt = int((gt == cls_idx).sum())
        labeled_total += cnt
        name_str = labels[cls_idx] if cls_idx < len(labels) else "???"
        marker = "  " if cnt > 0 else f"  {WARN}"
        print(f"  {cls_idx:>4}  {name_str:<35} {cnt:>8,}{marker}")
    bg_cnt = int((gt == -1).sum())
    print(f"  {'bg':>4}  {'Background (unlabeled)':<35} {bg_cnt:>8,}")
    print(f"  {'─'*55}")
    print(f"  {'':>4}  {'Labeled total':<35} {labeled_total:>8,}")

    zero_classes = [i for i in range(exp_cls) if (gt == i).sum() == 0]
    if zero_classes:
        fail(f"Classes with 0 pixels: {zero_classes}")
    else:
        ok("All classes have > 0 labeled pixels")

    return image, gt, labels


# ══════════════════════════════════════════════════════════════
# CHECK 2 – sample_gt splits
# ══════════════════════════════════════════════════════════════
def check_sample_gt(gt, labels, ratio=0.10, seed=42):
    section("sample_gt  (stratified split)")

    try:
        trainval_gt, test_gt = sample_gt(gt, ratio, seed)
        train_gt,   val_gt   = sample_gt(trainval_gt, 0.5, seed)
    except Exception as e:
        fail(f"sample_gt raised: {e}")
        return

    n_cls = len(labels)

    # No overlap
    trainval_mask = trainval_gt >= 0
    test_mask     = test_gt     >= 0
    overlap = (trainval_mask & test_mask).sum()
    if overlap == 0:
        ok("Train/test pixel overlap = 0  (no leakage)")
    else:
        fail(f"Train/test overlap = {overlap} pixels  (DATA LEAKAGE!)")

    # All labeled pixels accounted for
    total_labeled = int((gt >= 0).sum())
    split_total   = int(trainval_mask.sum()) + int(test_mask.sum())
    if split_total == total_labeled:
        ok(f"All {total_labeled:,} labeled pixels assigned to train or test")
    else:
        fail(f"Split total {split_total:,} ≠ labeled {total_labeled:,}  (pixels lost!)")

    # Stratification: every class present in both splits
    missing_train, missing_test = [], []
    for cls in range(n_cls):
        if (trainval_gt == cls).sum() == 0:
            missing_train.append(cls)
        if (test_gt == cls).sum() == 0:
            missing_test.append(cls)
    if not missing_train:
        ok("All classes present in trainval split")
    else:
        fail(f"Classes missing from trainval: {missing_train}")
    if not missing_test:
        ok("All classes present in test split")
    else:
        fail(f"Classes missing from test split: {missing_test}")

    # Ratio check  (allow ±2%)
    actual_ratio = trainval_mask.sum() / total_labeled
    if abs(actual_ratio - ratio) < 0.02:
        ok(f"Train ratio ≈ {actual_ratio:.3f}  (target {ratio:.2f})")
    else:
        fail(f"Train ratio = {actual_ratio:.3f}  (target {ratio:.2f}) — off by >{2}%")

    print(f"\n  Split summary:")
    print(f"    trainval : {int(trainval_mask.sum()):>7,} px  ({actual_ratio*100:.1f}%)")
    print(f"    test     : {int(test_mask.sum()):>7,} px  ({(1-actual_ratio)*100:.1f}%)")
    print(f"    train    : {int((train_gt>=0).sum()):>7,} px")
    print(f"    val      : {int((val_gt>=0).sum()):>7,} px")


# ══════════════════════════════════════════════════════════════
# CHECK 3 – HSIDataset & DataLoader
# ══════════════════════════════════════════════════════════════
def check_hsi_dataset(image, gt, labels, patch_size, dataset_name):
    section(f"HSIDataset + DataLoader  (patch_size={patch_size})")

    try:
        trainval_gt, test_gt = sample_gt(gt, 0.10, seed=42)
        train_gt, val_gt     = sample_gt(trainval_gt, 0.5, seed=42)
    except Exception as e:
        fail(f"sample_gt failed: {e}"); return

    try:
        train_set = HSIDataset(image, train_gt, patch_size=patch_size, data_aug=True,  return_dict=True)
        val_set   = HSIDataset(image, val_gt,   patch_size=patch_size, data_aug=False, return_dict=True)
        test_set  = HSIDataset(image, test_gt,  patch_size=patch_size, data_aug=False, return_dict=True)
    except Exception as e:
        fail(f"HSIDataset construction failed: {e}"); return

    ok(f"HSIDataset sizes → train={len(train_set)}, val={len(val_set)}, test={len(test_set)}")

    # Sample shape
    sample = train_set[0]
    inputs  = sample["inputs"]   # (1, C, P, P)
    targets = sample["targets"]  # scalar

    C = image.shape[-1]
    P = patch_size
    expected_shape = (1, C, P, P)
    if tuple(inputs.shape) == expected_shape:
        ok(f"inputs.shape = {tuple(inputs.shape)}  (1, C={C}, P={P}, P={P})")
    else:
        fail(f"inputs.shape = {tuple(inputs.shape)}  (expected {expected_shape})")

    if inputs.dtype == torch.float32:
        ok(f"inputs.dtype = float32")
    else:
        fail(f"inputs.dtype = {inputs.dtype}  (expected float32)")

    if targets.dtype == torch.int64:
        ok(f"targets.dtype = int64")
    else:
        fail(f"targets.dtype = {targets.dtype}  (expected int64)")

    n_cls = len(labels)
    if 0 <= int(targets.item()) < n_cls:
        ok(f"targets value = {targets.item()}  ∈ [0, {n_cls-1}]  ✓")
    else:
        fail(f"targets value = {targets.item()}  OUT OF [0, {n_cls-1}]  (label shift bug!)")

    # NaN / Inf in tensor
    if not torch.isnan(inputs).any():
        ok("No NaN in sample tensor")
    else:
        fail("NaN found in sample tensor!")

    if not torch.isinf(inputs).any():
        ok("No Inf in sample tensor")
    else:
        fail("Inf found in sample tensor!")

    # DataLoader batch
    loader = DataLoader(
        train_set, batch_size=32, shuffle=False,
        collate_fn=HSIDataset.data_collator
    )
    try:
        batch = next(iter(loader))
        b_inp = batch["inputs"]    # (32, 1, C, P, P)
        b_tgt = batch["targets"]   # (32,)
    except Exception as e:
        fail(f"DataLoader batch failed: {e}"); return

    if b_inp.shape == torch.Size([32, 1, C, P, P]):
        ok(f"DataLoader batch inputs shape = {list(b_inp.shape)}")
    else:
        fail(f"DataLoader batch inputs shape = {list(b_inp.shape)}  (expected [32,1,{C},{P},{P}])")

    if b_tgt.shape == torch.Size([32]):
        ok(f"DataLoader batch targets shape = {list(b_tgt.shape)}")
    else:
        fail(f"DataLoader batch targets shape = {list(b_tgt.shape)}")

    # Label range in full batch
    tgt_min, tgt_max = int(b_tgt.min()), int(b_tgt.max())
    if 0 <= tgt_min and tgt_max < n_cls:
        ok(f"Batch label range [{tgt_min}, {tgt_max}] ⊆ [0, {n_cls-1}]  ✓")
    else:
        fail(f"Batch label range [{tgt_min}, {tgt_max}] — out of [0, {n_cls-1}]!")

    # Class distribution in train set
    from collections import Counter
    label_counts = Counter(train_set.labels)
    missing = [i for i in range(n_cls) if label_counts.get(i, 0) == 0]
    if not missing:
        ok(f"All {n_cls} classes represented in train set")
    else:
        warn(f"Classes absent from train set: {missing}  (may be too-small ratio)")


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*",
                        help="Datasets to test (default: all with local files)")
    args = parser.parse_args()

    targets = args.datasets if args.datasets else list(AVAILABLE.keys())

    for ds_name in targets:
        if ds_name not in AVAILABLE:
            print(f"\n{WARN} '{ds_name}' not in test registry. Skipping.")
            continue

        cfg = AVAILABLE[ds_name]
        ds_dir = os.path.join(DATASET_DIR, ds_name)

        # Quick existence check before trying to load
        if ds_name == "ipnpy":
            data_exists = os.path.exists(os.path.join(DATASET_DIR, "Indian Pines", "indianpinearray.npy"))
        else:
            data_exists = os.path.isdir(ds_dir)

        if not data_exists:
            warn(f"Dataset '{ds_name}' files not found — skipping")
            continue

        image, gt, labels = check_load(ds_name, cfg)
        if image is None:
            continue

        check_sample_gt(gt, labels)
        check_hsi_dataset(image, gt, labels, patch_size=cfg["patch_size"], dataset_name=ds_name)

    # ── Summary ───────────────────────────────────────────────
    total = passed + failed
    print(f"\n{'═'*60}")
    print(f"  Results:  {passed}/{total} passed", end="")
    if failed:
        print(f"   ·  {FAIL} {failed} FAILED")
    else:
        print(f"   ·  {OK} All passed!")
    print(f"{'═'*60}\n")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
