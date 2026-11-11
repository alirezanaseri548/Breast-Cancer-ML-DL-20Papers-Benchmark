# -*- coding: utf-8 -*-
"""
ماژول قطعه‌بندی ناحیه تومور (Tumor Segmentation) و معیارهای ارزیابی مکانی
مرتبط با مقالات ۵، ۸ و ۱۵:
- استخراج ماسک باینری توده سرطانی از روی تصویر
- محاسبه ضریب تشابه دایس (Dice Similarity Coefficient):
  Dice = (2 * |Pred ∩ GT|) / (|Pred| + |GT|)
- محاسبه اشتراک روی اجتماع (Intersection over Union - IoU)
"""

import numpy as np
from src.image_generator import generate_synthetic_breast_image, normalize_he_stain

def segment_tumor_otsu_like(img, threshold=0.52):
    norm_img = normalize_he_stain(img)
    return (norm_img >= threshold).astype(float)

def compute_dice_and_iou(pred_mask, gt_mask):
    intersection = np.sum(pred_mask * gt_mask)
    total_area = np.sum(pred_mask) + np.sum(gt_mask)
    union = np.sum(np.clip(pred_mask + gt_mask, 0, 1))

    dice = (2.0 * intersection + 1e-8) / (total_area + 1e-8)
    iou = (intersection + 1e-8) / (union + 1e-8)
    return dice, iou

if __name__ == "__main__":
    img, gt_mask = generate_synthetic_breast_image(24, 24, has_tumor=True)
    pred_mask = segment_tumor_otsu_like(img, threshold=0.52)
    dice, iou = compute_dice_and_iou(pred_mask, gt_mask)

    print("\n[Tumor Segmentation - Paper 5 & 8] Spatial Overlap Evaluation:")
    print(f" - Ground Truth Tumor Area : {int(gt_mask.sum())} pixels")
    print(f" - Predicted Tumor Area    : {int(pred_mask.sum())} pixels")
    print(f" - Dice Similarity (DSC)   : {dice:.4f} (Paper 8 benchmark ~ 0.92)")
    print(f" - Jaccard Index (IoU)     : {iou:.4f}")