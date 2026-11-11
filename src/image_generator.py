# -*- coding: utf-8 -*-
"""
ماژول شبیه‌ساز تصاویر ماموگرافی و هیستوپاتولوژی به همراه داده‌افزایی (Data Augmentation)
مرتبط با مقالات ۳ و ۱۵:
- مقاله ۳: استفاده از Augmentation هندسی (چرخش و قرینه‌سازی) برای جبران کمبود تصاویر پزشکی.
- مقاله ۱۵: اعمال نرمال‌سازی رنگ هماتوکسیلین-ائوزین (H&E Stain Normalization) روی لام‌های بیوپسی
  جهت حذف تغییرات نوری آزمایشگاه‌های مختلف.
"""

import numpy as np

def generate_synthetic_breast_image(height=24, width=24, has_tumor=True, random_state=42):
    rng = np.random.default_rng(random_state)
    y_grid, x_grid = np.ogrid[:height, :width]

    # بافت زمینه طبیعی سینه با نویز گوسی ملایم
    img = rng.normal(0.18, 0.04, (height, width))
    ground_truth_mask = np.zeros((height, width), dtype=float)

    if has_tumor:
        cx, cy = height / 2.0 + rng.uniform(-2, 2), width / 2.0 + rng.uniform(-2, 2)
        radius = rng.uniform(3.5, 5.0)
        dist = np.sqrt((x_grid - cx)**2 + (y_grid - cy)**2)

        # ایجاد هسته متراکم تومور و لبه‌های نامنظم (Spiculated Margin)
        tumor_region = dist <= radius
        ground_truth_mask[tumor_region] = 1.0
        img[tumor_region] += rng.normal(0.68, 0.06, size=np.sum(tumor_region))

    return np.clip(img, 0.0, 1.0), ground_truth_mask

def augment_image(img):
    """داده‌افزایی هندسی شامل چرخش ۹۰ درجه و قرینه‌سازی افقی/عمودی (مقاله ۳)"""
    return {
        "original": img,
        "flipped_h": np.fliplr(img),
        "flipped_v": np.flipud(img),
        "rotated_90": np.rot90(img)
    }

def normalize_he_stain(img):
    """شبیه‌سازی نرمال‌سازی کنتراست و رنگ‌آمیزی H&E در تصاویر پاتولوژی (مقاله ۱۵)"""
    p2, p98 = np.percentile(img, (2, 98))
    norm_img = np.clip((img - p2) / (p98 - p2 + 1e-8), 0.0, 1.0)
    return norm_img

if __name__ == "__main__":
    img, mask = generate_synthetic_breast_image()
    aug_dict = augment_image(img)
    norm_img = normalize_he_stain(img)
    print(f"[Image Generator] Generated {img.shape} image | Tumor pixels: {int(mask.sum())}")
    print(f"[Augmentation - Paper 3] Created {len(aug_dict)} geometric variations.")
    print(f"[Stain Norm - Paper 15] Normalized intensity range: [{norm_img.min():.2f}, {norm_img.max():.2f}]")