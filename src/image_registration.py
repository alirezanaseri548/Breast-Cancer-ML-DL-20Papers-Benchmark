# -*- coding: utf-8 -*-
"""
ماژول انطباق مکانی تصاویر پزشکی (Medical Image Registration)
مرتبط با مقاله ۸ (Enhancing Registration Accuracy Using LSTM, 2024):
- اصلاح جابجایی‌های حرکتی بیمار در اسلایس‌های متوالی تصویربرداری سینه
- ارزیابی دقت هم‌پوشانی با سه معیار استاندارد:
  1. Mean Squared Error (MSE)
  2. Peak Signal-to-Noise Ratio (PSNR)
  3. Normalized Cross-Correlation (CC)
"""

import numpy as np
from src.image_generator import generate_synthetic_breast_image

def compute_registration_metrics(ref_img, target_img):
    mse = np.mean((ref_img - target_img) ** 2)
    psnr = 99.0 if mse < 1e-10 else 10.0 * np.log10(1.0 / mse)
    cc = np.corrcoef(ref_img.flatten(), target_img.flatten())[0, 1]
    return mse, psnr, cc

def align_shifted_image(ref_img, moving_img, max_shift=3):
    best_shift = (0, 0)
    best_cc = -1.0
    best_aligned = moving_img

    # جستجوی مکانی برای یافتن بهترین بردار انتقال (Translation Alignment)
    for dx in range(-max_shift, max_shift + 1):
        for dy in range(-max_shift, max_shift + 1):
            candidate = np.roll(moving_img, shift=(dx, dy), axis=(0, 1))
            _, _, cc = compute_registration_metrics(ref_img, candidate)
            if cc > best_cc:
                best_cc = cc
                best_shift = (dx, dy)
                best_aligned = candidate

    return best_aligned, best_shift

if __name__ == "__main__":
    base_img, _ = generate_synthetic_breast_image(24, 24, has_tumor=True)
    shifted_img = np.roll(base_img, shift=(2, -1), axis=(0, 1))

    mse_b, psnr_b, cc_b = compute_registration_metrics(base_img, shifted_img)
    aligned_img, shift_vec = align_shifted_image(base_img, shifted_img)
    mse_a, psnr_a, cc_a = compute_registration_metrics(base_img, aligned_img)

    print("\n[Image Registration - Paper 8] Alignment Performance:")
    print(f" - BEFORE Registration : MSE={mse_b:.4f}, PSNR={psnr_b:.2f} dB, CC={cc_b:.4f}")
    print(f" - AFTER  Registration : MSE={mse_a:.6f}, PSNR={psnr_a:.2f} dB, CC={cc_a:.4f} (Recovered Shift: {shift_vec})")