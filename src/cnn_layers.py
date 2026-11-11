# -*- coding: utf-8 -*-
"""
ماژول لایه‌های شبکه عصبی کانولوشنی و استخراج ویژگی چندمقیاسی (BCDCNN Architecture)
مرتبط با مقاله ۳ (Feature-Based CNN) و مقاله ۱۳ (BCDCNN Multi-Scale CNN):
- پیاده‌سازی کانولوشن دوبعدی با فیلترهای تشخیص لبه، بافت و گرادیان
- معماری چندمقیاسی (Multi-Scale Block): استخراج هم‌زمان ویژگی‌های ریز (میکروکلسیفیکاسیون)
  با کرنل 3x3 و ویژگی‌های کلان توده‌ای با کرنل 5x5
- اعمال تابع فعال‌ساز غیرخطی ReLU و کاهش ابعاد با Max-Pooling 2x2
"""

import numpy as np
from scipy.ndimage import convolve
from src.image_generator import generate_synthetic_breast_image

def relu(x):
    return np.maximum(0.0, x)

def max_pool2d(feature_map, pool_size=2):
    H, W = feature_map.shape
    H_trim, W_trim = H - (H % pool_size), W - (W % pool_size)
    trimmed = feature_map[:H_trim, :W_trim]
    return trimmed.reshape(H_trim // pool_size, pool_size, W_trim // pool_size, pool_size).max(axis=(1, 3))

def bcdcnn_multiscale_block(img):
    """
    بلوک استخراج ویژگی چندمقیاسی الهام‌گرفته از مقاله ۱۳ (BCDCNN, 2025):
    ترکیب پاسخ فیلتر لاپلاسین 3x3 (لبه‌های تیز) و فیلتر گوسی-مشتق 5x5 (ساختار کلی توده)
    """
    kernel_3x3_edge = np.array([
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
    ], dtype=float)

    kernel_5x5_macro = np.array([
        [-1, -1, -1, -1, -1],
        [-1,  1,  2,  1, -1],
        [-1,  2,  4,  2, -1],
        [-1,  1,  2,  1, -1],
        [-1, -1, -1, -1, -1]
    ], dtype=float) / 16.0

    feat_fine = relu(convolve(img, kernel_3x3_edge, mode='reflect'))
    feat_coarse = relu(convolve(img, kernel_5x5_macro, mode='reflect'))

    # نرمال‌سازی و ادغام کانال‌های چندمقیاسی (Concatenation / Fusion)
    feat_fine /= (feat_fine.max() + 1e-8)
    feat_coarse /= (feat_coarse.max() + 1e-8)
    fused_map = 0.6 * feat_fine + 0.4 * feat_coarse

    pooled_map = max_pool2d(fused_map, pool_size=2)
    return fused_map, pooled_map

if __name__ == "__main__":
    img, _ = generate_synthetic_breast_image(24, 24)
    fused, pooled = bcdcnn_multiscale_block(img)
    print(f"[BCDCNN Block - Paper 13] Input shape: {img.shape} -> Fused Feature Map: {fused.shape} -> Pooled: {pooled.shape}")
    print(f"[BCDCNN Block - Paper 13] Mean Multi-Scale Activation: {pooled.mean():.4f}")