# -*- coding: utf-8 -*-
"""
ماژول شبکه عصبی چندوجهی و مکانیزم توجه متقابل (Multimodal Fusion & Cross-Attention)
مرتبط با مقاله ۵ (Integrative Hybrid DL with Cross-Attention) و مقاله ۱۶ (Multimodal Imaging Model):
- مقاله ۱۶ نشان داد که ترکیب ویژگی‌های دو مدالیته ماموگرافی (MG) و اولتراسوند/سونوگرافی (US)
  به روش Intermediate Fusion باعث کاهش شدید بیوپسی‌های غیرضروری در بافت‌های متراکم می‌شود.
- مقاله ۵ از وزن‌دهی توجه (Attention Mechanism) برای برجسته کردن کانال‌های دارای بار اطلاعاتی بالاتر استفاده کرد.
"""

import numpy as np
from src.image_generator import generate_synthetic_breast_image
from src.cnn_layers import bcdcnn_multiscale_block

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def cross_attention_fusion(feat_mammogram, feat_ultrasound):
    """
    ادغام سطح میانی (Intermediate Fusion) با وزن‌دهی توجه متقابل (Cross-Attention)
    """
    v_mg = feat_mammogram.flatten()
    v_us = feat_ultrasound.flatten()

    # محاسبه امتیاز توجه (Attention Weights) بر اساس انرژی فعال‌سازی هر مدالیته
    score_mg = np.mean(v_mg ** 2)
    score_us = np.mean(v_us ** 2)
    exp_mg, exp_us = np.exp(score_mg), np.exp(score_us)
    alpha_mg = exp_mg / (exp_mg + exp_us)
    alpha_us = exp_us / (exp_mg + exp_us)

    # بردار ویژگی ادغام‌شده نهایی
    fused_vector = alpha_mg * v_mg + alpha_us * v_us
    malignancy_prob = sigmoid(12.0 * fused_vector.mean() - 2.5)

    return fused_vector, (alpha_mg, alpha_us), malignancy_prob

if __name__ == "__main__":
    mg_img, _ = generate_synthetic_breast_image(24, 24, has_tumor=True, random_state=10)
    us_img, _ = generate_synthetic_breast_image(24, 24, has_tumor=True, random_state=20)

    _, mg_pooled = bcdcnn_multiscale_block(mg_img)
    _, us_pooled = bcdcnn_multiscale_block(us_img)

    fused_vec, (w_mg, w_us), prob = cross_attention_fusion(mg_pooled, us_pooled)
    print("\n[Multimodal Fusion - Paper 5 & 16] Mammography + Ultrasound Intermediate Fusion:")
    print(f" - Cross-Attention Weights : Mammography={w_mg:.4f}, Ultrasound={w_us:.4f}")
    print(f" - Fused Feature Dimension : {fused_vec.shape[0]} features")
    print(f" - Predicted Malignancy    : {prob*100:.2f}% (Paper 16 Accuracy: 98.65%, Error: 1.35%)")