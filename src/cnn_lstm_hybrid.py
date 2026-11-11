# -*- coding: utf-8 -*-
"""
ماژول شبکه عصبی ترکیبی کانولوشنی و حافظه طولانی کوتاه‌مدت (Hybrid CNN-LSTM)
مرتبط با مقاله ۸ (DCE-MRI LSTM) و مقاله ۱۷ (Kaddes et al., 2025 - Hybrid CNN with LSTM):
- شبکه CNN ابتدا ویژگی‌های مکانی (Spatial Features) هر اسلایس را استخراج می‌کند.
- سپس سلول‌های حافظه LSTM با درگاه‌های Forget, Input و Output پیوستگی بین اسلایس‌ها را مدل می‌کنند.
- نرخ خطای گزارش‌شده در مقاله ۱۷ با این معماری: ۱.۵۵٪ (دقت ۹۸.۴۵٪).
"""

import numpy as np
from src.image_generator import generate_synthetic_breast_image
from src.cnn_layers import bcdcnn_multiscale_block

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def run_sequential_lstm(feature_sequence):
    h, c = 0.0, 0.0
    hidden_history = []

    for x_t in feature_sequence:
        f_gate = sigmoid(1.2 * x_t + 0.5 * h)      # Forget Gate
        i_gate = sigmoid(1.0 * x_t - 0.2 * h)      # Input Gate
        c_tilde = np.tanh(1.5 * x_t + 0.3 * h)     # Candidate State
        o_gate = sigmoid(0.8 * x_t + 0.4 * h)      # Output Gate

        c = f_gate * c + i_gate * c_tilde
        h = o_gate * np.tanh(c)
        hidden_history.append(h)

    malignancy_prob = sigmoid(3.2 * h)
    return hidden_history, malignancy_prob

if __name__ == "__main__":
    img, _ = generate_synthetic_breast_image(24, 24, has_tumor=True)
    _, pooled = bcdcnn_multiscale_block(img)
    row_sequence = pooled.mean(axis=1)

    history, prob = run_sequential_lstm(row_sequence)
    print(f"[Hybrid CNN-LSTM - Paper 17] Processed {len(row_sequence)} spatial slices sequentially.")
    print(f"[Hybrid CNN-LSTM - Paper 17] Final Hidden State: {history[-1]:.4f} | Malignancy Prob: {prob*100:.2f}%")