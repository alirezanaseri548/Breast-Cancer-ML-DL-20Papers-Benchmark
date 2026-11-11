# -*- coding: utf-8 -*-
"""
ماژول شبکه ترکیبی دوجهته CNN-BiLSTM مجهز به مکانیزم توجه (Attention Mechanism)
مرتبط با مقاله ۱۸ (Lilhore et al., 2025 - Hybrid CNN-BiLSTM Framework):
- پردازش توالی ویژگی‌های بافتی در دو جهت پیش‌رو (Forward) و پس‌رو (Backward)
- اعمال لایه Attention جهت وزن‌دهی بیشتر به اسلایس‌های حاوی تراکم غیرعادی تومور
- دستاورد مقاله ۱۸: دقت ۹۸.۸۰٪ و کاهش نرخ خطا به ۱.۲۰٪.
"""

import numpy as np
from src.image_generator import generate_synthetic_breast_image
from src.cnn_layers import bcdcnn_multiscale_block
from src.cnn_lstm_hybrid import run_sequential_lstm, sigmoid

def run_bilstm_with_attention(feature_sequence):
    # مسیر پیش‌رو (Forward Pass) و مسیر پس‌رو (Backward Pass)
    fwd_states, _ = run_sequential_lstm(feature_sequence)
    bwd_states, _ = run_sequential_lstm(feature_sequence[::-1])
    bwd_states = bwd_states[::-1]

    # ترکیب حالات پیش‌رو و پس‌رو (Bi-directional Concatenation)
    bi_states = np.array(fwd_states) + np.array(bwd_states)

    # مکانیزم توجه (Softmax Attention Weights روی گام‌های زمانی)
    att_scores = np.exp(bi_states) / np.sum(np.exp(bi_states))
    context_vector = np.sum(att_scores * bi_states)

    malignancy_prob = sigmoid(2.5 * context_vector)
    return att_scores, context_vector, malignancy_prob

if __name__ == "__main__":
    img, _ = generate_synthetic_breast_image(24, 24, has_tumor=True)
    _, pooled = bcdcnn_multiscale_block(img)
    seq = pooled.mean(axis=1)

    att_weights, ctx, prob = run_bilstm_with_attention(seq)
    print(f"[CNN-BiLSTM-Attention - Paper 18] Peak Attention Slice Index: {np.argmax(att_weights)}")
    print(f"[CNN-BiLSTM-Attention - Paper 18] Context Vector: {ctx:.4f} | Malignancy Prob: {prob*100:.2f}%")