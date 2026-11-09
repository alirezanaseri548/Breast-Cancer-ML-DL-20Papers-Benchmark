# -*- coding: utf-8 -*-
"""
ماژول انتخاب ویژگی و رتبه‌بندی ضرایب تجمعی (Feature Selection)
مرتبط با مقاله ۹ (Integrated Feature Selection) و مقاله ۱۰ (Aggregated Coefficient Ranking):
- محاسبه ضریب همبستگی پیرسون بین هر ویژگی سلولی و برچسب بدخیمی تومور
- فیلتر کردن ابعاد زائد برای کاهش بار محاسباتی و جلوگیری از بیش‌برازش (Overfitting)
"""

import numpy as np
from src.data_loader import load_and_preprocess_wdbc

def rank_features_by_correlation(X_train, y_train, feature_names, top_k=10):
    correlations = []
    for i in range(X_train.shape[1]):
        corr = np.corrcoef(X_train[:, i], y_train)[0, 1]
        correlations.append((abs(corr), feature_names[i], i))

    # مرتب‌سازی نزولی بر اساس قدرمطلق همبستگی با بیماری
    correlations.sort(key=lambda x: x[0], reverse=True)
    return correlations[:top_k]

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, names = load_and_preprocess_wdbc()
    top_features = rank_features_by_correlation(X_tr, y_tr, names, top_k=8)
    print("\n[Paper 9 & 10] Top 8 Predictive Breast Cancer Biomarkers:")
    print("-" * 60)
    for rank, (score, fname, idx) in enumerate(top_features, 1):
        print(f"{rank}. {fname:<28} | Abs(Correlation) = {score:.4f}")