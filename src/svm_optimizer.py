# -*- coding: utf-8 -*-
"""
ماژول بهینه‌سازی ماشین بردار پشتیبان (SVM) و تحلیل کرنل RBF
مرتبط با مقاله ۱ (Bilal et al., 2024)، مقاله ۱۱ (GA/PSO-SVM, 2025) و مقاله ۱۲ (2025):
- بررسی تاثیر پارامتر جریمه C (توازن بین عرض مارجین و خطای طبقه‌بندی)
- بررسی تاثیر پارامتر شعاع کرنل Gamma در فرمول K(x, z) = exp(-gamma * ||x - z||^2)
- محاسبه دقیق نرخ خطا: Error Rate = 100 - Accuracy
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from src.data_loader import load_and_preprocess_wdbc

def evaluate_svm_hyperparameters():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_wdbc()

    configs = [
        {"C": 0.1,  "gamma": 0.01, "desc": "Small C (Wide Margin, Underfit risk)"},
        {"C": 1.0,  "gamma": "scale", "desc": "Balanced Default RBF-SVM"},
        {"C": 3.5,  "gamma": 0.02, "desc": "Optimized RBF-SVM (Paper 1 & 11 Style)"},
        {"C": 50.0, "gamma": 0.5,  "desc": "Large C & Gamma (Overfit risk)"}
    ]

    print("\n[Paper 1, 11 & 12] SVM Hyperparameter Optimization (C & Gamma) Benchmark:")
    print(f"{'Configuration Description':<40} | {'Accuracy':<9} | {'Error Rate':<10} | {'F1-Score':<8} | {'FN (Missed)'}")
    print("-" * 88)

    for cfg in configs:
        model = SVC(C=cfg["C"], gamma=cfg["gamma"], kernel="rbf", random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds) * 100
        err_rate = 100.0 - acc
        f1 = f1_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        fn = cm[1, 0] # موارد منفی کاذب

        print(f"{cfg['desc']:<40} | {acc:>7.2f}% | {err_rate:>8.2f}% | {f1:>8.4f} | {fn:>5}")

if __name__ == "__main__":
    evaluate_svm_hyperparameters()