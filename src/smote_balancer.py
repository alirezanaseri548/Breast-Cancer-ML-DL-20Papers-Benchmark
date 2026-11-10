# -*- coding: utf-8 -*-
"""
ماژول توازن‌ساز داده‌های پزشکی به روش درون‌یابی خطی (SMOTE-like Oversampling)
مرتبط با مقالات ۹، ۱۰ و ۲۰:
- در دیتاست‌های بالینی سرطان سینه، تعداد نمونه‌های بدخیم (Malignant) معمولاً کمتر از خوش‌خیم است.
- الگوریتم SMOTE با انتخاب یک نمونه از کلاس اقلیت (X_a) و یکی از نزدیک‌ترین همسایه‌های هم‌کلاس آن (X_b)،
  نمونه سنتزی جدیدی روی پاره‌خط واصل آن‌ها می‌سازد:
  X_new = X_a + lambda * (X_b - X_a) ,  lambda ~ Uniform(0, 1)
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.data_loader import load_and_preprocess_wdbc

def apply_smote(X, y, k_neighbors=5, random_state=42):
    rng = np.random.default_rng(random_state)
    classes, counts = np.unique(y, return_counts=True)
    if len(classes) != 2 or counts[0] == counts[1]:
        return X, y

    minority_class = classes[np.argmin(counts)]
    majority_count = np.max(counts)
    minority_count = np.min(counts)
    n_synthetic = majority_count - minority_count

    X_minority = X[y == minority_class]

    # یافتن K همسایه نزدیک در داخل کلاس اقلیت
    nn = NearestNeighbors(n_neighbors=min(k_neighbors + 1, len(X_minority)))
    nn.fit(X_minority)
    _, indices = nn.kneighbors(X_minority)

    synthetic_samples = []
    for _ in range(n_synthetic):
        idx = rng.integers(0, len(X_minority))
        sample_a = X_minority[idx]
        # انتخاب تصادفی یکی از همسایه‌های نزدیک (به‌جز خود نقطه که اندیس ۰ است)
        neighbor_idx = rng.choice(indices[idx][1:])
        sample_b = X_minority[neighbor_idx]

        lam = rng.uniform(0.0, 1.0)
        new_sample = sample_a + lam * (sample_b - sample_a)
        synthetic_samples.append(new_sample)

    X_balanced = np.vstack([X, np.array(synthetic_samples)])
    y_balanced = np.concatenate([y, np.full(n_synthetic, minority_class)])

    return X_balanced, y_balanced

if __name__ == "__main__":
    X_tr, _, y_tr, _, _ = load_and_preprocess_wdbc()
    u_before, c_before = np.unique(y_tr, return_counts=True)
    X_bal, y_bal = apply_smote(X_tr, y_tr)
    u_after, c_after = np.unique(y_bal, return_counts=True)

    print(f"[SMOTE Balancer] Class distribution BEFORE: {dict(zip(u_before, c_before))}")
    print(f"[SMOTE Balancer] Class distribution AFTER : {dict(zip(u_after, c_after))}")