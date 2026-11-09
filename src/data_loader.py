# -*- coding: utf-8 -*-
"""
ماژول بارگذاری و پیش‌پردازش داده‌های سرطان سینه (WDBC Dataset)
مرتبط با متدولوژی پیش‌پردازش در مقالات ۲، ۶، ۹ و ۱۲:
- جداسازی لایه‌ای (Stratified Split) جهت حفظ نسبت تومورهای خوش‌خیم و بدخیم
- اعمال استانداردسازی Z-Score: Z = (X - Mean) / StdDev
- محاسبه میانگین و انحراف معیار صرفاً روی داده آموزش جهت جلوگیری از نشت داده (Data Leakage)
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_wdbc(test_size=0.2, random_state=42):
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names

    # تقسیم داده‌ها با حفظ توزیع کلاس‌ها
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # استانداردسازی Z-Score بدون نشت اطلاعات از مجموعه تست
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te, names = load_and_preprocess_wdbc()
    print(f"[Data Loader] Train shape: {X_tr.shape}, Test shape: {X_te.shape}")
    print(f"[Data Loader] Total features loaded: {len(names)}")