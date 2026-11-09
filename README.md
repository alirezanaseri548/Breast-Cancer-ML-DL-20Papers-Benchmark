# 🧬 تشخیص سرطان سینه با یادگیری ماشین و یادگیری عمیق (سنتز ۲۰ مقاله ژورنالی ۲۰۲۳-۲۰۲۵)
## Breast Cancer Diagnosis ML & DL Benchmark (Synthesis of 20 Journal Papers)

[![Author](https://img.shields.io/badge/Author-Alireza%20Naseri-blue.svg)](https://github.com/alirezanaseri548)
[![Status](https://img.shields.io/badge/Progress-Day%201%20of%205-orange.svg)]()

این ریپازیتوری به بررسی، تحلیل نرخ خطا و پیاده‌سازی الگوریتم‌های ۲۰ مقاله معتبر ژورنالی (**Springer Nature** و **IEEE**) از سال ۲۰۲۳ تا ۲۰۲۵ در زمینه تشخیص سرطان سینه می‌پردازد.

### 📊 بخش اول جدول مقالات بررسی‌شده (Papers 1 to 10)

| ردیف | عنوان و سال مقاله | ژورنال / ناشر | الگوریتم هسته | دقت | نرخ خطا | لینک |
| :---: | :--- | :--- | :--- | :---: | :---: | :---: |
| **1** | Optimized SVM for BC Detection (Bilal, 2024) | Scientific Reports | Quantum GWO + RBF-SVM | 98.60% | **1.40%** | [Link](https://www.nature.com/articles/s41598-024-61322-w) |
| **2** | Predictive Modeling for BC Classification (2024) | Scientific Reports | SVM, KNN, RF, XGBoost | 98.25% | **1.75%** | [Link](https://www.nature.com/articles/s41598-024-57740-5) |
| **3** | Feature-Based Detection Using CNN (2024) | Scientific Reports | Custom CNN + Transfer Learning | 97.80% | **2.20%** | [Link](https://www.nature.com/articles/s41598-024-73083-7) |
| **4** | Deep Learning & Explainable AI (2024) | Scientific Reports | DNN + Grad-CAM & SHAP | 98.10% | **1.90%** | [Link](https://www.nature.com/articles/s41598-024-80535-7) |
| **5** | Integrative Hybrid Deep Learning (2024) | Scientific Reports | ResNet + DenseNet Hybrid | 98.55% | **1.45%** | [Link](https://www.nature.com/articles/s41598-024-74305-8) |
| **6** | Integrative ML Framework for BC (2023) | Scientific Reports | Ensemble Voting (SVM, KNN) | 97.65% | **2.35%** | [Link](https://www.nature.com/articles/s41598-023-32029-1) |
| **7** | Histopathology Gene Expression Prediction (2023) | Scientific Reports | Deep CNN + Genomic Reg | 92.40% | **7.60%** | [Link](https://www.nature.com/articles/s41598-023-40219-0) |
| **8** | Enhancing Registration via LSTM (2024) | Scientific Reports | Spatio-Temporal CNN-LSTM | 97.10% | **2.90%** | [Link](https://www.nature.com/articles/s41598-024-82965-9) |
| **9** | Integrated Feature Selection & ML (2025) | Scientific Reports | mRMR + ReliefF + SVM/KNN | 99.12% | **0.88%** | [Link](https://www.nature.com/articles/s41598-025-97685-x) |
| **10** | Aggregated Coefficient Ranking (2025) | Scientific Reports | ACR Feature Ranking + SVM | 98.40% | **1.60%** | [Link](https://www.nature.com/articles/s41598-025-87826-7) |

---
### 🛠 ماژول‌های پیاده‌سازی شده در روز اول (Day 1 Modules)
- src/data_loader.py: بارگذاری دیتاست WDBC و استانداردسازی Z-Score بدون نشت داده (Data Leakage Prevention).
- src/feature_selection.py: انتخاب ویژگی و رتبه‌بندی بیومارکرها بر اساس ضریب همبستگی پیرسون (مقالات ۹ و ۱۰).
- src/svm_optimizer.py: بهینه‌سازی ماشین بردار پشتیبان (SVM) با کرنل RBF و تحلیل اثر $ و $\gamma$ (مقالات ۱، ۱۱ و ۱۲).