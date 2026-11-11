# 🧬 تشخیص سرطان سینه با یادگیری ماشین و یادگیری عمیق (سنتز ۲۰ مقاله ژورنالی ۲۰۲۳-۲۰۲۵)
## Breast Cancer Diagnosis ML & DL Benchmark (Synthesis of 20 Journal Papers)

[![Author](https://img.shields.io/badge/Author-Alireza%20Naseri-blue.svg)](https://github.com/alirezanaseri548)
[![Status](https://img.shields.io/badge/Progress-Day%202%20of%205-yellowgreen.svg)]()
[![Papers](https://img.shields.io/badge/Reviewed%20Journals-20%20Papers%20(Nature%20%26%20IEEE)-success.svg)]()

این ریپازیتوری به بررسی، تحلیل نرخ خطا و پیاده‌سازی الگوریتم‌های ۲۰ مقاله معتبر ژورنالی (**Springer Nature** و **IEEE**) از سال ۲۰۲۳ تا ۲۰۲۵ در زمینه تشخیص سرطان سینه می‌پردازد.

### 📊 جدول کامل ۲۰ مقاله بررسی‌شده (Complete 20-Paper Benchmark)

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
| **11** | Detection of BC Using ML Optimization (2025) | Scientific Reports | GA & PSO Optimized SVM | 98.75% | **1.25%** | [Link](https://www.nature.com/articles/s41598-025-12644-w) |
| **12** | Enhancing BC Diagnosis Through ML (2025) | Scientific Reports | SVM, KNN (K=5), DT, RF | 98.95% | **1.05%** | [Link](https://www.nature.com/articles/s41598-025-07628-9) |
| **13** | BCDCNN: Deep Convolutional Network (2025) | Scientific Reports | Multi-Scale Custom CNN | 98.30% | **1.70%** | [Link](https://www.nature.com/articles/s41598-025-09974-0) |
| **14** | Explainable AI-Driven DNN (2025) | Scientific Reports | Multi-Layer DNN + LIME/SHAP | 97.90% | **2.10%** | [Link](https://www.nature.com/articles/s41598-025-97718-5) |
| **15** | Histopathological Image Based Diagnosis (2025) | Scientific Reports | EfficientNet + DenseNet | 99.05% | **0.95%** | [Link](https://www.nature.com/articles/s41598-025-04136-8) |
| **16** | Multimodal Imaging DL Model (2025) | Scientific Reports | Multimodal CNN Fusion | 98.65% | **1.35%** | [Link](https://www.nature.com/articles/s41598-025-99535-2) |
| **17** | Hybrid CNN with LSTM (Kaddes, 2025) | Scientific Reports | Sequential CNN-LSTM | 98.45% | **1.55%** | [Link](https://www.nature.com/articles/s41598-025-88459-6) |
| **18** | Hybrid CNN-BiLSTM Framework (Lilhore, 2025) | Scientific Reports | CNN-BiLSTM + Attention | 98.80% | **1.20%** | [Link](https://www.nature.com/articles/s41598-025-95311-4) |
| **19** | Parametric Survival vs ML Prognosis (2025) | Scientific Reports | Cox/Weibull + Deep Survival | 96.70% | **3.30%** | [Link](https://www.nature.com/articles/s41598-025-15696-0) |
| **20** | Early BC Detection via Data Analysis (2024) | IEEE Access | Statistical FE + SVM/DL | 98.62% | **1.38%** | [PDF](https://ieeexplore.ieee.org/iel8/6287639/10380310/10720756.pdf) |

---
### 🛠 ماژول‌های پیاده‌سازی شده تا پایان روز دوم (Implemented Modules)
- **Day 1:** data_loader.py (پیش‌پردازش Z-Score)، eature_selection.py (رتبه‌بندی بیومارکرها)، svm_optimizer.py (بهینه‌سازی کرنل RBF در SVM).
- **Day 2:** smote_balancer.py (توازن کلاس‌ها با SMOTE)، knn_classifier.py (تحلیل همسایگی KNN)، ensemble_models.py (مدل‌های ترکیبی RF, AdaBoost, Voting)، 	abular_benchmark.py (بنچمارک جامع داده‌های بالینی).
---
### 🖼 معماری‌های پردازش تصویر و یادگیری عمیق (Day 3: CNN & Multimodal Imaging)
در گام سوم پروژه، الگوریتم‌های مبتنی بر تصویر از مقالات **۳، ۵، ۸، ۱۳، ۱۵ و ۱۶** پیاده‌سازی شدند:
1. **`src/image_generator.py`**: شبیه‌سازی توده‌های پستانی، داده‌افزایی هندسی (مقاله ۳) و نرمال‌سازی رنگ پاتولوژی H&E (مقاله ۱۵).
2. **`src/cnn_layers.py`**: پیاده‌سازی بلوک کانولوشنی چندمقیاسی (**BCDCNN** - مقاله ۱۳) با ترکیب فیلترهای 3x3 و 5x5.
3. **`src/tumor_segmentation.py`**: استخراج ماسک تومور و ارزیابی همپوشانی مکانی با معیار **Dice Similarity Coefficient** (مقاله ۸).
4. **`src/multimodal_fusion.py`**: ادغام ویژگی‌های ماموگرافی و سونوگرافی با مکانیزم **Cross-Attention** (مقالات ۵ و ۱۶ با نرخ خطای ۱.۳۵٪).

---
### 🧠 شبکه‌های بازگشتی هیبریدی و هوش مصنوعی توضیح‌پذیر (Day 4: CNN-LSTM, BiLSTM & XAI)
در گام چهارم، معماری‌های ترتیبی و الگوریتم‌های تفسیرپذیر از مقالات **۴، ۸، ۱۴، ۱۷ و ۱۸** اضافه شدند:
1. **`src/cnn_lstm_hybrid.py`**: مدل‌سازی توالی اسلایس‌های تصویربرداری با ترکیب CNN و درگاه‌های حافظه LSTM (مقاله ۱۷ با نرخ خطای ۱.۵۵٪).
2. **`src/cnn_bilstm_attention.py`**: شبکه دوجهته **CNN-BiLSTM** مجهز به مکانیزم **Attention** برای تمرکز روی اسلایس‌های بحرانی تومور (مقاله ۱۸ با نرخ خطای ۱.۲۰٪).
3. **`src/image_registration.py`**: انطباق مکانی تصاویر پزشکی و بهبود معیارهای **MSE، PSNR و CC** (مقاله ۸).
4. **`src/explainable_ai.py`**: پیاده‌سازی شبکه عصبی عمیق (**DNN**) همراه با نقشه‌های حرارتی **Grad-CAM** و تحلیل اهمیت ویژگی‌های بالینی (مقالات ۴ و ۱۴).
