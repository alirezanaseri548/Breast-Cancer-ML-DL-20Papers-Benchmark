# -*- coding: utf-8 -*-
"""
بنچمارک جامع الگوریتم‌های یادگیری ماشین روی داده‌های بالینی سرطان سینه
تلفیق ماژول‌های روز اول و روز دوم (مقالات ۱، ۲، ۶، ۹، ۱۰، ۱۱، ۱۲ و ۲۰):
- بارگذاری داده + نرمال‌سازی Z-Score + توازن کلاس با SMOTE
- مقایسه نهایی SVM، KNN، Logistic Regression، Random Forest، AdaBoost و Voting Ensemble
"""

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from src.data_loader import load_and_preprocess_wdbc
from src.smote_balancer import apply_smote
from src.ensemble_models import build_ensemble_models

def run_full_tabular_benchmark():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_wdbc()
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    models = {
        "Optimized RBF-SVM (Paper 1, 11)": SVC(C=3.0, gamma='scale', kernel='rbf', probability=True, random_state=42),
        "KNN (K=5, Euclidean - Paper 12)": KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    }
    models.update(build_ensemble_models())

    print("\n" + "="*88)
    print("COMPREHENSIVE TABULAR ML BENCHMARK (WDBC Dataset + Z-Score + SMOTE)")
    print("="*88)
    print(f"{'Model / Paper Reference':<38} | {'Accuracy':<9} | {'Error Rate':<10} | {'F1-Score':<8} | {'AUC':<6} | {'FN'}")
    print("-" * 88)

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, preds) * 100
        err = 100.0 - acc
        f1 = f1_score(y_test, preds)
        auc = roc_auc_score(y_test, probs)
        fn = confusion_matrix(y_test, preds)[1, 0]

        print(f"{name:<38} | {acc:>7.2f}% | {err:>8.2f}% | {f1:>8.4f} | {auc:>6.4f} | {fn:>2}")

if __name__ == "__main__":
    run_full_tabular_benchmark()