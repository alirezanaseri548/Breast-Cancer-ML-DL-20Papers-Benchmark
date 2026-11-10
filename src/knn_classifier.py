# -*- coding: utf-8 -*-
"""
ماژول طبقه‌بندی نزدیک‌ترین K-همسایه (KNN Classifier)
مرتبط با مقالات ۲، ۶، ۹، ۱۰ و ۱۲:
- مقاله ۱۲ نشان داد که در مقدار K=5 با معیار فاصله اقلیدسی، دقت KNN روی دیتاست ویسکانسین به ۹۷.۳۷٪ می‌رسد.
- این ماژول اثر تغییر K (از ۱ تا ۱۱) و معیار فاصله (Euclidean در برابر Manhattan) را بر نرخ خطا می‌سنجد.
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from src.data_loader import load_and_preprocess_wdbc
from src.smote_balancer import apply_smote

def evaluate_knn_variations():
    X_train, X_test, y_train, y_test, _ = load_and_preprocess_wdbc()
    X_train_bal, y_train_bal = apply_smote(X_train, y_train)

    print("\n[Paper 2 & 12] KNN Sensitivity Analysis (Effect of K and Distance Metric):")
    print(f"{'K-Value':<10} | {'Metric':<12} | {'Accuracy':<10} | {'Error Rate':<12} | {'F1-Score'}")
    print("-" * 65)

    for metric in ["euclidean", "manhattan"]:
        for k in [1, 3, 5, 7, 9]:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train_bal, y_train_bal)
            preds = knn.predict(X_test)

            acc = accuracy_score(y_test, preds) * 100
            err = 100.0 - acc
            f1 = f1_score(y_test, preds)
            print(f"K = {k:<6} | {metric:<12} | {acc:>8.2f}% | {err:>10.2f}% | {f1:.4f}")

if __name__ == "__main__":
    evaluate_knn_variations()