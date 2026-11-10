# -*- coding: utf-8 -*-
"""
ماژول الگوریتم‌های انسمبل و رای‌گیری یکپارچه (Ensemble Learning Framework)
مرتبط با مقالات ۲، ۶ و ۱۲:
- Random Forest (Bagging): آموزش موازی درخت‌های تصمیم روی زیرمجموعه‌های بوت‌استرپ جهت کاهش واریانس.
- AdaBoost (Boosting): آموزش ترتیبی درخت‌ها با افزایش وزن نمونه‌هایی که در مرحله قبل اشتباه طبقه‌بندی شده‌اند.
- Integrative Voting Framework (مقاله ۶): ترکیب ناهمگن SVM، KNN و Random Forest با رای‌گیری نرم (Soft Voting).
"""

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

def build_ensemble_models(random_state=42):
    rf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=random_state)
    ada = AdaBoostClassifier(n_estimators=60, learning_rate=0.8, random_state=random_state)
    lr = LogisticRegression(max_iter=1000, random_state=random_state)

    # چارچوب یکپارچه رای‌گیری وزنی مشابه مقاله ۶ (An integrative ML framework)
    svm_prob = SVC(C=3.0, gamma='scale', kernel='rbf', probability=True, random_state=random_state)
    knn_base = KNeighborsClassifier(n_neighbors=5, metric='euclidean')

    voting_integrative = VotingClassifier(
        estimators=[('svm', svm_prob), ('rf', rf), ('knn', knn_base), ('lr', lr)],
        voting='soft',
        weights=[3, 2, 1, 1]
    )

    return {
        "Logistic Regression (Paper 2)": lr,
        "Random Forest Bagging (Paper 2, 12)": rf,
        "AdaBoost Sequential (Paper 2, 6)": ada,
        "Integrative Soft-Voting (Paper 6)": voting_integrative
    }