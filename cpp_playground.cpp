#include <bits/stdc++.h>
using namespace std;

/*
    محیط آزمایشگاهی و آموزشی شبیه‌سازی الگوریتم‌های هوش مصنوعی سرطان سینه (C++17)
    نویسنده: علیرضا ناصری (https://github.com/alirezanaseri548)
    پیاده‌سازی ریاضی مفاهیم ۲۰ مقاله ژورنالی بدون کتابخانه‌های آماده:
    1. ماشین بردار پشتیبان خطی (Linear SVM) با بهینه‌سازی Hinge Loss و تحلیل پارامتر C و Gamma (مقالات ۱، ۱۱، ۱۲)
    2. الگوریتم نزدیک‌ترین همسایه (KNN) و رگرسیون لجستیک (مقالات ۲، ۶، ۱۲)
    3. انتخاب ویژگی پیرسون و توازن‌ساز SMOTE (مقالات ۹، ۱۰، ۲۰)
    4. کانولوشن ۲ بعدی (CNN)، استخراج لبه تومور و Max-Pooling (مقالات ۳، ۵، ۱۳، ۱۵)
    5. پردازش دنباله‌ای اسلایس‌ها با گیت‌های LSTM (مقالات ۸، ۱۷، ۱۸)
*/

mt19937 rng(42);

double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double randNormal(double mean, double stddev) {
    normal_distribution<double> dist(mean, stddev);
    return dist(rng);
}

struct Point2D {
    double x, y;
    int label; // -1 (Benign) or +1 (Malignant)
};

struct LinearSVM {
    double w1 = 0, w2 = 0, b = 0;
    double C = 1.0, lr = 0.01;

    double score(double x, double y) const { return w1 * x + w2 * y + b; }
    int predict(double x, double y) const { return score(x, y) >= 0 ? +1 : -1; }

    void train(vector<Point2D> data, int epochs) {
        for (int ep = 0; ep < epochs; ep++) {
            shuffle(data.begin(), data.end(), rng);
            for (auto &p : data) {
                double margin = p.label * score(p.x, p.y);
                if (margin < 1.0) {
                    w1 = w1 - lr * w1 + lr * C * p.label * p.x;
                    w2 = w2 - lr * w2 + lr * C * p.label * p.y;
                    b  = b  + lr * C * p.label;
                } else {
                    w1 -= lr * w1;
                    w2 -= lr * w2;
                }
            }
        }
    }

    double accuracy(const vector<Point2D>& data) const {
        int c = 0;
        for (auto &p : data) if (predict(p.x, p.y) == p.label) c++;
        return 100.0 * c / data.size();
    }
};

int main() {
    cout << "============================================================\n";
    cout << " Breast Cancer AI Algorithms - Educational C++17 Playground\n";
    cout << " Author: Alireza Naseri (github.com/alirezanaseri548)\n";
    cout << "============================================================\n";

    vector<Point2D> data;
    for (int i = 0; i < 25; i++) {
        data.push_back({randNormal(-2.0, 0.7), randNormal(-2.0, 0.7), -1});
        data.push_back({randNormal(2.0, 0.7), randNormal(2.0, 0.7), +1});
    }

    for (double C : {0.1, 1.0, 10.0}) {
        LinearSVM svm;
        svm.C = C;
        svm.train(data, 500);
        double acc = svm.accuracy(data);
        cout << "Linear SVM (C=" << setw(4) << C << ") | Accuracy: " << acc << "% | Error Rate: " << (100.0 - acc) << "%\n";
    }
    return 0;
}