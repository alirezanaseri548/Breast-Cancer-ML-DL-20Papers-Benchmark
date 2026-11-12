# -*- coding: utf-8 -*-
"""
اسکریپت جامع و یکپارچه اجرای کل پایپ‌لاین‌های ۲۰ مقاله ژورنالی سرطان سینه
Master Execution Script - Runs All Implemented Benchmarks Across All 20 Papers
Author: Alireza Naseri (https://github.com/alirezanaseri548)
"""

import numpy as np
from src.feature_selection import rank_features_by_correlation
from src.data_loader import load_and_preprocess_wdbc
from src.svm_optimizer import evaluate_svm_hyperparameters
from src.tabular_benchmark import run_full_tabular_benchmark
from src.image_generator import generate_synthetic_breast_image
from src.tumor_segmentation import segment_tumor_otsu_like, compute_dice_and_iou
from src.cnn_layers import bcdcnn_multiscale_block
from src.cnn_bilstm_attention import run_bilstm_with_attention
from src.image_registration import compute_registration_metrics, align_shifted_image
from src.explainable_ai import compute_dnn_feature_attribution
from src.gene_expression import predict_gene_expression_from_image
from src.survival_prognosis import evaluate_patient_survival_prognosis

def main():
    print("#" * 88)
    print("   BREAST CANCER AI BENCHMARK SUITE (SYNTHESIS OF 20 PEER-REVIEWED JOURNALS 2023-2025)")
    print("   Author: Alireza Naseri | GitHub: https://github.com/alirezanaseri548")
    print("#" * 88)

    X_tr, X_te, y_tr, y_te, names = load_and_preprocess_wdbc()
    top_feats = rank_features_by_correlation(X_tr, y_tr, names, top_k=5)
    print("\n>>> [1/6] TOP 5 BIOMARKERS (Papers 9 & 10):")
    for r, (sc, fn, _) in enumerate(top_feats, 1):
        print(f"    {r}. {fn:<26} (Correlation: {sc:.4f})")

    evaluate_svm_hyperparameters()
    run_full_tabular_benchmark()

    img, gt_mask = generate_synthetic_breast_image(24, 24, has_tumor=True)
    pred_mask = segment_tumor_otsu_like(img)
    dice, iou = compute_dice_and_iou(pred_mask, gt_mask)
    _, pooled = bcdcnn_multiscale_block(img)
    _, _, bilstm_prob = run_bilstm_with_attention(pooled.mean(axis=1))
    print(f"\n>>> [3/6] IMAGING & HYBRID CNN-BiLSTM (Papers 3, 5, 13, 15, 17, 18):")
    print(f"    - Tumor Segmentation Dice Score : {dice:.4f} | IoU: {iou:.4f}")
    print(f"    - CNN-BiLSTM-Attention Risk Prob: {bilstm_prob*100:.2f}%")

    shifted = np.roll(img, shift=(2, -1), axis=(0, 1))
    aligned, _ = align_shifted_image(img, shifted)
    mse_b, _, cc_b = compute_registration_metrics(img, shifted)
    mse_a, _, cc_a = compute_registration_metrics(img, aligned)
    print(f"\n>>> [4/6] IMAGE REGISTRATION (Paper 8):")
    print(f"    - Before Alignment: MSE={mse_b:.4f}, CC={cc_b:.4f} -> After Alignment: MSE={mse_a:.6f}, CC={cc_a:.4f}")

    dnn_acc, attrs = compute_dnn_feature_attribution(top_k=3)
    print(f"\n>>> [5/6] EXPLAINABLE AI & DNN (Papers 4 & 14):")
    print(f"    - DNN Accuracy: {dnn_acc:.2f}% (Error Rate: {100-dnn_acc:.2f}%) | Top Driver: {attrs[0][0]}")

    _, subtype = predict_gene_expression_from_image(img)
    rec_prob, surv_prob = evaluate_patient_survival_prognosis(58, 3.2, 2, 1)
    print(f"\n>>> [6/6] GENOMICS & SURVIVAL PROGNOSIS (Papers 7 & 19):")
    print(f"    - Histopathology Predicted Subtype : {subtype}")
    print(f"    - 5-Year Patient Survival Prob     : {surv_prob*100:.2f}% (Recurrence Risk: {rec_prob*100:.2f}%)")
    print("\n" + "#" * 88)
    print("   ALL 20-PAPER PIPELINES EXECUTED SUCCESSFULLY.")
    print("#" * 88)

if __name__ == "__main__":
    main()