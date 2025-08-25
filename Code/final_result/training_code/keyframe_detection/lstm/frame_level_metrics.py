import os
import numpy as np
from glob import glob
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import load_model
import tensorflow as tf
from losses import mse_with_similarity_and_confidence_dynamic

# === 核心帧级评估函数 ===
def compute_frame_level_metrics(model_path, feature_files,
                                margin_exclude=10, threshold=0.5, verbose=True):
    """
    Evaluate frame-level classification metrics on sweep features.
    """

    # 加载模型并注册自定义损失函数（注意不加括号）
    model = load_model(model_path, custom_objects={
        'mse_with_sim_and_conf_dynamic': mse_with_similarity_and_confidence_dynamic
    })

    y_true_all = []
    y_pred_all = []

    for path in feature_files:
        data = np.load(path, allow_pickle=True).item()
        features = data["features"]
        labels = data["labels"]

        if np.max(labels) < 0.5:
            continue  # 跳过无关键帧样本

        T = len(labels)
        x = np.expand_dims(features, axis=0)
        pred = model.predict(x, verbose=0)[0]  # 输出 shape: (T,)

        # 只对结果进行评估时裁剪帧
        pred = pred[margin_exclude: T - margin_exclude]
        labels = labels[margin_exclude: T - margin_exclude]

        y_true_all.append(labels)
        y_pred_all.append(pred)

    if not y_true_all:
        print("[WARNING] No valid samples with keyframes.")
        return {}

    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)

    y_true_bin = (y_true_all >= 0.5).astype(int)
    y_pred_bin = (y_pred_all >= threshold).astype(int)

    results = {
        "ROC-AUC": roc_auc_score(y_true_bin, y_pred_all),
        "mAP": average_precision_score(y_true_bin, y_pred_all),
        "Precision": precision_score(y_true_bin, y_pred_bin, zero_division=0),
        "Recall": recall_score(y_true_bin, y_pred_bin, zero_division=0),
        "F1-score": f1_score(y_true_bin, y_pred_bin, zero_division=0),
    }

    if verbose:
        print("=== Frame-Level Evaluation ===")
        print(f"Samples evaluated: {len(feature_files)}")
        print(f"Excluded margin: {margin_exclude} frames")
        for k, v in results.items():
            print(f"{k:10}: {v:.4f}")

    return results

