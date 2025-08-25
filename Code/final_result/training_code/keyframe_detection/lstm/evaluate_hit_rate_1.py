# import numpy as np
# import matplotlib.pyplot as plt
# from tensorflow.keras.models import load_model
# from scipy.ndimage import gaussian_filter1d
# from scipy.signal import savgol_filter
# from losses import mse_with_similarity_and_confidence_dynamic
# import os

# def evaluate_keyframe_hit_rate(model_path, val_feature_files, score_threshold=0.3,
#                                smooth_method='gaussian', exclude_margin=10,
#                                save_dir=None, max_visualize=10):
#     # 加载模型
#     model = load_model(model_path, custom_objects={
#         'mse_with_sim_and_conf_dynamic': mse_with_similarity_and_confidence_dynamic
#     })

#     total_predicted = 0
#     total_hit = 0
#     all_hit_ratios = []
#     valid_file_count = 0

#     if save_dir:
#         os.makedirs(save_dir, exist_ok=True)

#     for idx, npy_path in enumerate(val_feature_files):
#         data = np.load(npy_path, allow_pickle=True).item()
#         labels = data["labels"]
#         total_frames = len(labels)

#         # 忽略没有关键帧的样本
#         if np.max(labels) < 0.5:
#             continue

#         valid_file_count += 1
#         x = np.expand_dims(data["features"], axis=0)
#         y_pred = model.predict(x, verbose=0)[0]

#         # 平滑
#         if smooth_method == 'gaussian':
#             y_pred = gaussian_filter1d(y_pred, sigma=2)
#         elif smooth_method == 'savgol':
#             y_pred = savgol_filter(y_pred, window_length=11, polyorder=3)

#         # 有效帧范围
#         valid_range = set(range(exclude_margin, total_frames - exclude_margin))

#         # 预测关键帧
#         pred_indices = [i for i in valid_range if y_pred[i] >= score_threshold]

#         # 命中帧
#         hit_indices = [i for i in pred_indices if labels[i] >= 0.5]

#         total_hit += len(hit_indices)
#         total_predicted += len(pred_indices)
#         hit_ratio = len(hit_indices) / len(pred_indices) if pred_indices else 0.0
#         all_hit_ratios.append(hit_ratio)

#         # ===== 可视化效果图（证明效果） =====
#         if save_dir and idx < max_visualize:
#             plt.figure(figsize=(9, 4))
            
#             # 绘制真实标签曲线
#             plt.plot(labels, label="Ground Truth", marker='o', color='black', alpha=0.7)
#             # 绘制预测分数曲线
#             plt.plot(y_pred, label="Prediction", linestyle='--', color='blue', linewidth=1.5)
#             # 标记预测关键帧
#             plt.scatter(pred_indices, [y_pred[i] for i in pred_indices],
#                         color='red', marker='x', s=60, label='Predicted')
#             # 标记命中帧
#             if hit_indices:
#                 plt.scatter(hit_indices, [y_pred[i] for i in hit_indices],
#                             facecolors='none', edgecolors='green', marker='o',
#                             s=80, linewidth=2, label='Hit')

#             # 边界标记
#             plt.axvline(exclude_margin, color='gray', linestyle=':', alpha=0.5)
#             plt.axvline(total_frames - exclude_margin - 1, color='gray', linestyle=':', alpha=0.5)

#             plt.title(f"{os.path.basename(npy_path)} | Hit Ratio: {hit_ratio:.2f}")
#             plt.xlabel("Frame Index")
#             plt.ylabel("Score")
#             plt.legend()
#             plt.grid(True, alpha=0.3)
#             plt.tight_layout()
#             plt.savefig(os.path.join(save_dir, f"{os.path.basename(npy_path)}.png"), dpi=300)
#             plt.close()

#     if valid_file_count == 0:
#         print("[WARNING] No valid files with key frames ≥ 0.5 found.")
#         return 0.0, 0.0

#     overall_ratio = total_hit / total_predicted if total_predicted > 0 else 0.0
#     avg_per_file_ratio = np.mean(all_hit_ratios)

#     print("\n=== Key Frame Hit Rate Evaluation ===")
#     print(f"Valid Files Evaluated : {valid_file_count}")
#     print(f"Threshold             : score ≥ {score_threshold}")
#     print(f"Margin Excluded       : {exclude_margin} frames")
#     print(f"Total Predicted Frames: {total_predicted}")
#     print(f"Total Hits (GT ≥ 0.5) : {total_hit}")
#     print(f"Overall Hit Ratio     : {overall_ratio:.4f}")
#     print(f"Avg Hit Ratio/File    : {avg_per_file_ratio:.4f}")
#     if save_dir:
#         print(f"Visualizations saved to: {save_dir}")

#     return overall_ratio, avg_per_file_ratio
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from losses import mse_with_similarity_and_confidence_dynamic
import os
import random

def evaluate_keyframe_hit_rate(
    model_path,
    val_feature_files,
    score_threshold=0.3,
    smooth_method='gaussian',
    exclude_margin=10,
    save_dir=None,
    max_visualize=10,
    # 新增的轻量参数：
    shuffle=True,
    seed=0,
    match_tolerance=2  # 命中容差：与任意GT关键帧相差≤tolerance算命中
):
    """
    评估关键帧命中率 + 生成可视化（可复现随机抽样）+ 报告 Recall@K 和 Hit@1。
    - 不改变你原有输出的 overall_ratio / avg_per_file_ratio
    - 仅新增打印 Recall@K 和 Hit@1，和可选的可视化与随机打乱
    """

    # 可复现随机打乱（仅一次）
    files = list(val_feature_files)
    if shuffle:
        random.Random(seed).shuffle(files)

    # 加载模型
    model = load_model(model_path, custom_objects={
        'mse_with_sim_and_conf_dynamic': mse_with_similarity_and_confidence_dynamic
    })

    total_predicted = 0         # micro：所有预测关键帧个数
    total_hit = 0               # micro：命中总数
    all_hit_ratios = []         # macro：每文件的命中率（pred>阈值的帧里，有多少命中）
    valid_file_count = 0

    # 用于新版指标（按文件统计）
    per_file_recall_at_k = []   # 每文件 Recall@K
    per_file_hit1 = []          # 每文件 Hit@1（top1是否命中）
    visualized = 0

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for idx, npy_path in enumerate(files):
        data = np.load(npy_path, allow_pickle=True).item()
        labels = data["labels"]
        total_frames = len(labels)

        # 忽略没有关键帧的样本（GT<0.5）
        if np.max(labels) < 0.5:
            continue

        valid_file_count += 1
        x = np.expand_dims(data["features"], axis=0)
        y_pred = model.predict(x, verbose=0)[0]

        # 平滑
        if smooth_method == 'gaussian':
            y_pred = gaussian_filter1d(y_pred, sigma=2)
        elif smooth_method == 'savgol':
            # 防止长度过短时 window 报错
            win = min(11, (total_frames // 2) * 2 - 1) if total_frames >= 13 else None
            if win and win >= 5:
                y_pred = savgol_filter(y_pred, window_length=win, polyorder=3)

        # 有效帧范围（排除前后）
        valid_range = set(range(exclude_margin, total_frames - exclude_margin))

        # 预测关键帧（阈值法）
        pred_indices = [i for i in valid_range if y_pred[i] >= score_threshold]

        # GT关键帧索引（≥0.5）且在有效范围
        gt_indices = [i for i in range(total_frames) if (labels[i] >= 0.5 and i in valid_range)]

        # 命中统计（带容差）
        def is_hit(i_pred, gt_list, tol):
            for g in gt_list:
                if abs(i_pred - g) <= tol:
                    return True
            return False

        hits = sum(1 for i in pred_indices if is_hit(i, gt_indices, match_tolerance))

        # === 原有统计 ===
        total_hit += hits
        total_predicted += len(pred_indices)
        hit_ratio = hits / len(pred_indices) if pred_indices else 0.0
        all_hit_ratios.append(hit_ratio)

        # === 新增：Recall@K & Hit@1（Top-K 采用K=GT数量；Top-1采用预测最高分）===
        if len(gt_indices) > 0:
            K = len(gt_indices)
            # 取有效范围内的Top-K（不依赖阈值，更公平）
            valid_scores = [(i, y_pred[i]) for i in valid_range]
            topk_pred = [i for i, _ in sorted(valid_scores, key=lambda t: t[1], reverse=True)[:K]]

            # Recall@K（匹配容差）
            recall_hits = sum(1 for g in gt_indices if any(abs(g - p) <= match_tolerance for p in topk_pred))
            recall_k = recall_hits / K
            per_file_recall_at_k.append(recall_k)

            # Hit@1
            if valid_scores:
                top1 = max(valid_scores, key=lambda t: t[1])[0]
                per_file_hit1.append(1.0 if any(abs(top1 - g) <= match_tolerance for g in gt_indices) else 0.0)

        # 可视化（随机顺序中的前 N 个，避免挑样本嫌疑）
        if save_dir and visualized < max_visualize:
            plt.figure(figsize=(9, 4))
            # GT曲线
            plt.plot(labels, label="Ground Truth", marker='o', alpha=0.7)
            # 预测曲线
            plt.plot(y_pred, label="Prediction", linestyle='--', linewidth=1.5)
            # 阈值选出的预测点
            if pred_indices:
                plt.scatter(pred_indices, [y_pred[i] for i in pred_indices],
                            color='red', marker='x', s=60, label='Predicted (threshold)')
            # 命中点（阈值法下的命中）
            hit_indices = [i for i in pred_indices if is_hit(i, gt_indices, match_tolerance)]
            if hit_indices:
                plt.scatter(hit_indices, [y_pred[i] for i in hit_indices],
                            facecolors='none', edgecolors='green', marker='o',
                            s=80, linewidth=2, label='Hit (±{} fr)'.format(match_tolerance))

            # 边界标记
            plt.axvline(exclude_margin, color='gray', linestyle=':', alpha=0.5)
            plt.axvline(total_frames - exclude_margin - 1, color='gray', linestyle=':', alpha=0.5)

            # 标注 Top-K（蓝色竖线），与论文里“召回能力”呼应
            if len(gt_indices) > 0:
                for p in topk_pred:
                    plt.axvline(p, linestyle='--', alpha=0.15)

            base = os.path.basename(npy_path)
            plt.title(f"{base} | Hit Ratio: {hit_ratio:.2f} | Recall@K: {recall_k:.2f}")
            plt.xlabel("Frame Index")
            plt.ylabel("Score")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{base}.png"), dpi=300)
            plt.close()
            visualized += 1

    # 汇总
    if valid_file_count == 0:
        print("[WARNING] No valid files with key frames ≥ 0.5 found.")
        return 0.0, 0.0

    overall_ratio = total_hit / total_predicted if total_predicted > 0 else 0.0
    avg_per_file_ratio = float(np.mean(all_hit_ratios)) if all_hit_ratios else 0.0
    mean_recall_k = float(np.mean(per_file_recall_at_k)) if per_file_recall_at_k else 0.0
    mean_hit1 = float(np.mean(per_file_hit1)) if per_file_hit1 else 0.0

    print("\n=== Key Frame Hit Rate Evaluation ===")
    print(f"Valid Files Evaluated : {valid_file_count}")
    print(f"Threshold             : score ≥ {score_threshold}")
    print(f"Margin Excluded       : {exclude_margin} frames")
    print(f"Match Tolerance       : ±{match_tolerance} frames")
    print(f"Total Predicted Frames: {total_predicted}")
    print(f"Total Hits (GT ≥ 0.5) : {total_hit}")
    print(f"Overall Hit Ratio     : {overall_ratio:.4f}  (micro, threshold-based)")
    print(f"Avg Hit Ratio/File    : {avg_per_file_ratio:.4f} (macro, threshold-based)")
    print(f"Mean Recall@K         : {mean_recall_k:.4f}  (macro, K=#GT, tolerance-aware)")
    print(f"Mean Hit@1            : {mean_hit1:.4f}  (macro, tolerance-aware)")
    if save_dir:
        print(f"Visualizations saved to: {save_dir}")