import numpy as np
import matplotlib.pyplot as plt
import os
import json
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import gaussian_filter1d
from tensorflow.keras.models import load_model
import tensorflow as tf

# === 注册名必须一致 ===
def mse_with_similarity_and_confidence_dynamic(sim_weight=0.1):
    def loss_fn(y_true, y_pred):
        if tf.shape(y_true)[-1] == 2:
            y_true = y_true[..., 0]

        weights = tf.where(tf.equal(y_true, 1.0), 1.0, 0.5)
        mse = tf.reduce_mean(weights * tf.square(y_true - y_pred))

        y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)
        sim = tf.matmul(tf.expand_dims(y_pred_centered, 2), tf.expand_dims(y_pred_centered, 1))

        norm = tf.norm(y_pred_centered, axis=1, keepdims=True)
        denom = tf.matmul(tf.expand_dims(norm, 2), tf.expand_dims(norm, 1)) + 1e-6
        cosine_sim = sim / denom

        diff = tf.expand_dims(y_pred, 2) - tf.expand_dims(y_pred, 1)
        sim_loss = tf.reduce_mean(tf.square(diff) * cosine_sim)

        return mse + sim_weight * sim_loss

    loss_fn.__name__ = "mse_with_sim_and_conf_dynamic"
    return loss_fn


# === 主函数 ===
def evaluate_and_visualize_predictions(model_path, npy_file, save_json_path=None,
                                       top_k=3, score_threshold=0.1, flat_std_threshold=0.05,
                                       frame_exclude_margin=10, max_peak_count=5, relative_peak_ratio=0.00001,
                                       smooth_method=None):
    # 正确注册损失函数
    model = load_model(model_path, custom_objects={
        "mse_with_sim_and_conf_dynamic": mse_with_similarity_and_confidence_dynamic(0.1)
    })

    data = np.load(npy_file, allow_pickle=True).item()
    x = np.expand_dims(data["features"], axis=0)
    y_true = data["labels"]
    y_pred = model.predict(x)[0]

    # 平滑
    if smooth_method == 'gaussian':
        y_pred = gaussian_filter1d(y_pred, sigma=2)
    elif smooth_method == 'savgol':
        y_pred = savgol_filter(y_pred, window_length=11, polyorder=3)
    elif smooth_method is not None:
        raise ValueError("Unsupported smooth_method. Choose 'gaussian', 'savgol', or None.")

    total_frames = len(y_pred)
    pred_std = np.std(y_pred)
    print(f"Prediction std: {pred_std:.4f}")

    valid_start = frame_exclude_margin
    valid_end = total_frames - frame_exclude_margin
    valid_y_pred = y_pred[valid_start:valid_end]

    centered = valid_y_pred - np.mean(valid_y_pred)
    relative_threshold = relative_peak_ratio * np.max(centered)

    peaks, _ = find_peaks(centered, height=relative_threshold)
    peaks = peaks + valid_start
    print(f"Detected {len(peaks)} relative peaks")

    if len(peaks) > max_peak_count:
        filtered_indices = []
    else:
        top_indices = y_pred.argsort()[-top_k:][::-1]
        valid_range = set(range(valid_start, valid_end))
        top_indices = [i for i in top_indices if i in valid_range]

        if pred_std < flat_std_threshold:
            filtered_indices = top_indices
        else:
            filtered_indices = [i for i in top_indices if y_pred[i] >= score_threshold]

    print("Predicted key frames:", filtered_indices)
    print("Scores:", [y_pred[i] for i in filtered_indices])

    # 可视化
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label="Ground Truth", marker='o')
    plt.plot(y_pred, label="Prediction", linestyle='--')
    if filtered_indices:
        plt.scatter(filtered_indices, [y_pred[i] for i in filtered_indices],
                    color='red', label='Predicted Key Frames', zorder=5)
    plt.axvline(valid_start, color='gray', linestyle=':', label='Margin')
    plt.axvline(valid_end - 1, color='gray', linestyle=':')
    plt.title(f"Prediction: {os.path.basename(npy_file)}")
    plt.xlabel("Frame Index")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 保存 JSON
    if save_json_path:
        result = {
            "file": os.path.basename(npy_file),
            "filtered_indices": [int(i) for i in filtered_indices],
            "predicted_scores": [float(y_pred[i]) for i in filtered_indices]
        }
        with open(save_json_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Saved to: {save_json_path}")
