import os
import numpy as np
from models.sequence_model import build_sequence_model, mse_with_similarity_and_confidence_dynamic
from tensorflow.keras.models import load_model
#
# def load_and_predict_best_frame(feature_dir, model_path, input_shape=(115, 6144)):
#     """
#     给定某个 sweep 的特征文件路径，加载 LSTM 模型预测每帧得分，并返回得分最高帧索引
#     """
#     if not os.path.exists(feature_dir):
#         raise FileNotFoundError(f"Feature directory not found: {feature_dir}")
#
#     npy_path = os.path.join(feature_dir, "features.npy")
#     print(f"Loading features from {npy_path}")
#     if not os.path.exists(npy_path):
#         raise FileNotFoundError(f"features.npy not found at: {npy_path}")
#
#     features = np.load(npy_path)  # shape: (115, D)
#     features = features.astype(np.float32)
#     features = np.expand_dims(features, axis=0)  # shape: (1, 115, D)
#
#     model = build_sequence_model(input_shape=input_shape)
#     model.compile(optimizer='adam', loss=mse_with_similarity_and_confidence_dynamic())
#     model.load_weights(model_path)
#
#     scores = model.predict(features, verbose=0)[0]  # shape: (115,)
#     best_frame_idx = int(np.argmax(scores))
#     return best_frame_idx, scores

def load_and_predict_best_frame(feature_dir, model_path):
    """
    使用已训练模型预测关键帧得分，返回得分最高帧索引
    """
    npy_path = os.path.join(feature_dir, "features.npy")
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"features.npy not found at: {npy_path}")

    data = np.load(npy_path, allow_pickle=True).item()
    features = data["features"].astype(np.float32)  # shape: (115, D)
    features = np.expand_dims(features, axis=0)     # shape: (1, 115, D)

    model = load_model(model_path, compile=False, custom_objects={
        "mse_with_sim_and_conf_dynamic": mse_with_similarity_and_confidence_dynamic()
    })

    scores = model.predict(features, verbose=0)[0]
    best_idx = int(np.argmax(scores))
    return best_idx, scores