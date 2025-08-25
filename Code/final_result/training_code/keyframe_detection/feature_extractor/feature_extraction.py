import os
import numpy as np
from glob import glob
from tqdm import tqdm
from segmentation_loader import load_segmentation_feature_model
from sweep_utils import load_sweep_images_and_labels
import tensorflow as tf

def extract_features_from_sweeps(
    sweep_root,
    save_dir,
    max_frames=115,
    weight_path="/home/featurize/data/weights.h5"
):
    os.makedirs(save_dir, exist_ok=True)
    model, input_size = load_segmentation_feature_model(weight_path)
    input_h, input_w = input_size[1], input_size[0]

    sweep_dirs = glob(os.path.join(sweep_root, "*", "sweep_*"))
    for sweep_path in tqdm(sweep_dirs, desc="Extracting features"):
        result = load_sweep_images_and_labels(sweep_path, max_frames)
        if result is None:
            continue
        images, labels = result
        if not images:
            continue

        # 图像预处理
        images = [(img / 127.5 - 1.0).astype(np.float32) for img in images]
        images = np.stack(images, axis=0)  # shape: (N, H, W, 3)

        # 一次性提取特征（不再设置 batch_size）
        features = model(images, training=False).numpy()  # shape: (N, D)


        # 保存
        rel_path = os.path.relpath(sweep_path, sweep_root)
        save_path = os.path.join(save_dir, rel_path)
        os.makedirs(save_path, exist_ok=True)
        np.save(os.path.join(save_path, "features.npy"), {
            "features": features,
            "labels": np.array(labels, dtype=np.float32)
        })
