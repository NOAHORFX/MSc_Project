import os
import numpy as np
import SimpleITK as sitk
from PIL import Image
from tqdm import tqdm
from models.seg_model import load_segmentation_feature_model


def split_mha_into_sweeps(mha_path, save_root, frames_per_sweep=140, remove_tail=25, resize=(508, 384)):
    """
    拆分 .mha 文件为多个 sweep，每个 sweep 截取前 max_frames 帧保存为图像
    """
    os.makedirs(save_root, exist_ok=True)
    img = sitk.ReadImage(mha_path)
    volume = sitk.GetArrayFromImage(img)  # (Z, H, W)
    total_frames = volume.shape[0]
    num_sweeps = total_frames // frames_per_sweep

    base_name = os.path.splitext(os.path.basename(mha_path))[0]
    all_sweep_paths = []

    for i in range(num_sweeps):
        sweep_dir = os.path.join(save_root, f"{base_name}/sweep_{i+1}")
        os.makedirs(os.path.join(sweep_dir, "images"), exist_ok=True)

        start = i * frames_per_sweep
        end = start + frames_per_sweep - remove_tail
        sweep_volume = volume[start:end]  # (115, H, W)

        for j, frame in enumerate(sweep_volume):
            img = Image.fromarray(frame).convert("L").resize(resize)
            img.save(os.path.join(sweep_dir, "images", f"{j:03d}.png"))

        with open(os.path.join(sweep_dir, "label.json"), "w") as f:
            f.write('{"best": [], "second_best": []}')

        all_sweep_paths.append(sweep_dir)

    print("[Done] MHA splitting finished.")
    return all_sweep_paths


def extract_features_from_sweeps(sweep_dirs, max_frames=115, weight_path="assets/seg_model_weights.h5"):

    model, input_size = load_segmentation_feature_model(weight_path)
    input_w, input_h = input_size  # 注意返回的是 (508, 384)

    for sweep_path in tqdm(sweep_dirs, desc="Extracting features"):
        img_dir = os.path.join(sweep_path, "images")
        images = []
        for i in range(max_frames):
            img_path = os.path.join(img_dir, f"{i:03d}.png")
            if not os.path.exists(img_path):
                continue
            img = Image.open(img_path).convert("RGB").resize((input_w, input_h))
            img = np.array(img) / 127.5 - 1.0  # normalize
            images.append(img.astype(np.float32))

        if len(images) == 0:
            print(f"[⚠] Skip {sweep_path}: no valid frames")
            continue

        images = np.stack(images, axis=0)  # (N, H, W, 3)
        features = model(images, training=False).numpy()  # (N, 6144)

        np.save(os.path.join(sweep_path, "features.npy"), features)

    print("[Done] Feature extraction finished.")