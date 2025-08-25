import os
import json
from PIL import Image
import numpy as np

def load_sweep_images_and_labels(sweep_path, max_frames=115):
    img_dir = os.path.join(sweep_path, "images")
    label_path = os.path.join(sweep_path, "label.json")
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r") as f:
        label = json.load(f)

    best = label.get("best", [])
    second = label.get("second_best", [])
    label_indices = {i: 0.5 for i in second}
    label_indices.update({i: 1.0 for i in best})

    images = []
    labels = []
    for i in range(max_frames):
        path = os.path.join(img_dir, f"{i:03d}.png")
        if not os.path.exists(path):
            continue
        img = Image.open(path).convert("RGB").resize((508, 384))
        images.append(np.array(img))
        labels.append(label_indices.get(i, 0.0))

    return images, labels
