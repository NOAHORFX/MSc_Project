import os
import numpy as np
import SimpleITK as sitk
from PIL import Image


def extract_annotated_frames(root_dir, output_dir):
    image_dir = os.path.join(root_dir, "images")
    mask_dir = os.path.join(root_dir, "masks")

    image_out = os.path.join(output_dir, "images")
    mask_out = os.path.join(output_dir, "masks")
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(mask_out, exist_ok=True)

    count = 0  # 全局帧编号

    for fname in sorted(os.listdir(image_dir)):
        if not fname.endswith(".mha"):
            continue

        case_id = fname.replace(".mha", "")
        print(f"Processing {case_id}...")

        img_path = os.path.join(image_dir, fname)
        msk_path = os.path.join(mask_dir, fname)

        image = sitk.ReadImage(img_path)
        mask = sitk.ReadImage(msk_path)

        image_np = sitk.GetArrayFromImage(image)
        mask_np = sitk.GetArrayFromImage(mask)

        for i in range(image_np.shape[0]):
            msk = mask_np[i]
            if np.any(np.isin(msk, [1, 2])):  # 仅保存包含标注的帧
                img = image_np[i]

                img_norm = ((img - np.min(img)) / np.ptp(img) * 255).astype(np.uint8)
                msk_uint8 = msk.astype(np.uint8)

                frame_name = f"frame_{count:06d}.png"
                Image.fromarray(img_norm).save(os.path.join(image_out, frame_name))
                Image.fromarray(msk_uint8).save(os.path.join(mask_out, frame_name))
                count += 1

    print(f"Extraction complete, total number of annotated frames saved: {count}")
