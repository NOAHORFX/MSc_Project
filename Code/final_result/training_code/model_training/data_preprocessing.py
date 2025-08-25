import os
import glob
import numpy as np
import tensorflow as tf


def create_segmentation_datasets(image_dir, mask_dir, batch_size=24, val_split=0.2, test_split=0.1,
                                 resize_size=(384, 508), crop_size=(384, 508), ):
    img_all = sorted(glob.glob(os.path.join(image_dir, '*.png')))
    label_all = sorted(glob.glob(os.path.join(mask_dir, '*.png')))

    # 随机打乱
    num_samples = len(img_all)
    idx = np.random.permutation(num_samples)
    img_all = np.array(img_all)[idx]
    label_all = np.array(label_all)[idx]
    n_test = int(num_samples * test_split)
    n_val = int(num_samples * val_split)
    n_train = num_samples - n_val - n_test

    # 划分
    img_train = img_all[:n_train]
    mask_train = label_all[:n_train]
    img_val = img_all[n_train:n_train + n_val]
    mask_val = label_all[n_train:n_train + n_val]
    img_test = img_all[n_train + n_val:]
    mask_test = label_all[n_train + n_val:]

    print(f"Training samples: {n_train}, validation samples: {n_val}, test samples: {n_test}")

    def merge_labels(mask):
        return tf.where(mask == 2, tf.cast(1, mask.dtype), mask)

    def read_png(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_png(img, channels=3)
        return img

    def read_png_label(path):
        with tf.device('/cpu:0'):
            img = tf.io.read_file(path)
            img = tf.image.decode_png(img, channels=1)
            img = tf.where(img == 255, tf.zeros_like(img), img)
        return img

    def crop_img(img, mask):
        img_u8 = tf.cast(img, tf.uint8)
        concat = tf.concat([img_u8, mask], axis=-1)
        concat = tf.image.resize(concat, resize_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        combined_crop = tf.image.random_crop(concat, size=[crop_size[0], crop_size[1], tf.shape(concat)[-1]])
        img_crop = combined_crop[:, :, :3]
        mask_crop = combined_crop[:, :, 3:]
        return img_crop, tf.cast(mask_crop, mask.dtype)

    def normalize(img, mask):
        img = tf.cast(img, tf.float32) / 127.5 - 1.0
        mask = tf.cast(mask, tf.int32)
        return img, mask

    def load_image_train(img_path, mask_path):
        img = read_png(img_path)
        mask = read_png_label(mask_path)
        mask = merge_labels(mask)

        img = tf.image.random_saturation(img, lower=0.8, upper=1.2)
        img = tf.image.random_hue(img, max_delta=0.2)
        img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
        img = tf.clip_by_value(img, 0, 255)

        def add_speckle(x):
            x_float = tf.cast(x, tf.float32)
            noise = tf.random.normal(shape=tf.shape(x_float), mean=0.0, stddev=0.02 * 255.0)
            y = x_float + x_float * (noise / 255.0)
            y = tf.clip_by_value(y, 0.0, 255.0)
            return tf.cast(y, tf.uint8)

        if tf.random.uniform([]) < 0.5:
            img = add_speckle(img)

        scale = tf.random.uniform([], 0.9, 1.1)
        new_h = tf.cast(tf.cast(tf.shape(img)[0], tf.float32) * scale, tf.int32)
        new_w = tf.cast(tf.cast(tf.shape(img)[1], tf.float32) * scale, tf.int32)
        img = tf.image.resize(img, [new_h, new_w])
        mask = tf.image.resize(mask, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        img, mask = crop_img(img, mask)

        if tf.random.uniform([]) > 0.5:
            img = tf.image.flip_left_right(img)
            mask = tf.image.flip_left_right(mask)

        img, mask = normalize(img, mask)
        return img, mask

    def load_image_val_or_test(img_path, mask_path):
        img = read_png(img_path)
        mask = read_png_label(mask_path)
        mask = merge_labels(mask)

        img, mask = crop_img(img, mask)
        img, mask = normalize(img, mask)
        return img, mask

    auto = tf.data.experimental.AUTOTUNE

    # 训练集
    dataset_train = tf.data.Dataset.from_tensor_slices((img_train, mask_train))
    dataset_train = dataset_train.map(load_image_train, num_parallel_calls=auto)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(auto)

    # 验证集
    dataset_val = tf.data.Dataset.from_tensor_slices((img_val, mask_val))
    dataset_val = dataset_val.map(load_image_val_or_test, num_parallel_calls=auto)
    dataset_val = dataset_val.batch(batch_size)
    dataset_val = dataset_val.prefetch(auto)

    # 测试集
    dataset_test = tf.data.Dataset.from_tensor_slices((img_test, mask_test))
    dataset_test = dataset_test.map(load_image_val_or_test, num_parallel_calls=auto)
    dataset_test = dataset_test.batch(batch_size)
    dataset_test = dataset_test.prefetch(auto)

    return dataset_train, dataset_val, dataset_test
