import os
import numpy as np
import tensorflow as tf
from glob import glob
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks
from sequence_model_tf import build_sequence_model, focal_mse
from label_propagation import propagate_labels_by_similarity
from sequence_model_tf import build_sequence_model
from losses import mse_with_similarity_and_confidence_dynamic
from callbacks import UpdateSimWeightCallback

def expand_soft_label(label_array, total_frames=115):
    y = np.array(label_array, dtype=np.float32)
    expanded = np.zeros_like(y)

    for i in range(total_frames):
        if y[i] >= 0.5:
            for offset, val in [(-2, 0.3), (-1, 0.5), (0, 1.0), (1, 0.5), (2, 0.3)]:
                j = i + offset
                if 0 <= j < total_frames:
                    expanded[j] = max(expanded[j], val)

    return expanded


def train_sequence_model(feature_dir="D:/features_vit", input_shape=(115, 6144),
                         batch_size=8, epochs=10,
                         model_save_path="best_sequence_model.h5"):

    feature_files = glob(os.path.join(feature_dir, "**", "*.npy"), recursive=True)
    if len(feature_files) == 0:
        raise ValueError("No .npy files found in given feature_dir.")

    train_files, val_files = train_test_split(feature_files, test_size=0.2, random_state=42)

    def make_generator(file_list):
        def generator():
            for path in file_list:
                data = np.load(path, allow_pickle=True).item()
                features = data["features"]
                labels = expand_soft_label(data["labels"])
                labels = propagate_labels_by_similarity(features, labels)
                yield features.astype(np.float32), labels.astype(np.float32)
        return generator

    def make_dataset(file_list):
        return tf.data.Dataset.from_generator(
            make_generator(file_list),
            output_signature=(
                tf.TensorSpec(shape=input_shape, dtype=tf.float32),
                tf.TensorSpec(shape=(input_shape[0],), dtype=tf.float32)
            )
        ).shuffle(128).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    train_dataset = make_dataset(train_files)
    val_dataset = make_dataset(val_files)

    model = build_sequence_model(input_shape=input_shape)
    # model.compile(optimizer='adam', loss=mse_with_similarity_regularization(sim_weight=0.1), metrics=['mae'])
    model.compile(
        optimizer='adam',
        loss=mse_with_similarity_and_confidence_dynamic(),
        metrics=['mae']
    )


    cb = [
        UpdateSimWeightCallback(min_weight=0.0, max_weight=0.2, total_epochs=epochs),
        callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(filepath=model_save_path, monitor="val_loss", save_best_only=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    print("[INFO] Starting training...")
    model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=cb)
    print(f"[INFO] Best model saved to: {model_save_path}")
    return model, val_files
