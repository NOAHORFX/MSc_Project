# callbacks.py
import tensorflow as tf
from losses import SIM_WEIGHT

class UpdateSimWeightCallback(tf.keras.callbacks.Callback):
    def __init__(self, min_weight=0.0, max_weight=0.2, total_epochs=20):
        super().__init__()
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.total_epochs = total_epochs

    def on_epoch_begin(self, epoch, logs=None):
        ratio = epoch / self.total_epochs
        new_weight = self.min_weight + (self.max_weight - self.min_weight) * ratio
        tf.keras.backend.set_value(SIM_WEIGHT, new_weight)
        print(f"[Callback] sim_weight updated to: {new_weight:.4f}")
