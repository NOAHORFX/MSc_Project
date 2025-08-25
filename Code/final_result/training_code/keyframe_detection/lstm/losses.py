# losses.py
import tensorflow as tf

SIM_WEIGHT = tf.Variable(0.0, trainable=False, dtype=tf.float32)

def mse_with_similarity_and_confidence_dynamic():
    def loss_fn(y_true, y_pred):
        if tf.shape(y_true)[-1] == 2:
            y_true = y_true[..., 0]

        weights = tf.where(tf.equal(y_true, 1.0), 1.0, 0.5)
        mse = tf.reduce_mean(weights * tf.square(y_true - y_pred))

        # 相似性正则项
        y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)  # [B, T]
        sim = tf.matmul(tf.expand_dims(y_pred_centered, 2), tf.expand_dims(y_pred_centered, 1))  # [B, T, T]

        norm = tf.norm(y_pred_centered, axis=1, keepdims=True)  # [B, 1]
        denom = tf.matmul(tf.expand_dims(norm, 2), tf.expand_dims(norm, 1)) + 1e-6  # [B, 1, 1] → [B, T, T]
        cosine_sim = sim / denom  # [B, T, T]

        diff = tf.expand_dims(y_pred, 2) - tf.expand_dims(y_pred, 1)  # [B, T, T]
        sim_loss = tf.reduce_mean(tf.square(diff) * cosine_sim)  # [B, T, T]

        return mse + SIM_WEIGHT * sim_loss

    loss_fn.__name__ = "mse_with_sim_and_conf_dynamic"
    return loss_fn
