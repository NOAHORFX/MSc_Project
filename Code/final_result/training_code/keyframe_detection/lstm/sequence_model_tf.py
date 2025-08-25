import tensorflow as tf
from tensorflow.keras import layers, models

def build_sequence_model(input_shape=(115, 256)):
    inp = layers.Input(shape=input_shape)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(inp)
    x = layers.Bidirectional(layers.LSTM(96, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Bidirectional(layers.LSTM(32, return_sequences=True))(x)

    x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
    x = layers.TimeDistributed(layers.Dense(1))(x)

    out = layers.Reshape((input_shape[0],))(x)

    return models.Model(inp, out)

# def transformer_encoder(inputs, head_size=64, num_heads=4, ff_dim=128, dropout=0.1):
#     x = layers.MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
#     x = layers.Dropout(dropout)(x)
#     x = layers.LayerNormalization(epsilon=1e-6)(x + inputs)

#     x_ff = layers.Dense(ff_dim, activation='relu')(x)
#     x_ff = layers.Dense(inputs.shape[-1])(x_ff)
#     x = layers.Dropout(dropout)(x_ff)
#     return layers.LayerNormalization(epsilon=1e-6)(x + x_ff)

# def build_sequence_model(input_shape=(115, 256), num_layers=4):
#     seq_len, feature_dim = input_shape
#     inp = layers.Input(shape=input_shape)

#     # 添加可训练的位置编码
#     positions = tf.range(start=0, limit=seq_len, delta=1)
#     pos_embed = layers.Embedding(input_dim=seq_len, output_dim=feature_dim)(positions)
#     x = inp + pos_embed

#     # Transformer 编码器堆叠
#     for _ in range(num_layers):
#         x = transformer_encoder(x, head_size=64, num_heads=4, ff_dim=128)

#     # 每帧输出一个得分
#     x = layers.TimeDistributed(layers.Dense(64, activation='relu'))(x)
#     x = layers.TimeDistributed(layers.Dense(1))(x)
#     out = layers.Reshape((seq_len,))(x)

#     return models.Model(inp, out)


def focal_mse(y_true, y_pred, gamma=2.0):
    error = tf.square(y_true - y_pred)
    weights = tf.pow(tf.abs(y_true - y_pred), gamma)
    return tf.reduce_mean(weights * error)

SIM_WEIGHT = tf.Variable(0.0, trainable=False, dtype=tf.float32)

def mse_with_similarity_and_confidence_dynamic():
    def loss_fn(y_true, y_pred):
        weights = tf.where(y_true >= 0.5, 1.0, 0.1)
        mse = tf.reduce_mean(weights * tf.square(y_true - y_pred))

        y_pred_centered = y_pred - tf.reduce_mean(y_pred, axis=1, keepdims=True)
        sim = tf.matmul(y_pred_centered, y_pred_centered, transpose_b=True)
        norm = tf.norm(y_pred_centered, axis=1, keepdims=True)
        denom = tf.matmul(norm, norm, transpose_b=True) + 1e-6
        cosine_sim = sim / denom

        diff = tf.expand_dims(y_pred, 2) - tf.expand_dims(y_pred, 1)
        sim_loss = tf.reduce_mean(tf.square(diff) * cosine_sim)

        return mse + SIM_WEIGHT * sim_loss
    loss_fn.__name__ = "mse_with_sim_and_conf_dynamic"
    return loss_fn


