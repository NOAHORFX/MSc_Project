import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from model import model as build_model  # 从 model.py 中导入
from tensorflow.keras.layers import AveragePooling2D, Flatten

def load_segmentation_feature_model(weight_path):
    """
    加载已训练的语义分割模型，并输出特征提取模型（GAP输出）
    """
    seg_model = build_model(img_height=384, img_width=508, classes=2)
    seg_model.load_weights(weight_path)

    # 提取上采样前的融合特征（倒数第3层为 fused）
    fused_output = seg_model.layers[-3].output
    x = AveragePooling2D(pool_size=(2, 2))(fused_output)  # 输出 (6, 8, 128)
    x = Flatten()(x)                                      # 输出 (6144,)
    feature_model = Model(inputs=seg_model.input, outputs=x)
    return feature_model, (508, 384)
