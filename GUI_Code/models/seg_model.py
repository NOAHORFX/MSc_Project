import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import *
from keras.applications import Xception


def upsample(tensor, size):
    """双线性上采样到指定 size（height, width）"""
    name = tensor.name.split('/')[0] + '_upsample'
    return Lambda(lambda x: tf.image.resize(x, size), name=name)(tensor)


def depthwise_separable_conv2d(x, dilation_rate):
    """ASPP 中的深度可分离卷积 + BN + ReLU"""
    y = DepthwiseConv2D(kernel_size=3, dilation_rate=dilation_rate,
                        padding='same', use_bias=False)(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)
    y = Conv2D(128, kernel_size=1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    return Activation('relu')(y)


def edge_detection_module(x):
    """基于 Sobel 核的边缘检测分支（每通道共享同一核），并用 Lambda 包裹 tf.abs"""
    # 静态获取通道数
    _, h, w, c = K.int_shape(x)
    if c is None:
        raise ValueError("edge_detection_module 需要已知的通道数 (x.shape[-1])")

    # 定义 3×3 Sobel 核
    sobel_x = np.array([[-1., 0., 1.],
                        [-2., 0., 2.],
                        [-1., 0., 1.]], dtype=np.float32)
    sobel_y = np.array([[-1., -2., -1.],
                        [0., 0., 0.],
                        [1., 2., 1.]], dtype=np.float32)

    # 扩展到 (3,3,1,1) 并用 NumPy tile 到 (3,3,c,1)
    sobel_x = np.tile(sobel_x[:, :, np.newaxis, np.newaxis], (1, 1, c, 1))
    sobel_y = np.tile(sobel_y[:, :, np.newaxis, np.newaxis], (1, 1, c, 1))

    # 固定 Sobel 核做 depthwise 卷积
    gx = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, depthwise_initializer=tf.constant_initializer(sobel_x), trainable=False)(x)
    gy = DepthwiseConv2D(kernel_size=3, padding='same', depth_multiplier=1, use_bias=False, depthwise_initializer=tf.constant_initializer(sobel_y), trainable=False)(x)

    # 用 Lambda 包裹 tf.abs，变成合法的 Keras 层
    abs_gx = Lambda(lambda t: tf.abs(t), name='abs_gx')(gx)
    abs_gy = Lambda(lambda t: tf.abs(t), name='abs_gy')(gy)

    edge = Add()([abs_gx, abs_gy])
    edge = Conv2D(128, kernel_size=1, padding='same', use_bias=False)(edge)
    edge = BatchNormalization()(edge)
    return Activation('relu')(edge)


def aspp(tensor):
    """标准 ASPP 模块，不含 Transformer"""
    dims = K.int_shape(tensor)
    # Image Pooling 分支
    y_pool = AveragePooling2D(pool_size=(dims[1], dims[2]))(tensor)
    y_pool = Conv2D(128, 1, padding='same', use_bias=False)(y_pool)
    y_pool = BatchNormalization()(y_pool)
    y_pool = Activation('relu')(y_pool)
    y_pool = upsample(y_pool, size=(dims[1], dims[2]))

    # 1×1, 3×3×(6/12/18) 分支
    y_1 = Conv2D(128, 1, padding='same', use_bias=False)(tensor)
    y_1 = BatchNormalization()(y_1)
    y_1 = Activation('relu')(y_1)

    y_6 = depthwise_separable_conv2d(tensor, dilation_rate=6)
    y_12 = depthwise_separable_conv2d(tensor, dilation_rate=12)
    y_18 = depthwise_separable_conv2d(tensor, dilation_rate=18)

    # 拼接 & 1×1 再融合
    y = concatenate([y_pool, y_1, y_6, y_12, y_18])
    y = Conv2D(128, 1, padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    return Activation('relu')(y)


def model(img_height, img_width, classes=3):
    inputs = Input(shape=(img_height, img_width, 3))
    base = Xception(input_tensor=inputs, weights='imagenet', include_top=False)

    # 1) 边缘分支 输入浅层特征
    shallow = base.get_layer('block3_pool').output
    edge_feat = edge_detection_module(shallow)

    # 2) ASPP 分支 输入深层特征
    deep = base.get_layer('block14_sepconv2_act').output
    aspp_feat = aspp(deep)

    # 将边缘特征上采样到与 ASPP 输出一致的 size
    aspp_h, aspp_w = K.int_shape(aspp_feat)[1:3]
    edge_up = upsample(edge_feat, size=(aspp_h, aspp_w))

    # 3) 融合
    fused = Add()([aspp_feat, edge_up])

    # 4) 上采样到原图 & 最终分类
    x = upsample(fused, size=(img_height, img_width))
    # x = Conv2D(classes, kernel_size=1, activation='sigmoid' if classes == 1 else 'softmax')(x)
    x = Conv2D(classes, kernel_size=1, activation='softmax')(x)

    model_output = Model(inputs=inputs, outputs=x, name='ASPP_Edge_Fusion')
    print(f'*** Output Shape: {model_output.output_shape} ***')
    return model_output


from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, Flatten

def load_segmentation_feature_model(weight_path):
    seg_model = model(img_height=384, img_width=508, classes=2)
    seg_model.load_weights(weight_path)

    fused_output = seg_model.layers[-3].output
    x = AveragePooling2D(pool_size=(2, 2))(fused_output)  # 输出 (6, 8, 128)
    x = Flatten()(x)                                      # 输出 (6144,)
    feature_model = Model(inputs=seg_model.input, outputs=x)
    return feature_model, (508, 384)