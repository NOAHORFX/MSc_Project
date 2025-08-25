from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Concatenate)

def build_residual_model(input_shape=(562, 744, 4)):
    img_mask_input = Input(shape=input_shape, name="img_mask_input")

    ac_pred_input = Input(shape=(1,), name="ac_pred_input")

    x = Conv2D(32, 3, activation='relu')(img_mask_input)
    x = MaxPooling2D()(x)
    x = Conv2D(64, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Conv2D(128, 3, activation='relu')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)

    x = Concatenate()([x, ac_pred_input])

    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(1, name="residual_output")(x)

    model_p = Model(inputs=[img_mask_input, ac_pred_input], outputs=output)
    return model_p