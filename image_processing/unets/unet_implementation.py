import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def conv_block(inputs, num_filters):
    x = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(num_filters, 3, activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)

    return x

def encoder_block(inputs, num_filters):
    x = conv_block(inputs, num_filters)
    p = layers.MaxPooling2D(pool_size=(2, 2))(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    x = layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = layers.Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape=(32, 32, 3), n_classes=10):
    inputs = layers.Input(input_shape)

    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)

    b1 = conv_block(p3, 512)

    d1 = decoder_block(b1, s3, 256)
    d2 = decoder_block(d1, s2, 128)
    d3 = decoder_block(d2, s1, 64)

    outputs = layers.Conv2D(n_classes, (1, 1), padding='same', activation='softmax')(d3)
    model = keras.Model(inputs, outputs, name='unet')
    return model