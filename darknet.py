import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Concatenate, BatchNormalization, LeakyReLU, ZeroPadding2D
from tensorflow.keras.regularizers import l2


def block_darknet(inputs, num_filter, num_block):
    x = ZeroPadding2D((1, 0), (1, 0))(inputs)
    x = Conv2D(num_filter, (3, 3), strides=(2, 2), kernel_regularizer=l2(5e-4))(x)
    x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
    for i in range(num_block):
        x = Conv2D(num_filter // 2, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(5e-4))(x)
        x1 += x
    return Concatenate()([x1, x2])


def darknet_body(inputs):
    x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(5e-4), name='Conv_1')(inputs)
    x = block_darknet(x, 64, 1)
    return x


inputs = tf.keras.layers.Input((416, 416, 3))
model = tf.keras.Model(inputs, darknet_body(inputs)).summary()
