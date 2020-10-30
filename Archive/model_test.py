import tensorflow as tf
from tensorflow import keras
from keras import models, regularizers
from keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Concatenate

from functools import partial, wraps

from utils import compose
from data_pipeline import *


@wraps(Conv2D)
def DarknetConv2D(*args, **kwargs):
    """Wrapper to set Darknet parameters for Convolution2D."""
    darknet_conv_kwargs = {'kernel_regularizer': regularizers.l2(5e-4),
                           'padding': 'valid' if kwargs.get('strides') == (2, 2) else 'same'}
    darknet_conv_kwargs.update(kwargs)
    return Conv2D(*args, **darknet_conv_kwargs)


def DarknetConv2D_BN_Leaky(*args, **kwargs):
    """Darknet Convolution2D followed by BatchNormalization and LeakyReLU."""
    no_bias_kwargs = {'use_bias': False}
    no_bias_kwargs.update(kwargs)
    return compose(
        DarknetConv2D(*args, **no_bias_kwargs),
        BatchNormalization(),
        LeakyReLU(alpha=0.1))


def CSRPblock(inputs, num_filters, num_blocks):
    """A series of CSRPblocks starting with a down sampling Convolution2D"""
    # Darknet uses left and top padding instead of 'same' mode
    x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
    x = DarknetConv2D_BN_Leaky(num_filters, (3, 3), strides=(2, 2))(x)
    x1, x2 = tf.split(value=x, num_or_size_splits=2, axis=-1)
    for i in range(num_blocks):
        x = DarknetConv2D_BN_Leaky(num_filters // 2, (3, 3))(x1)
        x1 += x
    return Concatenate(axis=-1)([x1, x2])


def darknet_body(x):
    x = DarknetConv2D_BN_Leaky(32, (3, 3))(x)
    x = CSRPblock(x, 64, 1)
    x = CSRPblock(x, 128, 2)
    y1 = CSRPblock(x, 256, 8)
    y2 = CSRPblock(y1, 512, 8)
    y3 = CSRPblock(y2, 1024, 4)
    return [y3, y2, y1]


# Callable instance with anchors and num_classes as parameter when called.
train = Dataset('Data.v1.tfrecord/train/train.tfrecord')
test = Dataset('/home/levy/Documents/Yolo_project/Data.v1.tfrecord/test/test.tfrecord')

anchors = np.array([[0.079327, 0.050481],
                    [0.048077, 0.055288],
                    [0.076923, 0.086538],
                    [0.098558, 0.103365],
                    [0.062500, 0.067308]])
num_classes = 6
log_dir = '../Log'
batch = 4

generator = test.generator_wrapper(anchors, batch, num_classes)
inputs = keras.Input((416, 416, 3))
model = darknet_body(next(generator)[0])

with tf.GradientTape as tape:
    logits = darknet_body(next(generator)[0])
    tape = tape.gradient(logits, darknet_body(next(generator)[0]))
    print(tape)