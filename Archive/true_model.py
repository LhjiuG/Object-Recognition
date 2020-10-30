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


class CSRPBlock(keras.Model):

    def __init__(self, filters):
        super(CSRPBlock, self).__init__()
        self.conv = DarknetConv2D_BN_Leaky(filters // 2, (3, 3))
        self.zeropad = ZeroPadding2D(((1, 0), (1, 0)))
        self.down_sampling = DarknetConv2D_BN_Leaky(filters, (3, 3), strides=(2, 2))
        self.concat_1 = Concatenate(axis=0)
        self.splitter = lambda x: tf.split(x, num_or_size_splits=2, axis=-1)

    def call(self, inputs, **kwargs):
        x = self.zeropad(inputs)
        x = self.down_sampling(x)
        x1, x2 = self.splitter(x)
        for i in range(kwargs['num_blocks']):

            x = self.conv(x1)
            x1 += x
        return self.concat_1([x1, x2])


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

image = next(generator)[0]
inputs = keras.Input((None, None, 3))
test2 = CSRPBlock(32)
model = keras.Model(inputs, test2(inputs, num_blocks=2)).trainable_weights
print(model)
