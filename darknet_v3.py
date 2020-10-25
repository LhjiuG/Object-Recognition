from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Concatenate
from tensorflow.keras.regularizers import l2

from utils import compose
from data_pipeline import *

train = Dataset('Data.v1.tfrecord/train/train.tfrecord')
test = Dataset('/home/levy/Documents/Yolo_project/Data.v1.tfrecord/test/test.tfrecord')

anchors = np.array([[0.079327, 0.050481],
                    [0.048077, 0.055288],
                    [0.076923, 0.086538],
                    [0.098558, 0.103365],
                    [0.062500, 0.067308]])
num_classes = 6
log_dir = 'Log'
batch = 4

generator = test.generator_wrapper(anchors, batch, num_classes)


class FullConv(tf.keras.layers.Layer):
    """Apply a convolution, batch norm and leaky relu."""

    def __init__(self, *args, **kwargs):
        """

        Parameters
        ----------
        args : filters, kernel_size
        kwargs : padding, strides
        """
        super(FullConv, self).__init__()
        self.darknet_conv = Conv2D(*args, **kwargs, kernel_regularizer=l2(5e-4), use_bias=False)
        self.bn = BatchNormalization(trainable=False)
        self.leaky = LeakyReLU(trainable=False)  # TODO: Test the impact of trainable false/true

    def call(self, inputs, **kwargs):
        x = self.darknet_conv(inputs)
        x = self.bn(x)
        return self.leaky(x)


class CSRPBlock(tf.keras.layers.Layer):
    """CSRPBlock for darknet model"""

    def __init__(self, filters):
        """

        Parameters
        ----------
        filters : int
        """
        super(CSRPBlock, self).__init__()
        # Convolutional Layers
        self.conv = FullConv(filters // 2, (3, 3), padding='same')
        self.down_sample = FullConv(filters, (3, 3), strides=(2, 2))
        # Transformations Layers
        self.padding = ZeroPadding2D(((1, 0), (1, 0)), trainable=False)
        self.concat = Concatenate(axis=-1, trainable=False)

    def call(self, inputs, **kwargs):
        """

        Parameters
        ----------
        inputs : tensor
        kwargs : num_blocks

        Returns
        -------
        Tensor
        """
        x = self.padding(inputs)
        x = self.down_sample(x)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=-1)
        for i in range(kwargs['num_blocks']):
            x = self.conv(x1)
            x1 += x
        return self.concat([x1, x2])


def darknet_body(inputs_tensor):
    filters = (64, 128, 256, 512, 1024)
    num_blocks = (1, 2, 8, 8, 4)
    x = FullConv(32, (3, 3), padding='same')(inputs_tensor)
    for filters_, num_blocks_ in zip(filters, num_blocks):
        x = CSRPBlock(filters_)(x, num_blocks=num_blocks_)
    return x


def darknet_body_test(inputs_tensor):
    """Only used for testing in eager mode"""
    outputs = list()
    filters = (64, 128, 256, 512, 1024)
    num_blocks = (1, 2, 8, 8, 4)
    x = FullConv(32, (3, 3), padding='same')(inputs_tensor)
    for filters_, num_blocks_ in zip(filters, num_blocks):
        x = CSRPBlock(filters_)(x, num_blocks=num_blocks_)
        outputs.append(x)
    return outputs[:1:-1]


def last_layers(inputs_tensor, filters, num_predictions):
    """Make the prediction for yolo_body"""
    conv1 = FullConv(filters, (1, 1), padding='same')
    conv2 = FullConv(filters * 2, (3, 3), padding='same')
    predictions = Conv2D(num_predictions, (1, 1), padding='same', kernel_regularizer=l2(5e-4), use_bias=False)
    x = compose(conv1, conv2, conv1, conv2, conv1)(inputs_tensor)
    y = compose(conv2, predictions)(x)
    return x, y
