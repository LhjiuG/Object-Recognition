import tensorflow as tf
from tensorflow.keras.layers import Conv2D, BatchNormalization, LeakyReLU, ZeroPadding2D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model


class DarknetConv2DBNLeaky(tf.keras.layers.Layer):
    """Convolutional layer with batch normaltion and leaky relu"""

    def __init__(self, filters, k_size, strides=None, padding=None):
        super(DarknetConv2DBNLeaky, self).__init__()
        self.conv_1 = Conv2D(filters=filters,
                             kernel_size=k_size,
                             kernel_regularizer=l2(5e-4),
                             padding=padding if padding is not None else 'same',
                             strides=strides if strides is not None else (1, 1))
        self.batch_normalization_1 = BatchNormalization()
        self.leaky_relu_1 = LeakyReLU(alpha=0.1)

    def call(self, *args):
        x = self.conv_1(args)
        x = self.batch_normalization_1(x)
        return self.leaky_relu_1(x)


class DownSampling(tf.keras.layers.Layer):
    """Downsample using zeropadding((1, 0), (1, 0)) and a conv with strides 2"""

    def __init__(self, filters):
        super(DownSampling, self).__init__()
        self.conv_2 = DarknetConv2DBNLeaky(filters=filters,
                                           k_size=(3, 3),
                                           strides=(2, 2),
                                           padding='valid')
        self.zeropadding_1 = ZeroPadding2D(((1, 0), (1, 0)))

    def call(self, *args):
        x = self.zeropadding_1(*args)
        return self.conv_2(x)


class CSPRblock(tf.keras.layers.Layer):
    """Create an identity block for CSPRDarknet"""

    def __init__(self, filters, num_block):
        super(CSPRblock, self).__init__()
        self.conv_3 = DarknetConv2DBNLeaky(filters=filters, k_size=(3, 3))
        self.downsample_1 = DownSampling(filters)
        self.concat_1 = Concatenate(axis=0)
        self.splitter = lambda x: tf.split(x, num_or_size_splits=2, axis=0)
        self.num_block = num_block

    def call(self, *args):
        x = self.downsample_1(*args)
        x1, x2 = self.splitter(x)
        for block in range(self.num_block):
            x = self.conv_3(x1)
            x1 += x
        return self.concat_1([x1, x2])


class Darknet53(tf.keras.layers.Layer):
    """All of the CSPRDarknet_53 but the last layer"""

    def __init__(self):
        super(Darknet53, self).__init__()
        self.conv_4 = DarknetConv2DBNLeaky(32, (3, 3))
        self.block = dict()
        self.block[0] = CSPRblock(64, 1)
        self.block[1] = CSPRblock(128, 2)
        self.block[2] = CSPRblock(256, 8)
        self.block[3] = CSPRblock(512, 8)
        self.block[4] = CSPRblock(1024, 4)

    # def call(self, *args):
    #     y = []
    #     x = self.conv_4(*args)
    #     for i in range(len(self.block)):
    #         x = self.block[i](x)
    #         y.append(x)
    #     y = y[-3:]
    #     return y[::-1]

    def call(self, *args):
        x = self.conv_4(*args)
        x = self.block[0](x)
        x = self.block[1](x)
        y1 = self.block[2](x)
        y2 = self.block[3](y1)
        y3 = self.block[4](y2)
        return [y3, y2, y1]


class PredictionLayer(tf.keras.layers.Layer):
    """Layer that make the predictions"""

    def __init__(self, filters, num_classes):
        super(PredictionLayer, self).__init__()
        self.conv_5 = Conv2D(filters=num_classes * (num_classes + 5),
                             kernel_size=(3, 3),
                             kernel_regularizer=l2(5e-4),
                             padding='same')
        self.conv_6 = DarknetConv2DBNLeaky(filters * 2, (3, 3))
        self.conv_7 = DarknetConv2DBNLeaky(filters, (1, 1))

    def call(self, *args):
        x = self.conv_7(*args)  # (1, 1)
        x = self.conv_6(x)  # (3, 3)
        x = self.conv_7(x)  # (1, 1)
        x = self.conv_6(x)  # (3, 3)
        x = self.conv_7(x)  # (1, 1)

        y = self.conv_6(x)  # (3, 3)
        y = self.conv_5(y)

        return x, y


