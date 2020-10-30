import tensorflow as tf
from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Concatenate, LeakyReLU, BatchNormalization, UpSampling2D
from tensorflow.keras.regularizers import l2
from tensorflow import split


class DarknetConv2D_BN_Leaky(tf.keras.layers.Layer):
    """Convolutional layer with batch normalization and leaky relu"""

    def __init__(self, filters, size):
        super(DarknetConv2D_BN_Leaky, self).__init__()
        self.batch_norm = BatchNormalization()
        self.leaky_relu = LeakyReLU(0.1)
        self.conv_1 = Conv2D(filters, size, padding='same', kernel_regularizer=l2(5e-4))

    def call(self, *args):
        x = self.conv_1(*args)
        x = self.batch_norm(x)
        return self.leaky_relu(x)


class DownSampling(tf.keras.layers.Layer):
    """Downsample with zeropadding and a convolution with 2x2 stride"""

    def __init__(self, filters):
        super(DownSampling, self).__init__()
        self.zeropadding = ZeroPadding2D(((1, 0), (1, 0)))
        self.batch_norm = BatchNormalization()
        self.leaky_relu = LeakyReLU(0.1)
        self.conv_strides2 = Conv2D(filters,
                                    kernel_size=(3, 3),
                                    strides=(2, 2),
                                    padding='valid',
                                    kernel_regularizer=l2(5e-4))

    def call(self, *args):
        x = self.zeropadding(*args)
        x = self.conv_strides2(x)
        x = self.batch_norm(x)
        return self.leaky_relu(x)


class CSPRblock(tf.keras.layers.Layer):
    """Create a identity block"""

    def __init__(self, filters, num_block):
        super(CSPRblock, self).__init__()
        self.DarknetConv2D_BN_Leaky = DarknetConv2D_BN_Leaky(filters, (3, 3))
        self.Downsample = DownSampling(filters)
        self.concatenate = Concatenate(axis=0)
        self.splitter = lambda x: split(x, num_or_size_splits=2, axis=0)
        self.num_block = num_block

    def call(self, *args):
        x = self.Downsample(*args)
        x1, x2 = self.splitter(x)

        for block in range(self.num_block):
            x = self.DarknetConv2D_BN_Leaky(x1)
            x1 += x
        return self.concatenate([x1, x2])


class Darknet_53(tf.keras.Model):
    """The 52 layers of Darknet 53"""

    def __init__(self):
        super(Darknet_53, self).__init__()
        self.conv_1 = DarknetConv2D_BN_Leaky(32, (3, 3))
        self.block = dict()
        self.block[0] = CSPRblock(64, 1)
        self.block[1] = CSPRblock(128, 2)
        self.block[2] = CSPRblock(256, 8)
        self.block[3] = CSPRblock(512, 8)
        self.block[4] = CSPRblock(1024, 4)

    def call(self, *args):
        y = []
        x = self.conv_1(*args)
        for i in range(5):
            x = self.block[i](x)
            y.append(x)
        y = y[2:]
        return y[::-1]


class PredictionsLayer(tf.keras.layers.Layer):
    """Last layer that will make the predictions"""

    def __init__(self, filters, num_predictions):
        super(PredictionsLayer, self).__init__()
        self.make_prediction = Conv2D(num_predictions,
                                      kernel_size=(3, 3),
                                      kernel_regularizer=l2(5e-4),
                                      padding='same')
        self.conv_1 = DarknetConv2D_BN_Leaky(filters * 2, (3, 3))
        self.conv_2 = DarknetConv2D_BN_Leaky(filters, (1, 1))

    def call(self, *args):
        x = self.conv_2(*args)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_1(x)
        x = self.conv_2(x)

        y = self.conv_1(x)
        y = self.make_prediction(y)

        return x, y


class Yolo_Body(tf.keras.Model):
    def __init__(self, num_anchors, num_classes):
        super(Yolo_Body, self).__init__()
        self.prediction1 = PredictionsLayer(512, num_anchors * (num_classes + 5))
        self.prediction2 = PredictionsLayer(256, num_anchors * (num_classes + 5))
        self.prediction3 = PredictionsLayer(128, num_anchors * (num_classes + 5))
        self.conv_1a = DarknetConv2D_BN_Leaky(256, (1, 1))
        self.conv_1b = DarknetConv2D_BN_Leaky(128, (1, 1))
        self.conv_2a = DarknetConv2D_BN_Leaky(512, (1, 1))
        self.conv_2b = DarknetConv2D_BN_Leaky(256, (1, 1))
        self.upsampling = UpSampling2D(2)
        self.concatenate = Concatenate()
        self.darknet = Darknet_53()

    def call(self, *args):
        """
        Parameters
        ----------
        input = list of 3 tensor for the 3 last stage of darknet top to bottom
        """

        inputs = self.darknet(*args)

        x, y1 = self.prediction1(inputs[0])

        x = self.conv_1a(x)
        x = self.upsampling(x)
        x = self.concatenate([x, inputs[1]])
        x = self.conv_2a(x)

        x, y2 = self.prediction2(x)

        x = self.conv_1b(x)
        x = self.upsampling(x)
        x = self.concatenate([x, inputs[2]])
        x = self.conv_2b(x)

        x, y3 = self.prediction3(x)

        return [y1, y2, y3]
