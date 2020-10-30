import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input, BatchNormalization, LeakyReLU, ZeroPadding2D, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras import Model
from data_pipeline import *


class DarknetModel(tf.keras.models.Model):

    def __init__(self):
        super(DarknetModel, self).__init__()

    @staticmethod
    def darknet_conv_bn_leaky(filters, k_size, inputs, strides=None, padding=None):
        x = Conv2D(filters=filters,
                   kernel_size=k_size,
                   kernel_regularizer=l2(5e-4),
                   padding=padding if padding is not None else 'same',
                   strides=strides if strides is not None else (1, 1))(inputs)
        x = BatchNormalization()(x)
        return LeakyReLU(0.1)(x)

    def down_sampling(self, filters, inputs):
        x = ZeroPadding2D(((1, 0), (1, 0)))(inputs)
        return self.darknet_conv_bn_leaky(filters, (3, 3), inputs=x, strides=(2, 2), padding='valid')

    def csprblock(self, filters, num_blocks, inputs):
        x = self.down_sampling(filters, inputs)
        x1, x2 = tf.split(x, num_or_size_splits=2, axis=0)
        for block in range(num_blocks):
            x = self.darknet_conv_bn_leaky(filters, (3, 3), inputs=x1)
            x1 += x
        return Concatenate(axis=0)([x1, x2])

    def darknet_body(self, inputs):
        x = self.darknet_conv_bn_leaky(32, (3, 3), inputs)
        x = self.csprblock(64, 1, x)
        x = self.csprblock(128, 2, x)
        y1 = self.csprblock(256, 8, x)
        y2 = self.csprblock(512, 8, y1)
        y3 = self.csprblock(1024, 4, y2)
        return [y3, y2, y1]

    def call(self, inputs, training=None, mask=None):
        return self.darknet_body(inputs)


# Callable instance with anchors and num_classes as parameter when called.
train = Dataset('Data.v1.tfrecord/train/train.tfrecord')
test = Dataset('Data.v1.tfrecord/test/test.tfrecord')

anchors = np.array([[0.079327, 0.050481],
                    [0.048077, 0.055288],
                    [0.076923, 0.086538],
                    [0.098558, 0.103365],
                    [0.062500, 0.067308]])
num_classes = 6
log_dir = '../Log'
batch = 4

darknet = DarknetModel()

generator = test.generator_wrapper(anchors, batch, num_classes)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.CategoricalCrossentropy()
loss_history = []


def train_step(image):
    with tf.GradientTape() as tape:
        logits = darknet(image)
        grads = tape.gradient(logits, darknet.trainable_weights)
        print(grads)


for image in generator:
    train_step(tf.cast(image[0], tf.float32))

# for img in image:
#     print('Logits', darknet(tf.cast(img, tf.float32)))
