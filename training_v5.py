from tensorflow.keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from darknet_v3 import *
from Yolo_v3 import Yolo_Body, Yolo_loss

from data_pipeline import *


def _main():
    # Callable instance with anchors and num_classes as parameter when called.
    train = Dataset('Data.v1.tfrecord/train/train.tfrecord')
    test = Dataset('Data.v1.tfrecord/test/test.tfrecord')

    anchors = np.array([[0.079327, 0.050481],
                        [0.048077, 0.055288],
                        [0.076923, 0.086538],
                        [0.098558, 0.103365],
                        [0.062500, 0.067308]])
    num_classes = 6
    log_dir = 'Log'
    batch = 4

    logging = TensorBoard(log_dir=log_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, save_freq=3)


    model_body = Darknet53()
    _ = model_body(np.zeros([4, 416, 416, 3]))
    model_optimizer = tf.keras.optimizers.Adam(lr=0.001)
    model_loss = Yolo_loss(anchors, num_classes, ignore_tresh=.5)


if __name__ == '__main__':
    _main()
