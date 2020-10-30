from Yolo_v3 import *
from Data_loading_v7 import *
from config import *

import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

path_list = list(get_training_path())
anchors = np.array([[0.079327, 0.050481],
                    [0.048077, 0.055288],
                    [0.076923, 0.086538],
                    [0.098558, 0.103365],
                    [0.062500, 0.067308]])
num_anchors = len(anchors)
classes = ['Bomb', 'Enemy_1', 'Enemy_2', 'Spawner', 'player']
num_classes = len(classes)
input_shape = (608, 608)
log_dir = 'Log'
val_split = 0.1

np.random.seed(10101)
np.random.shuffle(path_list)
num_val = int(len(path_list) * val_split)
num_train = len(path_list) - num_val

num_layers = 3
batch_size = 8
epochs = 10


def _main():
    model = Yolo_Body(num_anchors, num_classes)
    model.compile(
        optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-8),
        loss=Yolo_loss(anchors, num_classes, ignore_tresh=.5))
    model.fit(
        data_generator_wrapper(path_list[:num_train], batch_size, input_shape, anchors, num_classes),
        validation_data=data_generator_wrapper(path_list[num_train:], batch_size, input_shape, anchors, num_classes),
        epochs=epochs,
        verbose=2,
        callbacks=get_callback(log_dir))


def generator_test(path_list, batch_size, input_shape, anchors, num_classes):
    """ Generate the batch of image and ground truth for training"""
    n = len(path_list)
    i = 0
    while True:
        image_data = []
        box_data = []
        for b in range(batch_size):
            if i == 0:
                np.random.shuffle(path_list)
            image = load_image(f'{DIRECTORY.TRAIN_DIR}/{path_list[i][0]}')
            box = load_boxes(f'{DIRECTORY.TRAIN_DIR}/{path_list[i][1]}')
            image_data.append(image)
            box_data.append(box)
            i = (i + 1) % n
        image_data = np.concatenate(image_data)
        box_data = np.concatenate(box_data)
        y_true = preprocess_true_box(box_data, anchors, input_shape, num_classes)
        yield image_data, y_true, np.zeros(batch_size)


def data_generator_wrapper(path_list, batch_size, input_shape, anchors, num_classes):
    n = len(path_list)
    if n == 0 or batch_size <= 0: return None
    return generator_test(path_list, batch_size, input_shape, anchors, num_classes)


def get_callback(log_dir):
    logging = TensorBoard(log_dir=log_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)
    return [logging, reduce_lr, early_stopping, checkpoint]


if __name__ == '__main__':
    _main()



