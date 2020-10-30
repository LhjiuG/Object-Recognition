from tensorflow.keras.layers import Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

import numpy as np

from Object_Detection.Archive.yolo_v2 import yolo_body, yolo_loss
from Data_loading_v7 import load_boxes, load_image, get_training_path, preprocess_true_box
from config import *


def _main():
    path_list = list(get_training_path())
    anchors = np.array([[0.079327, 0.050481],
                        [0.048077, 0.055288],
                        [0.076923, 0.086538],
                        [0.098558, 0.103365],
                        [0.062500, 0.067308]])
    num_anchors = anchors.shape[0]
    classes = ['Bomb',
               'Enemy_1',
               'Enemy_2',
               'Spawner',
               'player']
    num_classes = len(classes)
    input_shape = (608, 608)
    log_dir = 'Log'

    image_input = Input(shape=(None, None, 3))
    h = [input_shape[0] // {0: 32, 1: 16, 2: 8}[l] for l in range(3)]
    w = [input_shape[1] // {0: 32, 1: 16, 2: 8}[l] for l in range(3)]
    y_true = [Input(shape=(h[l], w[l], num_anchors, num_classes + 5)) for l in range(3)]

    model_body = yolo_body(image_input, num_anchors, num_classes)
    print('Create YOLOv3 model with {} anchors and {} classes.'.format(num_anchors, num_classes))

    model_loss = Lambda(yolo_loss, output_shape=(1,), name="yolo_loss",
                        arguments={'anchors': anchors, 'num_classes': num_classes, 'ignore_thresh': 0.5})(
        [*model_body.output, *y_true])

    model = Model([model_body.input, *y_true], model_loss)

    logging = TensorBoard(log_dir=log_dir)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    checkpoint = ModelCheckpoint(log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                 monitor='val_loss', save_weights_only=True, save_best_only=True, period=3)

    val_split = 0.1
    np.random.seed(10101)
    np.random.shuffle(path_list)
    num_val = int(len(path_list) * val_split)
    num_train = len(path_list) - num_val

    model.compile(optimizer=Adam(lr=1e-3), loss={'yolo_loss': lambda y_true, y_pred: y_pred})

    batch_size = 8

    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    model.fit(data_generator_wrapper(path_list[:num_train], batch_size, input_shape, anchors, num_classes),
              steps_per_epoch=max(1, num_train // batch_size),
              validation_data=data_generator_wrapper(path_list[num_train:], batch_size, input_shape, anchors,
                                                     num_classes),
              validation_steps=max(1, num_val // batch_size),
              epochs=50,
              initial_epoch=0,
              callbacks=[logging, checkpoint]
              )
    model.save_weights(log_dir + 'trained_weights.h5')


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
        yield [image_data, *y_true], np.zeros(batch_size)


def data_generator_wrapper(path_list, batch_size, input_shape, anchors, num_classes):
    n = len(path_list)
    if n == 0 or batch_size <= 0: return None
    return generator_test(path_list, batch_size, input_shape, anchors, num_classes)


if __name__ == '__main__':
    _main()
