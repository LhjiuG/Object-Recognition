from abc import ABC

import tensorflow as tf
import numpy as np
import itertools
from collections import defaultdict


def find_padding(generator):
    """In a generator, find the maximum of rows of all the array for padding."""
    comparator = 0
    for array in generator:
        if array.shape[0] > comparator:
            comparator = array.shape[0]
        else:
            continue
    return int(comparator)


# Features that are present in the tfrecord file.
features_descriptions = {
    'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True,
                                                            default_value=0.0),
    'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True,
                                                            default_value=0.0),
    'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True,
                                                            default_value=0.0),
    'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True,
                                                            default_value=0.0),
    'image/object/class/label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True,
                                                              default_value=0),
    'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True,
                                                             default_value=''),
    'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value='')
}


class Dataset:

    def __init__(self, tfrecord):
        self.raw_dataset = tf.data.TFRecordDataset(tfrecord)
        self.parsed_dataset = self.raw_dataset.map(
            lambda x: tf.io.parse_example(x, features_descriptions))

    def _decoded_image(self, batch):
        image_func = lambda x: tf.cast(tf.io.decode_jpeg(x['image/encoded']), tf.float32)
        return self.parsed_dataset.map(image_func).batch(batch)

    def _boxes(self):
        boxes_dataset = self.parsed_dataset.map(lambda x: {
            'ymin': x['image/object/bbox/ymin'],
            'xmin': x['image/object/bbox/xmin'],
            'ymax': x['image/object/bbox/ymax'],
            'xmax': x['image/object/bbox/xmax'],
            'class': x['image/object/class/label']
        }).as_numpy_iterator()

        for img_boxes in boxes_dataset:
            boxes_dict = defaultdict(list)
            for name in img_boxes:
                for t, values in enumerate(img_boxes[name]):
                    boxes_dict[t].append(values)
            yield np.array(list(boxes_dict.values()))

    def _preprocess_true_boxes(self, anchors, batch, num_classes):

        boxes_gen, padding_gen = itertools.tee(self._boxes())
        # Only used to get the input shape
        parsed_dataset = self.parsed_dataset.as_numpy_iterator().next()

        # Convert to tensor for compatibility.
        input_shape = tf.convert_to_tensor([parsed_dataset['image/width'], parsed_dataset['image/height']])
        num_layers = 3

        true_boxes = tf.data.Dataset.from_generator(lambda: boxes_gen, tf.float32).padded_batch(batch, padded_shapes=(
            find_padding(padding_gen), 5))
        # TODO: Change from mins max to xywh
        # Iterate over all the batch
        for boxes in true_boxes:
            # Take batch size here because the leftover batch size might be different.
            m = boxes.shape[0]

            # Expand dims for broadcasting.
            anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)
            _anchors = tf.expand_dims(anchors, 0)
            anchors_maxes = _anchors / 2.
            anchors_mins = -anchors_maxes

            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
            y_true = [
                np.zeros((m, grid_shapes[l][1], grid_shapes[l][0], len(anchors), 5 + num_classes), dtype='float32') for
                l in range(num_layers)]

            for b, one_boxes in enumerate(boxes):
                box_wh = one_boxes[..., 2:4] - one_boxes[..., 0:2]  # mins - maxes
                valid_mask = box_wh[..., 0] > 0
                wh = box_wh[valid_mask]
                if len(wh) == 0:
                    continue
                wh = tf.expand_dims(wh, -2)
                box_maxes = wh / 2.
                box_mins = -box_maxes

                intersect_mins = tf.math.maximum(box_mins, anchors_mins)
                intersect_maxes = tf.math.minimum(box_maxes, anchors_maxes)
                intersect_wh = tf.math.maximum(intersect_maxes - intersect_mins, 0.)
                intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
                box_area = wh[..., 0] * wh[..., 1]
                anchor_area = _anchors[..., 0] * _anchors[..., 1]
                iou = intersect_area / (box_area + anchor_area - intersect_area)
                best_anchors = tf.math.argmax(iou, axis=-1)
                for t, n in enumerate(best_anchors):
                    for l in range(num_layers):
                        i = np.array(
                            tf.cast(tf.math.floor(boxes[b, t, 0] * tf.cast(grid_shapes[l][1], tf.float32)), tf.int32))
                        j = np.array(
                            tf.cast(tf.math.floor(boxes[b, t, 1] * tf.cast(grid_shapes[l][1], tf.float32)), tf.int32))
                        k = np.array(n)
                        c = np.array(tf.cast(boxes[b, t, 4], tf.int32))
                        y_true[l][b, j, i, k, 0:4] = boxes[b, t, 0:4]
                        y_true[l][b, j, i, k, 4] = 1
                        y_true[l][b, j, i, k, 4 + c] = 1  # start at 4 because the smallest class index is 1 and not 0

            yield tf.convert_to_tensor(y_true[0]), tf.convert_to_tensor(y_true[1]), tf.convert_to_tensor(y_true[2])

    def generator_wrapper(self, *args):
        anchors, batch, num_classes = args
        boxes_gen = self._preprocess_true_boxes(anchors, batch, num_classes)
        image_gen = self._decoded_image(batch).as_numpy_iterator()
        for image, boxes in zip(image_gen, boxes_gen):
            yield np.array(image), boxes

# TODO: Get the x, y generator ready and look up custom gradient loop
