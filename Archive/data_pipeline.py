import tensorflow as tf
import os
import numpy as np
from functools import *
import sys

test_serialized = 'Data.v1.tfrecord/test/test.tfrecord'


# test_raw_dataset = tf.data.TFRecordDataset(test_serialized)
#
# # Create a description of the features.
# feature_description = {
#     'image/height': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'image/width': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'image/filename': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'image/format': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'image/object/bbox/ymax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
#     'image/object/bbox/xmax': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
#     'image/object/bbox/ymin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
#     'image/object/bbox/xmin': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True, default_value=0.0),
#     'image/object/class/label': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True, default_value=0),
#     'image/object/class/text': tf.io.FixedLenSequenceFeature([], tf.string, allow_missing=True, default_value=''),
#     'image/encoded': tf.io.FixedLenFeature([], tf.string, default_value='')
# }
#
# anchors = np.array([[0.079327, 0.050481],
#                     [0.048077, 0.055288],
#                     [0.076923, 0.086538],
#                     [0.098558, 0.103365],
#                     [0.062500, 0.067308]])
# num_anchors = len(anchors)
# classes = ['Bomb', 'Enemy_1', 'Enemy_2', 'Spawner', 'player', 'aa']
# num_classes = len(classes)
# input_shape = (608, 608)
#
#
# # def _features_parsing(example_proto):
# #     return tf.io.parse_single_example(example_proto, feature_description)
# #
# #
# # def _decode_image(example_proto):
# #     return tf.io.decode_jpeg(example_proto['image/encoded'])
#
#
# # def _get_raw_boxes(example_proto):
# #     boxes = {
# #         'ymin': example_proto['image/object/bbox/ymin'],
# #         'xmin': example_proto['image/object/bbox/xmin'],
# #         'ymax': example_proto['image/object/bbox/ymax'],
# #         'xmax': example_proto['image/object/bbox/xmax'],
# #         'class': example_proto['image/object/class/label']
# #     }
# #     return boxes
#
#
# def preprocess_true_box(true_boxes, anchors, input_shape, num_classes):
#     num_layers = 3
#     true_boxes = np.array(true_boxes, dtype='float32')
#     input_shape = np.array(input_shape, dtype='int32')
#     boxes_xy = np.array((true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2)
#     boxes_wh = np.array(true_boxes[..., 2:4] - true_boxes[..., 0:2])
#
#     m = true_boxes.shape[0]
#     grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
#     y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchors), 5 + num_classes), dtype='float32') for l
#               in range(num_layers)]
#
#     # Expand dim to apply broadcasting.
#     anchors = np.expand_dims(anchors, 0)
#     anchor_maxes = anchors / 2.
#     anchor_mins = -anchor_maxes
#     valid_mask = boxes_wh[..., 0] > 0
#
#     for b in range(m):
#
#         # Discard zero rows.
#         wh = boxes_wh[b, valid_mask[b]]
#         # skip over image with no bounding box
#         if len(wh) == 0: continue
#         # Expand dim to apply broadcasting.
#         wh = np.expand_dims(wh, -2)
#         box_maxes = wh / 2.
#         box_mins = -box_maxes
#
#         intersect_mins = np.maximum(box_mins, anchor_mins)
#         intersect_maxes = np.minimum(box_maxes, anchor_maxes)
#         intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
#         intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
#         box_area = wh[..., 0] * wh[..., 1]
#         anchor_area = anchors[..., 0] * anchors[..., 1]
#         iou = intersect_area / (box_area + anchor_area - intersect_area)
#
#         # Find best anchor for each true box
#         best_anchor = np.argmax(iou, axis=-1)
#
#         for t, n in enumerate(best_anchor):
#             for l in range(num_layers):
#                 i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
#                 j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
#                 k = n
#                 c = true_boxes[b, t, 4].astype('int32')
#                 y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
#                 y_true[l][b, j, i, k, 4] = 1
#                 y_true[l][b, j, i, k, 4 + c] = 1  # start at 4 because the smallest class index is 1 and not 0
#
#     return y_true
#
#
# def foo(boxes):
#     for batch in boxes:
#         boxes_as_array = np.zeros((max_shape[1], max_shape[2], 5))
#         batch_array = np.array(list(dict.values(batch)))
#         for i in range(batch_array.shape[1]):
#             for t in range(batch_array.shape[2]):
#                 ymin = batch_array[0][i][t]
#                 xmin = batch_array[1][i][t]
#                 ymax = batch_array[2][i][t]
#                 xmax = batch_array[3][i][t]
#                 classes = batch_array[4][i][t]
#                 boxes_as_array[i, t, 0:] = np.array([ymin, xmin, ymax, xmax, classes])
#         yield boxes_as_array
#
#
# def preprocessing_generator(boxes):
#     for batch in boxes.take(-1):
#         yield test(batch)
#
#
# # Deserialize dataset
# parsed_dataset = test_raw_dataset.map(_features_parsing)
# image_input = parsed_dataset.map(_decode_image).batch(8)
# boxes_dict = parsed_dataset.map(_get_raw_boxes).padded_batch(8)
#
# # Preprocessing stage 1
# boxes_as_batch = boxes_dict.as_numpy_iterator()
# foo_1 = list()
# num_of_batch = len([foo_1.append('1') for i in boxes_as_batch])
# shape_list = list()
# for batch in boxes_dict.take(-1):
#     as_array = np.array(list(dict.values(batch)))
#     shape_list.append(as_array.shape)
#
# max_shape = np.max(np.array(shape_list), axis=0)  # get the maximum amount of boxes and image in batch
# boxes_as_gen = lambda: foo(boxes_dict.take(-1))
# boxes_as_dataset = tf.data.Dataset.from_generator(boxes_as_gen, output_types=tf.float32)
# test = partial(preprocess_true_box, anchors=anchors, num_classes=num_classes, input_shape=input_shape)
# preprocess_gen = lambda: preprocessing_generator(boxes_as_dataset)
# preprocessed_dataset = tf.data.Dataset.from_generator(preprocess_gen)
# for i in preprocessed_dataset.as_numpy_iterator():
#     print(i)
class Data:
    def __init__(self):
        self.anchors = np.array([[0.079327, 0.050481],
                                 [0.048077, 0.055288],
                                 [0.076923, 0.086538],
                                 [0.098558, 0.103365],
                                 [0.062500, 0.067308]])
        self.num_anchors = len(self.anchors)
        self.classes = ['Bomb', 'Enemy_1', 'Enemy_2', 'Spawner', 'aa', 'player']
        self.num_classes = len(self.classes)
        self.input_shape = (608, 608)


class FeaturesDescriptions:
    """ This return a dictionary of features for tfrecord parsing"""

    def __init__(self):
        self.features_descriptions = {
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


class ParsedDataset:
    """
    Class that read and parse the initial tfrecord file.
    """

    def __init__(self, tfrecord_path):
        self.raw_dataset = tf.data.TFRecordDataset(tfrecord_path)
        self.features_descriptions = FeaturesDescriptions().features_descriptions
        self.parsed_dataset = self.raw_dataset.map(self.features_parsing)

    def features_parsing(self, args):
        return tf.io.parse_single_example(args, self.features_descriptions)


class ImageInput(ParsedDataset):
    """Hold the Decoded images"""

    def __init__(self, tfrecord_path):
        super().__init__(tfrecord_path)

    def decoded_image(self):
        return tf.io.decode_jpeg(self.parsed_dataset['image/encoded'])


class BoxesDataset(ParsedDataset, Data):
    """Preprocessed the ground Truth"""

    def __init__(self, tfrecord_path):
        super().__init__(tfrecord_path)
        Data.__init__(self)
        self.batch_boxes = self.parsed_dataset.map(self.get_raw_boxes).padded_batch(8)
        self.num_batch = self.get_num_batch()
        self.max_boxes = self.max_num_boxes_batch()
        self.boxes_array_dataset = tf.data.Dataset.from_generator(
            lambda: self.boxes_as_array(self.batch_boxes.take(-1)), output_types=tf.float32)
        self.ground_truth = tf.data.Dataset.from_generator(lambda: self.ground_truth_gen(), output_types=tf.string)

    @staticmethod
    def get_raw_boxes(args):
        boxes = {
            'ymin': args['image/object/bbox/ymin'],
            'xmin': args['image/object/bbox/xmin'],
            'ymax': args['image/object/bbox/ymax'],
            'xmax': args['image/object/bbox/xmax'],
            'class': args['image/object/class/label']
        }
        return boxes

    def get_num_batch(self):
        """Get the amount of batch in the dataset"""
        boxes_as_batch = self.batch_boxes
        counter = 0
        for batch in boxes_as_batch.take(-1):
            counter += 1
        return counter

    def max_num_boxes_batch(self):
        """Get the maximum amount of boxes in a images in all the dataset"""
        shape_list = list()
        for batch in self.batch_boxes.take(-1):
            as_array = np.array(list(dict.values(batch)))
            shape_list.append(as_array.shape)
        return np.max(np.array(shape_list), axis=0)

    def boxes_as_array(self, boxes_dataset):
        for batch in boxes_dataset:
            boxes_as_array = np.zeros((self.max_boxes[1], self.max_boxes[2], 5))
            batch_as_array = np.array(list(dict.values(batch)))
            for i in range(batch_as_array.shape[1]):
                for t in range(batch_as_array.shape[2]):
                    ymin = batch_as_array[0][i][t]
                    xmin = batch_as_array[1][i][t]
                    ymax = batch_as_array[2][i][t]
                    xmax = batch_as_array[3][i][t]
                    classes = batch_as_array[4][i][t]
                    boxes_as_array[i, t, 0:] = np.array([ymin, xmin, ymax, xmax, classes])
            yield boxes_as_array

    def preprocess_true_box(self, true_boxes, anchors, input_shape, num_classes):
        num_layers = 3
        true_boxes = np.array(true_boxes, dtype='float32')
        input_shape = np.array(input_shape, dtype='int32')
        boxes_xy = np.array((true_boxes[..., 0:2] + true_boxes[..., 2:4]) / 2)
        boxes_wh = np.array(true_boxes[..., 2:4] - true_boxes[..., 0:2])

        m = true_boxes.shape[0]
        grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
        y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchors), 5 + num_classes), dtype='float32')
                  for l
                  in range(num_layers)]

        # Expand dim to apply broadcasting.
        anchors = np.expand_dims(anchors, 0)
        anchor_maxes = anchors / 2.
        anchor_mins = -anchor_maxes
        valid_mask = boxes_wh[..., 0] > 0

        for b in range(m):

            # Discard zero rows.
            wh = boxes_wh[b, valid_mask[b]]
            # skip over image with no bounding box
            if len(wh) == 0: continue
            # Expand dim to apply broadcasting.
            wh = np.expand_dims(wh, -2)
            box_maxes = wh / 2.
            box_mins = -box_maxes

            intersect_mins = np.maximum(box_mins, anchor_mins)
            intersect_maxes = np.minimum(box_maxes, anchor_maxes)
            intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
            intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
            box_area = wh[..., 0] * wh[..., 1]
            anchor_area = anchors[..., 0] * anchors[..., 1]
            iou = intersect_area / (box_area + anchor_area - intersect_area)

            # Find best anchor for each true box
            best_anchor = np.argmax(iou, axis=-1)

            for t, n in enumerate(best_anchor):
                for l in range(num_layers):
                    i = np.floor(true_boxes[b, t, 0] * grid_shapes[l][1]).astype('int32')
                    j = np.floor(true_boxes[b, t, 1] * grid_shapes[l][0]).astype('int32')
                    k = n
                    c = true_boxes[b, t, 4].astype('int32')
                    y_true[l][b, j, i, k, 0:4] = true_boxes[b, t, 0:4]
                    y_true[l][b, j, i, k, 4] = 1
                    y_true[l][b, j, i, k, 4 + c] = 1  # start at 4 because the smallest class index is 1 and not 0

        return y_true

    def ground_truth_gen(self):
        true_boxes = self.boxes_array_dataset.as_numpy_iterator()
        for batch in true_boxes:
            ground_truth = self.preprocess_true_box(batch, self.anchors, self.input_shape, self.num_classes)
            yield ground_truth


class InputDataset:
    """Class for the data that is ready to enter the model"""

    def __init__(self, tfrecord_path):
        ground_truth_classes = BoxesDataset(tfrecord_path)


test = BoxesDataset('/Data.v1.tfrecord/test/test.tfrecord')
test2 = test.boxes_array_dataset.as_numpy_iterator()
anchors = Data().anchors
num_classes = Data().num_classes


def aa(*args, **kwargs):
    anchors, num_classes = args
    print(anchors)
    print(num_classes)


aa(anchors, num_classes)
