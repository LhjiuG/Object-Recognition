from abc import ABC

import tensorflow as tf
import numpy as np

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


class Inputs:
    """
    This class take in the tfrecord path and then return a generator when the instance is called.
    The input returned by the generator are already in batch.

    For now, the anchors and the number of classes must be passed when the instance is called.
    """

    def __init__(self, tfrecord_path):
        self.raw_dataset = tf.data.TFRecordDataset(tfrecord_path)

    def get_parsed_dataset(self):
        """Parsed and make padded batch of the raw dataset"""
        parsed_dataset = self.raw_dataset.map(
            lambda x: tf.io.parse_single_example(x, features_descriptions))
        return parsed_dataset

    def get_decoded_images(self):
        """Decode the image to be ready to enter the model"""
        # The image being sent to tf.io.decode_jpeg in batch cause an error.
        parsed_dataset = self.get_parsed_dataset().unbatch()
        image_dataset = parsed_dataset.map(lambda x: tf.io.decode_jpeg(x['image/encoded'])).batch(8).as_numpy_iterator()
        for image in image_dataset:
            yield image

    def get_boxes(self):
        """Filter the parsed dataset and return only the boxes features"""
        parsed_dataset = self.get_parsed_dataset()
        boxes = parsed_dataset.map(lambda x: {
            'ymin': x['image/object/bbox/ymin'],
            'xmin': x['image/object/bbox/xmin'],
            'ymax': x['image/object/bbox/ymax'],
            'xmax': x['image/object/bbox/xmax'],
            'class': x['image/object/class/label']
        })
        return boxes

    def get_num_batch(self):
        """Get the total number of batch"""
        boxes = self.get_boxes()
        counter = 0
        for i in boxes.take(-1):
            counter += 1
        return counter

    def maximum_boxes_num(self):
        """Get the maximum amount of boxes an images can have in all the dataset"""
        boxes = self.get_boxes()
        shape_list = list()
        for i in boxes.take(-1):
            as_array = np.array(list(dict.values(i)))
            shape_list.append(as_array.shape)
        return np.max(np.array(shape_list), axis=0)

    def boxes_as_array(self, boxes):
        """Reorder and return the boxes as array"""
        # boxes = self.get_boxes()
        max_boxes = self.maximum_boxes_num()
        for batch in boxes.take(-1):
            true_batch = np.zeros((max_boxes[1], max_boxes[2], 5))
            batch_as_array = np.array(list(dict.values(batch)))
            for i in range(batch_as_array.shape[1]):
                for t in range(batch_as_array.shape[2]):
                    ymin = batch_as_array[0][i][t]
                    xmin = batch_as_array[1][i][t]
                    ymax = batch_as_array[2][i][t]
                    xmax = batch_as_array[3][i][t]
                    classes = batch_as_array[4][i][t]
                    true_batch[i, t, 0:] = np.array([ymin, xmin, ymax, xmax, classes])
            yield true_batch

    def placeholder(self, boxes):
        ymin = boxes['ymin']
        xmin = boxes['xmin']
        ymax = boxes['ymax']
        xmax = boxes['xmax']
        _class = tf.cast(boxes['class'], dtype=ymax.dtype)
        stack = tf.stack([ymin, xmin, ymax, xmax, _class])
        list1 = list()
        for i in range(stack.shape[1]):
            list1.append([stack[0][i], stack[1][i], stack[2][i], stack[3][i], stack[4][i]])
        return np.array(list1)


    def preprocess_true_box(self, anchors, num_classes):
        """Preprocess the ground truth to be ready to enter the model"""
        parsed_dataset = next(self.get_parsed_dataset().as_numpy_iterator())
        input_shape = np.array([parsed_dataset['image/width'][0], parsed_dataset['image/height'][0]])
        boxes_as_array = self.boxes_as_array()
        num_layers = 3
        while True:
            true_boxes = next(boxes_as_array)
            boxes_wh = np.array(true_boxes[..., 2:4] - true_boxes[..., 0:2])

            m = true_boxes.shape[0]
            grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]
            y_true = [
                np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], len(anchors), 5 + num_classes), dtype='float32')
                for l
                in range(num_layers)]

            # Expand dim to apply broadcasting.
            anchors = anchors
            _anchors = np.expand_dims(anchors, 0)
            anchor_maxes = _anchors / 2.
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
                anchor_area = _anchors[..., 0] * _anchors[..., 1]
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

            yield y_true

    def __call__(self, *args):
        """

        Parameters
        ----------
        args = anchors, num_classes

        Returns
        -------
        generator = [x, y]
        """
        anchors, num_class = args
        boxes = self.preprocess_true_box(anchors, num_class)
        image = self.get_decoded_images()
        while True:
            _image = next(image)
            _boxes = next(boxes)
            yield _image, [*_boxes]


test_inputs = Inputs('../Data.v1.tfrecord/test/test.tfrecord')

boxxes = test_inputs.get_boxes().map(test_inputs.placeholder)
def _gen():
    while True:
        yield next(boxxes)

print(test_inputs.placeholder(next(_gen())))


