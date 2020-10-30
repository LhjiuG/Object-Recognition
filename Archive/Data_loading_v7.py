import os
import numpy as np
from Object_Detection.Archive.config import *
from tensorflow import io, newaxis, float32
from tensorflow import image as img
import datetime
from keras import backend as K

max_param = 20
width, height = 608, 608


# Pyramid Layer 1
# ---------------
def get_training_path():
    """ Generator for the path of the images and annotations files

    output
    ------
    zipped_list : generator
    Return bundle of 1 annotations file path and its respective image file path
    """
    image_path = (jpg for jpg in sorted(os.listdir(DIRECTORY.TRAIN_DIR))
                  if jpg.endswith(".jpg"))
    boxes_path = (boxes for boxes in sorted(os.listdir(DIRECTORY.TRAIN_DIR))
                  if boxes.endswith(".txt"))
    zipped_list = (path for path in zip(image_path, boxes_path))
    return zipped_list


# Pyramid Layer 2
# ---------------
def load_image(image_path):
    """
    Convert all image into array of shape (1, width, height, 3)
    """
    image = io.read_file(image_path)
    image = img.decode_image(image, channels=3)
    image = img.convert_image_dtype(image, float32)

    image = img.resize(image, (width, height))
    image = image[newaxis, :]
    return image


def load_boxes(annotations_path, training=True):
    """ Get boxes[labels, x, y, width, height]"""
    boxes = np.loadtxt(annotations_path, delimiter=" ", dtype='float32')
    # Add a dimension if the image only has 1 boxes and therefore is (5, )
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, 0)
    # add a batch dimension and multiple boxes filled with zero to get the final shape of (1, max_param, 5)
    if training:
        zero_array = np.zeros((max_param, 5))
        boxes = np.expand_dims(np.concatenate((boxes, zero_array))[:max_param], 0)
        return np.roll(boxes, -1, axis=2)
    return np.roll(boxes, -1, axis=1)


# Pyramid Layer 3
# ---------------
def preprocess_true_box(true_boxes, anchors, input_shape, num_classes):
    """
    Parameters
    ----------
    true_boxes : array, shape=(batch, max_param, 5)
        List of ground truth boxes in form of relative class, x, y, w, h.
        Relative coordinates are in range [0, 1] indicating a percentage
        of the orignal image dimensions
    anchors : array, shape=(9, 2)
        List of anchors in form of w, h.
        Anchors are in the range of [0, 1] indicating a percentage of the
        original image dimensions.
    input_shape: array-like, hw, multiples of 32
    num_classes: integer
    """
    true_boxes = np.roll(np.array(true_boxes, dtype='float32'), -1, axis=2)
    input_shape = np.array(input_shape, dtype='int32')

    # Decimal Value relative to the original image
    box_wh = true_boxes[..., 2:4]
    # TODO: Change hardcoding
    num_layers = 3  # Defaults

    m = true_boxes.shape[0]
    # Default is [(19, 19), (38, 38), (76, 76)] for an input shape of (608, 608)
    grid_shapes = [input_shape // {0: 32, 1: 16, 2: 8}[l] for l in range(num_layers)]

    # the 2 last dimension represent the gt per cells
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], 5, 5 + num_classes),
                       dtype='float32') for l in range(num_layers)]

    anchors = K.expand_dims(anchors, 0)
    anchor_maxes = anchors / 2.
    anchor_mins = -anchor_maxes
    valid_mask = box_wh[..., 0] > 0

    for b in range(m):

        # Discard zero rows.
        wh = box_wh[b, valid_mask[b]]
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
                y_true[l][b, j, i, k, 5 + c] = 1

    return y_true


# External Pyramid Layer
# ----------------------
def anchors_generator(boxes_path, cluster):
    box_wh = np.zeros((1, 2))
    for i in range(len(boxes_path)):
        bbox = load_boxes(f'{DIRECTORY.TRAIN_DIR}/{boxes_path[i][1]}', False)[..., 2:4]
        if bbox.shape[1] != 0:
            box_wh = np.append(box_wh, bbox, axis=0)

    bbox = list()
    for box in box_wh[1:]:  # [1:] so we dont select the row of zero
        ymin = box[1] / 2.
        xmin = box[0] / 2.
        ymax = -ymin
        xmax = -xmin

        bbox.append([ymin, xmin, ymax, xmax])
    bbox = np.array(bbox)

    result = kmeans(bbox, k=cluster)

    anchors = []
    for box in result:
        width = box[1] * 2
        height = box[0] * 2

        anchors.append([width, height])

    return np.array(anchors)


def save_anchors(anchors):
    filename = datetime.datetime.now()
    anchors_path = 'Data/anchors'

    np.savetxt(f"{anchors_path}/{filename.strftime('%H:%M %d %B %Y') + '.txt'}", anchors, fmt='%f')


def read_anchors(selection=None):
    anchors_path = 'Data/anchors'
    last_anchors = os.listdir(anchors_path)[-1]
    print(last_anchors)
    if selection:
        anchors = np.loadtxt(f'{anchors_path}/{selection}', dtype='float32')
        return anchors
    else:
        anchors = np.loadtxt(f'{anchors_path}/{last_anchors}', dtype='float32')
        return anchors
