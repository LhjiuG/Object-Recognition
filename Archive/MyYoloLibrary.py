import keras.backend as K

import numpy as np


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
    y_true = [np.zeros((m, grid_shapes[l][0], grid_shapes[l][1], 2, 5 + num_classes),
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
