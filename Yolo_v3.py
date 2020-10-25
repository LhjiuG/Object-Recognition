import tensorflow as tf
from tensorflow.keras.layers import UpSampling2D, Concatenate
import tensorflow_addons as tfa

from darknet_v3 import darknet_body, last_layers, FullConv, darknet_body_test
from utils import compose
from data_pipeline import *


def yolo_body(inputs_tensor, num_anchors, num_classes):
    """Create Yolo body which output the prediction of the 3 last stage of darknet"""
    darknet = tf.keras.Model(inputs_tensor, darknet_body(inputs_tensor))
    x, y1 = last_layers(darknet.output, 512, num_anchors * (num_classes + 5))

    x = compose(FullConv(256, (1, 1), padding='same'), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[-2].output])
    x = FullConv(512, (3, 3), padding='same')(x)
    x, y2 = last_layers(x, 256, num_anchors * (5 + num_classes))

    x = compose(FullConv(128, (1, 1), padding='same'), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet.layers[-3].output])
    x = FullConv(256, (3, 3), padding='same')(x)
    x, y3 = last_layers(x, 128, num_anchors * (num_classes + 5))
    return [y1, y2, y3]


def yolo_body_test(inputs_tensor, num_anchors, num_classes):
    """Create Yolo body which output the prediction of the 3 last stage of darknet
    ONLY USED FOR TESTING PURPOSE"""

    darknet = darknet_body_test(inputs_tensor)
    x, y1 = last_layers(darknet[0], 512, num_anchors * (num_classes + 5))

    x = compose(FullConv(256, (1, 1), padding='same'), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet[1]])
    x = FullConv(512, (3, 3), padding='same')(x)
    x, y2 = last_layers(x, 256, num_anchors * (5 + num_classes))

    x = compose(FullConv(128, (1, 1), padding='same'), UpSampling2D(2))(x)
    x = Concatenate()([x, darknet[-1]])
    x = FullConv(256, (3, 3), padding='same')(x)
    x, y3 = last_layers(x, 128, num_anchors * (num_classes + 5))
    return [y1, y2, y3]


def Yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Create the spatial grid and adjust the prediction to their respective cell"""
    num_anchors = len(anchors)
    # Add necessary dimension to anchors for broadcasting
    anchors_tensor = tf.reshape(tf.constant(anchors, dtype=feats.dtype), [1, 1, 1, num_anchors, 2])

    conv_dims = tf.shape(feats)[1:3]  # height, width
    # Create the grid
    height_index, width_index = tf.range(conv_dims[0]), tf.range(conv_dims[1])
    grid = tf.cast(tf.reshape(tf.meshgrid(width_index, height_index), [conv_dims[1], conv_dims[0], 1, 2]),
                   dtype=feats.dtype)

    # Reshape for broadcasting by separating the last dimension in a dimension for each anchors
    feats = tf.reshape(feats, [-1, conv_dims[1], conv_dims[0], num_anchors, (num_classes + 5)])

    # Adjust predictions to each spatial grid point and anchor size.
    box_xy = (tf.math.sigmoid(feats[..., :2]) + grid) / tf.cast(conv_dims[::-1], dtype=feats.dtype)
    box_wh = tf.math.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[:: -1], dtype=feats.dtype)
    box_confidence = tf.math.sigmoid(feats[..., 4:5])
    box_class_probs = tf.math.sigmoid(feats[..., 5:])
    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


def box_iou(b1, b2):
    """Return iou tensor
    Parameters
    ----------
    b1: tensor, shape=(i1,...,iN, 4), xywh
    b2: tensor, shape=(j, 4), xywh

    Returns
    -------
    iou: tensor, shape=(i1,...,iN, j)
    """
    # Expand dim to apply broadcasting.
    b1 = tf.expand_dims(b1, -2)
    b1_xy = b1[..., :2]
    b1_wh = b1[..., 2:4]
    b1_wh_half = b1_wh / 2.
    b1_mins = b1_xy - b1_wh_half
    b1_maxes = b1_xy + b1_wh_half

    # Expand dim to apply broadcasting.
    b2 = tf.expand_dims(b2, 0)
    b2_xy = b2[..., :2]
    b2_wh = b2[..., 2:4]
    b2_wh_half = b2_wh / 2.
    b2_mins = b2_xy - b2_wh_half
    b2_maxes = b2_xy + b2_wh_half

    intersect_mins = tf.maximum(b1_mins, b2_mins)
    intersect_maxes = tf.minimum(b1_maxes, b2_maxes)
    intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
    b1_area = b1_wh[..., 0] * b1_wh[..., 1]
    b2_area = b2_wh[..., 0] * b2_wh[..., 1]
    iou = intersect_area / (b1_area + b2_area - intersect_area)

    return iou


class Yolo_loss(tf.keras.losses.Loss):
    def __init__(self, anchors, num_classes, ignore_tresh=.5, print_loss=False):
        super(Yolo_loss, self).__init__()
        self.anchors = anchors
        self.num_classes = num_classes
        self.ignore_tresh = ignore_tresh
        self.print_loss = print_loss
        self.num_layers = 3

    def call(self, y_true, y_pred):
        input_shape = tf.cast(tf.shape(y_pred[0])[1:3] * 32, y_true[0].dtype)
        grid_shape = [tf.cast(tf.shape(y_pred[l])[1:3], y_true[0].dtype) for l in range(self.num_layers)]

        loss = 0
        m = y_pred[0].shape[0]  # Batch size
        mf = tf.cast(m, y_pred[0].dtype)  # Float batch size

        for l in range(self.num_layers):
            # Tensor of 1 or 0 if an object is in the box
            object_mask = y_true[l][..., 4:5]  # [m, grid_shape[0], grid_shape[1], anchors, 1]
            # Tensor of 1 or 0 for the classes in the box one_hot_encoded. (if no classes then all 0)
            true_class_probs = y_true[l][..., 5:]  # [m, grid_shape[0], grid_shape[1], anchors, num_classes]

            grid, raw_pred, pred_xy, pred_wh = Yolo_head(y_pred[l],
                                                         self.anchors,
                                                         self.num_classes,
                                                         input_shape,
                                                         calc_loss=True)
            pred_box = tf.concat([pred_xy, pred_wh], axis=-1)

            # Darknet raw box to calculate loss.
            raw_true_xy, raw_true_wh = ground_truth_to_layers_dims(y_true, grid_shape, grid, self.anchors, input_shape,
                                                                   object_mask)

            # Find ignore mask, iterate over each of batch.
            ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
            object_mask_bool = tf.cast(object_mask, 'bool')

            b = 0
            while b < m:
                true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
                iou = box_iou(pred_box[b], true_box)
                best_iou = tf.reduce_max(iou, axis=-1)
                ignore_mask.write(b, tf.cast(best_iou < self.ignore_tresh, true_box.dtype))
                b += 1

            ignore_mask = tf.expand_dims(ignore_mask.stack(), -1)

            true_mins = raw_true_xy - raw_true_wh / 2.
            true_maxes = raw_true_xy + raw_true_wh / 2.
            pred_mins = pred_xy - pred_wh / 2.
            pred_maxes = pred_xy + pred_wh / 2.
            pred_mm = tf.concat([pred_mins, pred_maxes], axis=-1)
            true_mm = tf.concat([true_mins, true_maxes], axis=-1)

            giou_loss = tfa.losses.giou_loss(true_mm, pred_mm, mode='giou')
            confidence_loss = object_mask * tf.keras.losses.binary_crossentropy(object_mask, raw_pred[..., 4:5],
                                                                                from_logits=True) + (
                                      1 - object_mask) * tf.keras.losses.binary_crossentropy(object_mask,
                                                                                             raw_pred[..., 4:5],
                                                                                             from_logits=True) * ignore_mask
            class_loss = object_mask * tf.keras.losses.binary_crossentropy(true_class_probs, raw_pred[..., 5:],
                                                                           from_logits=True)

            box_loss = tf.reduce_sum(giou_loss) / mf
            confidence_loss = tf.reduce_sum(confidence_loss) / mf
            class_loss = tf.reduce_sum(class_loss) / mf
            loss += box_loss + class_loss + confidence_loss
            if self.print_loss:
                loss = tf.print(loss, [loss, giou_loss, confidence_loss, class_loss, tf.reduce_sum(ignore_mask)],
                                message='loss: ')
            return loss

        return loss


def get_shape(y_true, y_pred, num_layers, grid=False):
    shape = [tf.cast(tf.shape(y_pred[l])[1:3], y_true[0].dtype) for l in range(num_layers)]
    if grid:
        return shape
    else:
        return shape[0] * 32


def ground_truth_to_layers_dims(layer, y_true, grid_shape, grid, anchors, input_shape, object_mask):
    """Calculate the raw box of darknet with respect to the current layers outputs dims"""
    true_xy = y_true[layer][..., :2] * grid_shape[layer][::-1] - grid
    true_wh = tf.math.log(y_true[layer][..., 2:4]) / anchors * input_shape[::-1]
    # TODO: use tf function instead of a keras backend one
    true_wh = tf.keras.backend.switch(object_mask, true_wh, tf.zeros_like(true_wh))
    return true_xy, true_wh


def get_giou_loss(raw_true_xy, raw_true_wh, pred_xy, pred_wh):
    """Get the Giou loss"""
    true_mins = raw_true_xy - raw_true_wh / 2.
    true_maxes = raw_true_xy + raw_true_wh / 2.
    pred_mins = pred_xy - pred_wh / 2.
    pred_maxes = pred_xy + pred_wh / 2.
    pred_mm = tf.concat([pred_mins, pred_maxes], axis=-1)
    true_mm = tf.concat([true_mins, true_maxes], axis=-1)
    giou_loss = tfa.losses.giou_loss(true_mm, pred_mm, mode='giou')
    return giou_loss


@tf.function
def get_best_iou_mask(layer, m, y_true, pred_box, object_mask, ignore_mask, ignore_tresh=.5):
    b = 0
    object_mask_bool = tf.cast(object_mask, 'bool')
    while b < m:
        true_box = tf.boolean_mask(y_true[layer][b, ..., 0:4], object_mask_bool[b, ..., 0])
        iou = box_iou(pred_box[b], true_box)
        best_iou = tf.reduce_max(iou, axis=-1)
        ignore_mask.write(b, tf.cast(best_iou < ignore_tresh, true_box.dtype))
        b += 1
    return tf.expand_dims(ignore_mask.stack(), -1)
