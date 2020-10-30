import tensorflow as tf
import tensorflow_addons as tfa


@tf.function
def Yolo_head(feats, anchors, num_classes, input_shape, calc_loss=False):
    """Create the spatial grid and adjust the prediction to their respective cell"""
    num_anchors = len(anchors)
    # Reshape to batch, height, width, num_anchors, box_params.
    anchors_tensor = tf.reshape(tf.constant(anchors), [1, 1, 1, num_anchors, 2])

    conv_dims = tf.shape(feats)[1:3]
    conv_height_index = tf.range(conv_dims[0])
    conv_width_index = tf.range(conv_dims[1])

    x_axis, y_axis = tf.meshgrid(conv_width_index, conv_height_index)
    grid = tf.cast(tf.concat([x_axis, y_axis]), dtype=tf.dtypes.DType(feats))

    feats = tf.reshape(feats, [-1, conv_dims[0], conv_dims[1], num_anchors, num_classes + 5])

    # Adjust preditions to each spatial grid point and anchor size.
    box_xy = (tf.math.sigmoid(feats[..., :2]) + grid) / tf.cast(conv_dims[::-1], tf.dtypes.DType(feats))
    box_wh = tf.math.exp(feats[..., 2:4]) * anchors_tensor / tf.cast(input_shape[::-1], tf.dtypes.DType(feats))
    box_confidence = tf.math.sigmoid(feats[..., 4:5])
    box_class_probs = tf.math.sigmoid(feats[..., 5:])
    if calc_loss:
        return grid, feats, box_xy, box_wh
    return box_xy, box_wh, box_confidence, box_class_probs


@tf.function
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
        self.num_layers = 3  # Default

    def call(self, y_true, y_pred):
        input_shape = tf.cast(tf.shape(y_pred[0])[1:3] * 32, tf.dtypes.DType(y_true[0]))
        grid_shape = [tf.cast(tf.shape(y_pred[l])[1:3], tf.dtypes.DType(y_true[0])) for l in range(self.num_layers)]
        loss = 0
        m = tf.shape(y_pred[0])[0]  # Batch size
        mf = tf.cast(m, tf.dtypes.DType(y_pred[0]))

        for l in range(self.num_layers):
            object_mask = y_true[l][..., 4:5]
            true_class_probs = y_true[l][..., 5:]

            grid, raw_pred, pred_xy, pred_wh = Yolo_head(y_pred[l],
                                                         self.anchors,
                                                         self.num_classes,
                                                         input_shape,
                                                         self.print_loss)
            pred_box = tf.concat([pred_xy, pred_wh])

            # Darknet raw box to calculate loss.
            raw_true_xy = y_true[l][..., :2] * grid_shape[l][::-1] - grid
            raw_true_wh = tf.math.log(y_true[l][..., 2:4]) / self.anchors * input_shape[::-1]
            raw_true_wh = tf.switch_case(object_mask, raw_true_wh, tf.zeros_like(raw_true_wh))
            raw_true = tf.concat([raw_true_xy, raw_true_wh], axis=-1)

            # Find ignore mask, iterate over each of batch.
            ignore_mask = tf.TensorArray(tf.dtypes.DType(y_true[0]), size=1, dynamic_size=True)
            object_mask_bool = tf.cast(object_mask, 'bool')

            def loop_body(b, ignore_mask):
                true_box = tf.boolean_mask(y_true[l][b, ..., 0:4], object_mask_bool[b, ..., 0])
                iou = box_iou(pred_box[b], true_box)
                best_iou = tf.reduce_max(iou, axis=-1)
                ignore_mask = ignore_mask.write(b, tf.cast(best_iou < self.ignore_tresh, tf.dtypes.DType(true_box)))
                return b + 1, ignore_mask

            _, ignore_mask = tf.while_loop(lambda b, *args: b < m, loop_body, [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, -1)

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

            box_loss = tf.reduce_sum(giou_loss) / mf * 5
            confidence_loss = tf.reduce_sum(confidence_loss) / mf
            class_loss = tf.reduce_sum(class_loss) / mf
            loss += box_loss + class_loss + confidence_loss
            if self.print_loss:
                loss = tf.print(loss, [loss, giou_loss, confidence_loss, class_loss, tf.reduce_sum(ignore_mask)],
                                message='loss: ')
            return loss

        return loss
