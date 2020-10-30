from data_pipeline import *
from darknet_v3 import darknet_body_test, last_layers
from Yolo_v3 import *

train = Dataset('Data.v1.tfrecord/train/train.tfrecord')
test = Dataset('C:/Users/levyg/Desktop/Yolo_project_new/Data.v1.tfrecord/test/test.tfrecord')

anchors = np.array([[0.079327, 0.050481],
                    [0.048077, 0.055288],
                    [0.076923, 0.086538],
                    [0.098558, 0.103365],
                    [0.062500, 0.067308]])

num_classes = 6
log_dir = 'Log'
batch = 4
tf.random.set_seed(1)
generator = test.generator_wrapper(anchors, batch, num_classes)
# image = tf.random.normal((4, 416, 416, 3))

y_true = next(generator)[1]
# y_true = [tf.random.normal((4, 13, 13, 5, 11)), tf.random.normal((4, 26, 26, 5, 11)),
#           tf.random.normal((4, 52, 52, 5, 11))]

# darknet = darknet_body_test(image)
# yolo_body = yolo_body_test(image, 5, num_classes)
# yolo_body = [tf.random.normal((4, 13, 13, 55)), tf.random.normal((4, 52, 52, 55)), tf.random.normal((4, 52, 52, 55))]
#
grid, feats, box_xy, box_wh = Yolo_head(yolo_body[0], anchors, num_classes, (416, 416), calc_loss=True)
pred_box = tf.concat([box_xy, box_wh], axis=-1)
grid_shape = get_shape(y_true, yolo_body, 3, True)
input_shape = get_shape(y_true, yolo_body, 3)
object_mask = tf.cast(y_true[0][..., 4:5], tf.int32)
raw_true_xy, raw_true_wh = ground_truth_to_layers_dims(0, y_true, grid_shape, grid, anchors, input_shape, object_mask)
ignore_mask = tf.TensorArray(y_true[0].dtype, size=1, dynamic_size=True)
get_best_iou_mask(0, 4, y_true, pred_box, object_mask, ignore_mask)
loss = get_giou_loss(raw_true_xy, raw_true_wh, box_xy, box_wh)
print(loss)


