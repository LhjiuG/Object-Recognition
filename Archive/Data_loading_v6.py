import os
from tensorflow import io, newaxis, float32
from tensorflow import image as img
import numpy as np
from Anchor_generator import kmeans
import datetime

max_param = 20
width, height = 608, 608


def get_path(directory_path, selection):
    """
    parameters
    -----------
    directory_path : TRAIN_DIR, TEST_DIR or VALID_DIR
    selection : ".txt" or ".jpg"

    output
    ------ 
    list of string of path to each indiviual jpg or txt file.
    """
    path_list = list()
    for filename in os.listdir(directory_path):
        # Select only items finishing with the selection (.txt or .jpg)
        if filename.endswith(selection):
            path_list.append(f"{directory_path}/{filename}")
    path_list.sort()
    return path_list


def load_labels(labels_path):
    with open(labels_path, 'r') as labels_file:
        labels = labels_file.read().splitlines()
        for i in range(len(labels)):
            labels[i] = labels[i].replace(" ", "")
    return labels


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


def get_boxes(annotations_path, training=True):
    """ Get boxes[labels, x, y, width, height]"""
    boxes = np.loadtxt(annotations_path, delimiter=" ", dtype='float32')
    # Add a dimension if the image only has 1 boxes and therefore is (5, )
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, 0)
    # add a batch dimension and multiple boxes filled with zero to get the final shape of (1, max_param, 5)
    elif training:
        zero_array = np.zeros((max_param, 5))
        boxes = np.expand_dims(np.concatenate((boxes, zero_array))[:max_param], 0)
        return np.roll(boxes, -1, axis=2)
    return np.roll(boxes, -1, axis=1)


def anchors_generator(boxes_path, cluster):
    box_wh = np.zeros((1, 2))
    for i in range(len(boxes_path)):
        bbox = get_boxes(boxes_path[i], False)[..., 2:4]
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


