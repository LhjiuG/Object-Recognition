import os
import tensorflow as tf
from tensorflow import image, io
import numpy as np


def load_labels(labels_path):

    with open(labels_path, 'r') as labels_file:
        labels = labels_file.read().splitlines()
        for i in range(len(labels)):
            labels[i] = labels[i].replace(" ", "")
    return labels


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

def load_image(image_path):
    """
    Convert all image into array of shape (1, 416, 416, 3)
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, (608, 608))
    image = image[tf.newaxis, :]
    return image


def get_boxes(annotations_path, training=True):
    """ Get boxes[labels, x, y, width, height]"""
    boxes = np.loadtxt(annotations_path, delimiter=" ", dtype='float32')
    # Add a dimension if the image only has 1 boxes and therefore is (5, )
    if boxes.ndim == 1:
        boxes = np.expand_dims(boxes, 0)
    # add a batch dimension and multiple boxes filled with zero to get the final shape of (1, 20, 5)
    elif training:
        zero_array = np.zeros((20, 5))
        boxes = np.expand_dims(np.concatenate((boxes, zero_array))[:20], 0)
        return boxes


    return boxes


def get_wh(boxes):
    wh = np.zeros((1,2))
    for i in range(len(boxes)):
        bbox = get_boxes(boxes[i], False)[..., 3:5]
        if bbox.shape[1] != 0:
            wh = np.append(wh , bbox, axis=0)



def bbox_to_corner(boxes):
    dataset = list()
    for box in boxes:
        ymin = box[1] / 2.
        xmin = box[0] / 2.
        ymax = -ymin
        xmax = -xmin

        dataset.append([ymin, xmin, ymax, xmax])
    return np.array(dataset)


def revert(out):
    dataset = list()
    for box in out:
        width = box[1] * 2
        height = box[0] * 2

        dataset.append([width, height])

    return np.array(dataset)