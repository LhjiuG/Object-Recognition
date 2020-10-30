# NOTE :
# This python file will only have to contain function
# All variable should be assigned where they will
# be use.

import os
import tensorflow as tf
from tensorflow import image, io
import pandas as pd
import numpy as np


def load_labels(labels_path):

    with open(labels_path, 'r') as labels_file:
        labels = labels_file.read().splitlines()
        for i in range(len(labels)):
            labels[i] = labels[i].replace(" ", "")
    return labels


def load_data(directory_path, selection):
    """
    parameters
    -----------
    directory_path : TRAIN_DIR, TEST_DIR or VALID_DIR
    selection : ".txt" or ".jpg"

    output
    ------ 
    list of string of path to each indiviual jpg or txt file.
    """
    names_list = []
    for filename in os.listdir(directory_path):
        # Select only items finishing with the selection (.txt or .jpg)
        if filename.endswith(selection):
            names_list.append(f"{directory_path}/{filename}")
    names_list.sort()
    return names_list


def load_image(image_path):
    """
    Convert all image into array of shape (1, 416, 416, 3)
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_image(image, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize(image, (416, 416))
    image = image[tf.newaxis, :]
    return image


def get_boxes(annotations_path):
    """ Get boxes[labels, x, y, width, height]"""
    dataframe = pd.read_csv(annotations_path, delimiter=" ", header=None)
    dataframe = dataframe.rename(columns={0: "labels",
                                          1: "x",
                                          2: "y",
                                          3: "width",
                                          4: "height"})
    return dataframe

