# NOTE :
# This python file will only have to contain function
# All variable should be assigned where they will
# be use.

import os
import tensorflow as tf
from tensorflow import image, io
import pandas as pd
import numpy as np


# !!!
# VARIABLE ONLY FOR TESTING PURPOSE
# !!!

labels_length = 5


def load_labels(labels_path):
    """
    :param: both darknet.labels from train or test folder are the same,
    so either path is fine.
    :output: list of string of labels
    """
    with open(labels_path, 'r') as labels_file:
        # used splitlines instead of readlines in order to not have the /n.
        labels = labels_file.read().splitlines()
        for i in range(len(labels)):
            # Remove whitespace between the word
            labels[i] = labels[i].replace(" ", "")
    return labels


def load_data(directory_path, selection):
    """
    execute select_txt_jpg() and in its turn execute concat_path_to_names()
    this give us a list containing each path to each file separatly.

    :param: directory_path : TRAIN_DIR, TEST_DIR or VALID_DIR
            selection: ".txt" or ".jpg"

    :output: list of string of path to each indiviual jpg or txt file.
    """

    def select_txt_jpg():
        """
        Create a list containing all the filename from the directory
        and extract the jpg or txt from it.

        output : List of string of name from each indivual jpg or txt file.
        """
        names_list = []
        # Iterate over a list of item coresponding to each file in the directory.
        for filename in os.listdir(directory_path):
            # Select only items finishing with the selection (.txt or .jpg)
            if filename.endswith(selection):
                names_list.append(filename)
        # Each items is a couple made of 1 txt and 1 jpg, which both hold the same
        # name. We sort them as prevention of futur error which could lead to the
        # data losing their initial index rank.
        names_list.sort()

        def concat_path_to_names():
            """
            Add the the directory path to each file name in the list.
            """
            path_list = []
            for name in names_list:
                # Add directory_path to each string in the list.
                path_list.append(f"{directory_path}/{name}")
            return path_list

        return concat_path_to_names()

    return select_txt_jpg()


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


def one_hot_encoding(dataframe):
    """
    :input: Pandas dataframe of boxes from 1 image in from int and float.
    :output: Dataframe with labels one_hot_encoded
    {0: x, 1: y, 2: width, 3: height, 4: labels0, ..., nlabels}
    """
    # Create a dataframe of int coresponding of 0 to len(labels).
    labels_dataframe = pd.DataFrame((range(labels_length)), dtype='object')
    labels_dataframe = labels_dataframe.rename(columns={0: 'labels'})
    # Mixing labels_dataframe to dataframe allows dataframe to get at least
    # one example of each classes during the one hot encoding.
    dataframe = dataframe.append(labels_dataframe)
    # Create a one hot encoding of the columns labels
    one_hot = pd.get_dummies(dataframe.labels, prefix='labels')
    # Stack one_hot and dataframe togheter horizontally
    dataframe = pd.concat((dataframe, one_hot), axis=1)
    # Drop row coresponding to labels_dataframe and the columns labels.
    dataframe = dataframe.dropna().drop('labels', axis=1)
    return dataframe


def get_annotations(annotations_path):
    dataframe = pd.read_csv(annotations_path, delimiter=" ", header=None)
    dataframe = dataframe.rename(columns={0: "labels",
                                          1: "x",
                                          2: "y",
                                          3: "width",
                                          4: "height"})
    dataframe = one_hot_encoding(dataframe)
    return dataframe

