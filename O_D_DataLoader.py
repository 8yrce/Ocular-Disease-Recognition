"""
O_D_DataLoader.py
    Handles loading and otherwise representing our training / testing data for an O-D model
Bryce Harrington
9/23/2022
"""
import random

from pandas import read_excel, DataFrame
import numpy as np
from imutils import resize
from os.path import exists, join
from os import listdir
import cv2


def od_data_loader(data_path: str, labels_path: str, split: list = [0.7, 0.2, 0.1]):
    """
    Base method for loading in ocular disease data
    :param data_path: path to the od images
    :param labels_path: path to the excel data file
    :param split: data splits for train, validation, and testing
    :return: loaded_data: numpy data in an ML friendly format
    """
    def load_od_labels():
        """
        Load our data into a friendly format
        :returns: data: pandas dataframe of the loaded data
        """
        labels = read_excel(labels_path) if exists(labels_path) else None
        if labels is None:
            raise FileNotFoundError("[INFO] Label file not found")
        return labels

    def parse_od_labels(labels: DataFrame, loaded_image_names):
        """
        Parse the loaded in data into an ML friendly format / drop data we don't need
        :param data: pandas data frame of OD data
        :param loaded_image_names: images found in the dataset ( to compare against images req'd for labels )
        :returns: reshaped_data: numpy array of labeled data ready for training
            Quick note:
                Normal (N),
                Diabetes (D),
                Glaucoma (G),
                Cataract (C),
                Age related Macular Degeneration (A),
                Hypertension (H),
                Pathological Myopia (M),
                Other diseases/abnormalities (O)
        """
        # Columns we need: 0/A (ID), 7/H - 14/O ( Conditions present )
        image_names = labels[['Left-Fundus', 'Right-Fundus']]
        labels = labels[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]
        print(f"[INFO] Found: {len(labels)} labels representing multiple objects")

        # reshape so we have all data across row x for each column
        reshaped_labels = []
        for i, row in enumerate(labels.iterrows()):
            # if we have an identified class / have data required for the label
            if 1 in row[1].values:
                if image_names.values[i][0] in loaded_image_names:
                    reshaped_labels.append(row[1].values)
                # if we have two images listed, make sure we have two labels
                if image_names.values[i][1] in loaded_image_names:
                    reshaped_labels.append(row[1].values)

        print(f"[INFO] Loaded {len(reshaped_labels)} labels")
        return np.array(reshaped_labels)

    def load_od_data():
        """
        Load the ocular disease images
        :return: data: ndarray of od images ( in order )
                  loaded_image_names: array of the image we found
        """
        if not exists(data_path):
            raise FileNotFoundError("[ERROR] Data filepath not found")
        print(f"[INFO] Found: {len(listdir(data_path))} images")

        data = []
        loaded_image_names = []
        for image_path in sorted(listdir(data_path)):
            image = cv2.imread(join(data_path, image_path))
            image = resize(image, width=(image.shape[1]//2), height=(image.shape[1]//2))
            data.append(np.array(image) / 255.0)
            loaded_image_names.append(image_path)

        print(f"[INFO] Loaded: {len(loaded_image_names)} images")
        return np.array(data), loaded_image_names

    def split_od_data(data: np.ndarray, labels: np.ndarray, randomize: bool = True):
        """
        Split the data
        :param data: loaded ocular disease image data
        :param labels: loaded ocular disease image labels
        :param randomize: shuffle datasets before splitting?
        :return: split_data, split_labels: the input data / labels split ( ordered by train, val, test )
        """
        split_data, split_labels = [], []
        if randomize:
            random.seed(1337)
            random.shuffle(data)
            random.shuffle(labels)

        splits = [int(s * len(data)) for s in split]
        #ps = 0
        for i, s in enumerate(splits):
            split_data.append(data[splits[i-1] if i > 1 else 0: s])
            split_labels.append(labels[splits[i-1] if i > 1 else 0: s])

        return split_data, split_labels

    def validate(split_data: list, split_labels: list):
        """
        Confirm the data is of uniform size, both total and per split
        :param split_data: loaded and split ocular disease image data
        :param split_labels: loaded and split ocular disease label data
        :raises: exception for data cardinality
        """
        # check total size
        if len(data) is not len(labels):
            raise Exception("[ERROR] data / label sizes are inconsistent")
        # check per split size
        for (sd, sl) in zip(split_data, split_labels):
            if sd.shape[0] != sl.shape[0]:
                raise Exception(f"[ERROR] data / label size in split are inconsistent")
        print("[INFO] Data passed validation")

    data, image_names = load_od_data()
    labels = parse_od_labels(load_od_labels(), image_names)
    data, labels = split_od_data(data, labels)
    validate(data, labels)

    return data, labels


if __name__ == "__main__":
    od_data_loader(
        labels_path='/home/byrce/workspace/projects/OcularDiseaseRecognition/Data/archive/ODIR-5K/ODIR-5K/data.xlsx',
        data_path='/home/byrce/workspace/projects/OcularDiseaseRecognition/Data/archive/preprocessed_images'
    )
