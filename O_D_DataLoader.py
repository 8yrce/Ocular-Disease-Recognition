"""
O_D_DataLoader.py
    Handles loading and otherwise representing our training / testing data for an O-D model
Bryce Harrington
9/23/2022
"""
from pandas import read_excel, DataFrame
import numpy as np
from os.path import exists, join
from os import listdir
from keras_preprocessing.image import load_img


def od_data_loader(data_path: str, labels_path: str):
    """
    Base method for loading in ocular disease data
    :param data_path: path to the od images
    :param labels_path: path to the excel data file
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

    def parse_od_labels(labels: DataFrame):
        """
        Parse the loaded in data into an ML friendly format / drop data we don't need
        :param data: pandas data frame of OD data
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
        labels = labels[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]

        # reshape so we have all data across row x for each column
        reshaped_labels = []
        for row in labels.iterrows():
            reshaped_labels.append(row[1].values)
        return np.array(reshaped_labels)

    def load_od_data():
        """
        Load the ocular disease images
        :return: data: ndarray of od images ( in order )
        """
        if not exists(data_path):
            raise FileNotFoundError("[ERROR] Data filepath not found")

        data = []
        for image_path in listdir(data_path):
            data.append(load_img(join(data_path, image_path)))
        return data

    labels = parse_od_labels(load_od_labels())
    print(labels[0])

    data = load_od_data()

    return data, labels


if __name__ == "__main__":
    od_data_loader(
        labels_path='/home/byrce/workspace/projects/OcularDiseaseRecognition/Data/archive/ODIR-5K/ODIR-5K/data.xlsx',
        data_path='/home/byrce/workspace/projects/OcularDiseaseRecognition/Data/archive/preprocessed_images'
    )
