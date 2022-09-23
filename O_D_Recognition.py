"""
O_D_Recognition.py
    Main file for detecting ocular diseases on images
Bryce Harrington
9/23/2022
"""
import numpy as np
from O_D_Model import ODModel
from O_D_DataLoader import od_data_loader


def ocular_disease_recognition_training():
    """
    Main method for training an od model
    """
    data, labels = od_data_loader(
        labels_path='/home/byrce/workspace/projects/OcularDiseaseRecognition/Data/archive/ODIR-5K/ODIR-5K/data.xlsx',
        data_path='/home/byrce/workspace/projects/OcularDiseaseRecognition/Data/archive/preprocessed_images')

    odm = ODModel()
    odm.train_model(train_data=[data, labels], val_data=[data, labels])


if __name__ == "__main__":
    ocular_disease_recognition_training()
