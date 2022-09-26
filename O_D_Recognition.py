"""
O_D_Recognition.py
    Main file for detecting ocular diseases on images
Bryce Harrington
9/23/2022
"""
from O_D_Model import ODModel
from O_D_DataLoader import od_data_loader

from argparse import ArgumentParser
parser = ArgumentParser("Ocular Disease Recognition")
parser.add_argument("--data_path", metavar='d', type=str, help="Path to the image data to use",
                    default = '/home/byrce/workspace/projects/OcularDiseaseRecognition/Data/archive/preprocessed_images')
parser.add_argument("--label_path", metavar='l', type=str, help="Path to the image data to use",
                    default='/home/byrce/workspace/projects/OcularDiseaseRecognition/Data/archive/ODIR-5K/ODIR-5K/data.xlsx')
args = parser.parse_args()


def ocular_disease_recognition_training(data_path: str, label_path: str):
    """
    Main method for training an od model
    :param data_path: path to the image data
    :param label_path: path to the labels for the image data
    """
    data, labels = od_data_loader(
        labels_path=label_path,
        data_path=data_path)

    odm = ODModel()
    odm.train_model(train_data=[data[0], labels[0]], val_data=[data[1], labels[1]])


if __name__ == "__main__":
    ocular_disease_recognition_training(data_path = args.data_path, label_path = args.label_path)
