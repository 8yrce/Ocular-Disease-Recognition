"""
O-D-DataLoader.py
    Handles loading and otherwise representing our training / testing data for an O-D model
Bryce Harrington
9/23/2022
"""

from os.path import exists
import pandas as pd
import numpy as np


def od_data_loader(data_path: str):
    """
    Base method for loading in ocular disease data
    :param data_path: path to the data file
    :return: loaded_data: numpy data in an ML friendly format
    """
    """
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
    def load_od_data():
        """
        Load our data into a friendly format
        :returns: data: pandas dataframe of the loaded data
        """
        data = pd.read_excel(data_path) if exists(data_path) else None
        if data is None:
            raise FileNotFoundError
        return data

    def parse_od_data(data: pd.DataFrame):
        """
        Parse the loaded in data into an ML friendly format / drop data we don't need
        :param data: pandas data frame of OD data
        :returns: reshaped_data: numpy array of labeled data ready for training
        """
        # Columns we need: 0/A (ID), 7/H - 14/O ( Conditions present )
        data = data[['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']]

        # reshape so we have all data across row x for each column
        reshaped_data = []
        for row in data.iterrows():
            reshaped_data.append(row[1].values)
        return np.array(reshaped_data)

    data = parse_od_data(load_od_data())
    print(data[0])
    return data


if __name__ == "__main__":
    od_data_loader('/home/byrce/workspace/projects/OcularDiseaseRecognition/Data/archive/ODIR-5K/ODIR-5K/data.xlsx')
