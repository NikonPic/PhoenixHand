# %% File for handling the conversion between optitrack stream and Tracker positions
import json
import pandas as pd
import numpy as np


class Tracker(object):
    """class to handle the"""

    def __init__(self, id_num: int, defname: str, ctname: str):
        """init the tracker
        1. read files from pointlist
        """
        self.id_num = id_num
        self.defname = defname
        self.ctname = ctname
        # read jsonfile into memory
        self.read_jsonfile()
        # read definition file into memory
        self.read_csv()

    def read_jsonfile(self):
        """read the data of the marker points from the ct scan"""
        # 1. load mounting points
        with open(f'{self.ctname}.mrk.json') as jsonfile:
            data = json.load(jsonfile)
        # extract point infos
        point_data = data['markups'][0]['controlPoints']
        self.points = [point['position'] for point in point_data]

    def read_csv(self):
        coordinates = []
        with open(f'trackers/{self.defname}', 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the first row (header)
            next(reader)  # Skip the second row (version)
            next(reader)  # Skip the third row (name)
            next(reader)  # Skip the fourth row (ID)
            next(reader)  # Skip the fifth row (color)
            next(reader)  # Skip the sixth row (units)
            for row in reader:
                coordinates.append(
                    (float(row[2]), float(row[3]), float(row[4])))
        return coordinates


def calculate_transformation_matrix(markers1, markers2):
    # Convert lists of markers to arrays
    markers1 = np.array(markers1)
    markers2 = np.array(markers2)

    # Center the markers at the origin
    markers1 -= np.mean(markers1, axis=0)
    markers2 -= np.mean(markers2, axis=0)

    # Calculate the cross-covariance matrix
    cross_cov = np.dot(markers1.T, markers2)

    # Calculate the singular value decomposition
    U, S, V_T = np.linalg.svd(cross_cov)

    # Calculate the rotation matrix
    R = np.dot(U, V_T)

    # Check for reflection
    if np.linalg.det(R) < 0:
        V_T[2, :] *= -1
        R = np.dot(U, V_T)

    # Calculate the translation vector
    t = np.mean(markers2, axis=0) - np.dot(np.mean(markers1, axis=0), R)

    # Concatenate the rotation and translation matrices
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t

    return transformation_matrix


# %%0


def calculate_transformation_matrix(markers1, markers2):
    # Convert lists of markers to arrays
    markers1 = np.array(markers1, dtype="float64")
    markers2 = np.array(markers2, dtype="float64")

    # Center the markers at the origin
    markers1 -= np.mean(markers1, axis=0)
    markers2 -= np.mean(markers2, axis=0)

    # Calculate the cross-covariance matrix
    cross_cov = np.dot(markers1.T, markers2)

    # Calculate the singular value decomposition
    U, S, V_T = np.linalg.svd(cross_cov)

    # Calculate the rotation matrix
    R = np.dot(U, V_T)

    # Check for reflection
    if np.linalg.det(R) < 0:
        V_T[2, :] *= -1
        R = np.dot(U, V_T)

    # Calculate the translation vector
    t = np.mean(markers2, axis=0) - np.dot(np.mean(markers1, axis=0), R)

    # Concatenate the rotation and translation matrices
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = t

    return transformation_matrix


# %%
markers1 = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
markers2 = [(11, 12, 13), (14, 15, 16), (17, 18, 19)]
trans_matrix = calculate_transformation_matrix(markers1, markers2)
print(trans_matrix)

# %%
