# %% File for handling the conversion between optitrack stream and Tracker positions
import math
import csv
import json
import pandas as pd
import numpy as np
import torch
from pyquaternion.quaternion import Quaternion


def return_sorted_points(points1, points2):
    """summarize the functions"""
    pairs1 = sort_points_by_dis(points1)
    pairs2 = sort_points_by_dis(points2)
    return compare_point_lists(pairs1, points1, pairs2, points2)


def sort_points_by_dis(points):
    n = len(points)
    pairs = []
    for i in range(n):
        for j in range(n - 1, -1 + i, -1):
            if j != i:
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                dz = points[j][2] - points[i][2]
                d_diff = math.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
                # erstelle Liste die alle Punkte distanzen, plus euklidische Distanz enthält
                pairs.append([points[j], points[i], d_diff])
    # Sortiere Punktepaare nach euklidischer Distanz. Aufsteigend von kurz nach lang.
    pairs.sort(key=lambda x: x[2])

    return pairs


def compare_point_lists(pairs1, points1, pairs2, points2):
    '''Lese Punktepaare, und Punktewolken ein. Vergleiche Positionen der Punkte anhand 
    der Stellen in den Punktepaaren, in denen sie
    vorkommen.'''
    distance_value_in_points1 = [[], [], [], [], []]
    distance_value_in_points2 = [[], [], [], [], []]

    '''Erstelle für jeden Punkt eine Liste, in der Steht in welcher Distanz er vokommt.'''
    for i in range(len(pairs1)):
        for j in range(len(points1)):

            if points1[j] in pairs1[i]:
                pairs1[i].index(points1[j])
                distance_value_in_points1[j].append(i + 1)

            if points2[j] in pairs2[i]:
                pairs2[i].index(points2[j])
                distance_value_in_points2[j].append(i + 1)

    '''An dieser Stelle soll die Liste der Punkte aus dem CT (2) anhand der Punkte aus dem Opti-Export (1)
    sortiert werden. Dafür wird der Index einer Distanz Index Kombi von (2) in (1) gesucht und der Index gepseichert.
    Anhand der entstehenden Liste von Indexen werden die Punkte von (2) umsortiert.'''''
    index_list = []
    for i in range(len(points1)):
        index_list.append(distance_value_in_points1.index(
            distance_value_in_points2[i]))

    # https://stackoverflow.com/questions/6618515/sorting-list-based-on-values-from-another-list
    points_2_out = [x for _, x in sorted(zip(index_list, points2))]

    return points1, points_2_out


class TMatrix(object):
    """The Ligntning Module containing the logic of 4x4 Transformation matrices"""

    def __init__(self, quat: Quaternion, pos: np.ndarray):
        """define all initial parameters"""
        super(TMatrix, self).__init__()
        # define matrix and rotation
        self.quat = torch.tensor(quat.q).float()
        self.rotmat = torch.tensor(quat.rotation_matrix).float()
        self.pos = torch.tensor(pos).float()

        self.forw = self.get_matrix()
        self.backw = self.get_matrix(inverse=True)

    def norm(self):
        """ensure quat is always normalised first"""
        sum_sq = torch.dot(self.quat, self.quat)
        norm = torch.sqrt(sum_sq)
        q_norm = self.quat / norm
        return q_norm

    def quat_to_rot(self):
        """add quat to rot to computation graph"""
        quat_norm = self.norm()
        qw, qx, qy, qz = quat_norm

        matrix = torch.zeros(3, 3)

        matrix[0, 0] = 1. - 2. * qy ** 2 - 2. * qz ** 2
        matrix[1, 1] = 1. - 2. * qx ** 2 - 2. * qz ** 2
        matrix[2, 2] = 1. - 2. * qx ** 2 - 2. * qy ** 2

        matrix[0, 1] = 2. * qx * qy - 2. * qz * qw
        matrix[1, 0] = 2. * qx * qy + 2. * qz * qw

        matrix[0, 2] = 2. * qx * qz + 2 * qy * qw
        matrix[2, 0] = 2. * qx * qz - 2 * qy * qw

        matrix[1, 2] = 2. * qy * qz - 2. * qx * qw
        matrix[2, 1] = 2. * qy * qz + 2. * qx * qw

        return matrix

    def get_pos(self):
        """assign pos"""
        cur_pos = torch.zeros(3)
        cur_pos[0] = self.pos[0]
        cur_pos[1] = self.pos[1]
        cur_pos[2] = self.pos[2]
        return cur_pos

    def get_matrix(self, inverse=False):
        """ge the complete transformation matrix"""
        matrix = torch.zeros(4, 4)
        rot_mat = self.quat_to_rot().T if inverse else self.quat_to_rot()
        matrix[:3, :3] = rot_mat
        matrix[3, 3] = 1
        matrix[:3, 3] = - \
            torch.matmul(rot_mat, self.get_pos()
                         ) if inverse else self.get_pos()
        return matrix

    def forward(self, x, inverse=False):
        """transform vector with tmat, depending on inverse or not"""
        # easier to calculate
        matrix = self.backw if inverse else self.forw
        return torch.matmul(matrix, x)

    def info(self):
        """print the info of 4x4 transformations, pos and quat"""
        print(self.quat.detach().numpy())
        print(self.pos.detach().numpy())
        print(self.quat_to_rot().detach().numpy())
        print(self.get_matrix().detach().numpy())


# %%


class Tracker(object):
    """class to handle the"""

    def __init__(self, id_num: int, defname: str, ctname=None):
        """init the tracker
        1. read files from pointlist
        """
        self.id_num = id_num
        self.defname = defname
        self.ctname = ctname
        # read definition file into memory
        self.read_markerdata()

        if ctname is not None:
            # read jsonfile from ct into memory
            self.read_ctdata()
            self.calculate_transformation_matrix()
        else:
            # define a test scenario
            self.perform_test()

    def perform_test(self):
        """function to perform the test
        # take the 
        """
        q_init_f = Quaternion(np.random.randn(4))
        p_init_f = np.array(np.random.randn(3))
        self.transform_fake = TMatrix(q_init_f, p_init_f)
        print(self.transform_fake.get_matrix())

        # now transform the markers by the fake matrix:
        self.marker_pos_ct = []
        for point in self.marker_pos_def:
            torch_vec = torch.tensor([point[0], point[1], point[2], 1])
            trans_vec = self.transform_fake.forward(torch_vec)
            trans_vec = trans_vec.cpu().detach().numpy().tolist()
            self.marker_pos_ct.append(
                [trans_vec[0], trans_vec[1], trans_vec[2]])
        points = np.random.permutation(np.array(self.marker_pos_ct)).tolist()
        self.marker_pos_ct = points
        _, self.marker_pos_ct = return_sorted_points(
            self.marker_pos_def, self.marker_pos_ct)
        t_ct_def = self.calculate_transformation_matrix()
        print(t_ct_def)
        t_def_ct = np.linalg.inv(np.array(t_ct_def))
        print(t_def_ct)

    def read_ctdata(self):
        """read the data of the marker points from the ct scan"""
        # 1. load mounting points
        with open(f'{self.ctname}.mrk.json') as jsonfile:
            data = json.load(jsonfile)
        # extract point infos
        point_data = data['markups'][0]['controlPoints']
        self.marker_pos_ct = [point['position'] for point in point_data]

    def read_markerdata(self):
        """read the data from the csv file"""
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
                    [float(row[2]), float(row[3]), float(row[4])])
        self.marker_pos_def = coordinates

    def calculate_transformation_matrix(self):
        """the required tranformation matrix between system 1 and 2"""
        markers1 = self.marker_pos_def
        markers2 = self.marker_pos_ct

        # Convert lists of markers to arrays
        markers1 = np.array(markers1)
        markers2 = np.array(markers2)

        # Center the markers at the origin
        markers1_mean = np.mean(markers1, axis=0)
        markers2_mean = np.mean(markers2, axis=0)
        markers1 -= markers1_mean
        markers2 -= markers2_mean

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
        t = markers1_mean - np.dot(markers2_mean, R)

        # Concatenate the rotation and translation matrices
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t
        """
        This matrix points FROM def TO ct
        """
        return transformation_matrix


# %%
tr = Tracker(0, '51k.csv')

# %%
