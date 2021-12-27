# %%

import numpy as np
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from ipywidgets import widgets
import os
from pyquaternion import Quaternion

"""
Strecker 1 zeigerfinger: 5
Strecker 2 zeigerfinger: 4

Beuger 1 zeigerfinger: 1
Beuger 2 zeigerfinger: 3

Strecker 1 Daumen: 6
Strecker 2 Daumen: 7

Daumen Abspreitzer: 2
Daumen Beuger: 0


Id
ZF PP : 1007
ZF DP : 1008

DAUMEN DP : 1009
DAUMEN MC : 1010

empty: 1011
"""


@dataclass
class FingerAssignment:
    zf_strecker_1: int
    zf_strecker_2: int

    zf_beuger_1: int
    zf_beuger_2: int

    daumen_strecker_1: int
    daumen_strecker_2: int

    daumen_spreitzer: int
    daumen_beuger: int


@dataclass
class RigidBodyAssignment:
    zf_pp: int
    zf_dp: int
    daumen_dp: int
    daumen_mc: int
    force_torque: int


class TestEvaluator():

    def __init__(self, finger_a: FingerAssignment, body_a: RigidBodyAssignment, name='init_1vs1.json'):

        # read the json

        self.name = f'./data/test_november/{name}'
        with open(self.name, encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        # extract the main vectors
        self.obs = data['observation']
        self.act = data['action']
        self.time = [t / 1000 for t in data['time']]

        self.body_a = body_a
        self.finger_a = finger_a
        self.assign_rigid_bodies()

    def assign_rigid_bodies(self):
        """assign the rigid bodies acco to the names"""
        for attribute in dir(self.body_a):

            if ('_' in attribute[0]) == False:
                print(attribute)
                setattr(self, attribute,
                        self.obs['rigid_bodies'][getattr(self.body_a, attribute)])
                self.inverse_quaternions(attribute)

    def inverse_quaternions(self, call_name):
        """inverse the quaternions"""
        rigid_b = getattr(self, call_name)

        prev = np.array([1, 0, 0, 0])
        # watch signums
        for i, ele in enumerate(zip(rigid_b['qw'], rigid_b['qx'], rigid_b['qy'], rigid_b['qz'])):
            w, x, y, z = ele
            cur = np.array([w, x, y, z])

            if np.linalg.norm(prev-cur) < np.linalg.norm(prev + cur):

                rigid_b['qw'][i] = -w
                rigid_b['qx'][i] = -x
                rigid_b['qy'][i] = -y
                rigid_b['qz'][i] = -z

            prev = np.array([w, x, y, z])

        # catch inverse flips
    def inverse_quat2(self, call_name, eps=0.2):
        rigid_b = getattr(self, call_name)
        prev = np.array([1, 0, 0, 0])

        for i, ele in enumerate(zip(rigid_b['qw'], rigid_b['qx'], rigid_b['qy'], rigid_b['qz'])):
            w, x, y, z = ele
            cur = np.array([w, x, y, z])

            if abs(x) > eps and abs(abs(cur[1]) - abs(prev[1])) < eps and prev[1] * cur[1] < 0:

                rigid_b['qw'][i] = w
                rigid_b['qx'][i] = -x
                rigid_b['qy'][i] = -y
                rigid_b['qz'][i] = -z

            prev = np.array([w, x, y, z])

    def plot_rigid_bodies(self, call_name, pl_pos=False, lim=0):
        """make a simple plot, containing the general quaternion and position data"""
        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1) if pl_pos else None
        plt.title(call_name)
        rigid_b = getattr(self, call_name)

        if lim == 0:
            lim = len(self.time)

        plt.plot(self.time[:lim], rigid_b['qx'][:lim], label='qx')
        plt.plot(self.time[:lim], rigid_b['qy'][:lim], label='qy')
        plt.plot(self.time[:lim], rigid_b['qz'][:lim], label='qz')
        plt.plot(self.time[:lim], rigid_b['qw'][:lim], label='qw')
        plt.grid()
        plt.ylabel('quaternion')
        plt.legend()

        if pl_pos:
            plt.subplot(2, 1, 2)
            plt.plot(self.time[:lim], rigid_b['x'][:lim], label='x')
            plt.plot(self.time[:lim], rigid_b['y'][:lim], label='y')
            plt.plot(self.time[:lim], rigid_b['z'][:lim], label='z')
            plt.grid()
            plt.legend()
            plt.ylabel('position')

        plt.xlabel('time [s]')


finger_a = FingerAssignment(4, 6, 3, 2, 7, 5, 0, 1)
body_a = RigidBodyAssignment(0, 1, 2, 3, 4)
data = TestEvaluator(finger_a, body_a)


def read_all_files(idx, lim=5000):
    filename = testfiles[idx]
    print(filename)

    finger_a = FingerAssignment(4, 6, 3, 2, 7, 5, 0, 1)
    body_a = RigidBodyAssignment(0, 1, 2, 3, 4)
    data = TestEvaluator(finger_a, body_a, name=filename)

    # data.plot_rigid_bodies('force_torque')
    data.plot_rigid_bodies('zf_pp', lim=lim)
    data.plot_rigid_bodies('zf_dp', lim=lim)
    data.plot_rigid_bodies('daumen_dp', lim=lim)
    data.plot_rigid_bodies('daumen_mc', lim=lim)


def t_filt(arr, t=0.9):
    new_arr = arr.copy()
    for i in range(1, len(arr)):
        new_arr[i] = (1 - t) * arr[i] + t * new_arr[i - 1]
    return new_arr


# %%
if __name__ == '__main__':
    data = TestEvaluator(finger_a, body_a, name='pincer_ft_final.json')
    testfiles = os.listdir('./data/test_november')
    idx_test = widgets.IntSlider(min=0, max=len(testfiles), value=0)
    widgets.interact(read_all_files, idx=idx_test)

    data.obs['force_torques'][0].keys()
    # %%
    fx = [data.obs['force_torques'][i]['fx']
          for i in range(len(data.obs['force_torques']))][0]
    fy = [data.obs['force_torques'][i]['fy']
          for i in range(len(data.obs['force_torques']))][0]
    fz = [data.obs['force_torques'][i]['fz']
          for i in range(len(data.obs['force_torques']))][0]

    f_all = [np.sqrt(fxi**2 + fyi**2 + fzi**2)
             for fxi, fyi, fzi in zip(fx, fy, fz)]

    plt.plot(fx)
    plt.plot(fy)
    plt.plot(fz)

    plt.plot(fx[0])

    plt.figure()
    plt.plot(data.time[:4400], f_all[:4400])
    plt.grid()
    plt.xlabel('time [s]', fontsize=14)
    plt.ylabel('pincer force [N]', fontsize=14)


# %%