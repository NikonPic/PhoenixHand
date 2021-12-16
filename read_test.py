# %%

import numpy as np
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from ipywidgets import widgets
import os

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

    def __init__(self, finger_a: FingerAssignment, body_a: RigidBodyAssignment, name='closeer.json'):

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

    def plot_rigid_bodies(self, call_name):
        """make a simple plot, containing the general quaternion and position data"""
        plt.figure(figsize=(12, 12))

        plt.subplot(2, 1, 1)
        plt.title(call_name)
        rigid_b = getattr(self, call_name)

        plt.plot(self.time, rigid_b['qx'], label='qx')
        plt.plot(self.time, rigid_b['qy'], label='qy')
        plt.plot(self.time, rigid_b['qz'], label='qz')
        plt.plot(self.time, rigid_b['qw'], label='qw')
        plt.grid()
        plt.xlabel('time [s]')
        plt.ylabel('quaternion')
        plt.legend()

        plt.subplot(2, 1, 2)
        plt.plot(self.time, rigid_b['x'], label='x')
        plt.plot(self.time, rigid_b['y'], label='y')
        plt.plot(self.time, rigid_b['z'], label='z')
        plt.grid()
        plt.legend()


finger_a = FingerAssignment(4, 6, 3, 2, 7, 5, 0, 1)
body_a = RigidBodyAssignment(0, 1, 2, 3, 4)
data = TestEvaluator(finger_a, body_a)


def read_all_files(idx):
    filename = testfiles[idx]
    print(filename)

    finger_a = FingerAssignment(4, 6, 3, 2, 7, 5, 0, 1)
    body_a = RigidBodyAssignment(0, 1, 2, 3, 4)
    data = TestEvaluator(finger_a, body_a, name=filename)

    # data.plot_rigid_bodies('force_torque')
    data.plot_rigid_bodies('zf_pp')
    data.plot_rigid_bodies('zf_dp')
    data.plot_rigid_bodies('daumen_dp')
    data.plot_rigid_bodies('daumen_mc')


# %%
if __name__ == '__main__':
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

    plt.figure(figsize=(14, 8))
    plt.plot(data.time, f_all)
    plt.grid()
    plt.xlabel('time [s]', fontsize=16)
    plt.ylabel('pincer force [N]', fontsize=16)


# %%
