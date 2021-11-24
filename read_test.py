# %%
import json
import matplotlib.pyplot as plt
from dataclasses import dataclass

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

    def plot_rigid_bodies(self, call_name):
        """make a simple plot, containing the general quaternion and position data"""
        plt.figure(figsize=(12, 12))
        plt.subplot(2, 1, 1)
        rigid_b = getattr(self, call_name)
        print(rigid_b.keys())
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
test = TestEvaluator(finger_a, body_a)

# %%
test.plot_rigid_bodies('force_torque')
# %%
test.plot_rigid_bodies('zf_pp')
# %%
test.plot_rigid_bodies('zf_dp')
# %%
test.plot_rigid_bodies('daumen_dp')
# %%
test.plot_rigid_bodies('daumen_mc')

# %%
plt.plot(test.obs['analogs'][3]['force'])

# %%
