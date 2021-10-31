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


class TestEvaluator():

    def __init__(self, finger_a: FingerAssignment, body_a: RigidBodyAssignment, name='./data/data.json'):

        # read the json
        with open(name, encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        # extract the main vectors
        self.obs = data['observation']
        self.act = data['action']
        self.time = data['time']

        self.body_a = body_a
        self.finger_a = finger_a
        self.assign_rigid_bodies()


    def assign_rigid_bodies(self):
        for attribute in dir(self.body_a):
            
            if ('_' in attribute[0]) == False:
                print(attribute)
                setattr(self, attribute,
                        self.obs['rigid_bodies'][getattr(self.body_a, attribute)])


finger_a = FingerAssignment(5, 4, 1, 3, 6, 7, 2, 0)
body_a = RigidBodyAssignment(0, 1, 2, 3)
test = TestEvaluator(finger_a, body_a)


# plt.plot(data['time'])
# %%
test.daumen_dp
# %%
body_a.print_atts()
# %%
test.zf_pp
# %%
test.obs
# %%
