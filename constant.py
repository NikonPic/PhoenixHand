
# %%
import numpy as np
import sys
import signal
from datetime import datetime
from argmax_gym import TcpRemoteGym
import json
from datasplit import ObservationHandler


def millis():
    """return the time since start in milliseconds"""
    dt = datetime.now() - start_time
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * \
        1000 + dt.microseconds / 1000.0
    return ms


def sigint_handler(signal, frame):
    print('Interrupted')
    sys.exit(0)


signal.signal(signal.SIGINT, sigint_handler)
batch_size = 1
obs_handler = ObservationHandler(num_rigid_bodies=5)

env = TcpRemoteGym(
    "phoenix-v0",
    hosts=["141.39.191.228"],
    target_device="x86_64",
    prefix="phoenix"
)

print('init')
obs, image = env.reset(record=False)
print('reset')

# Motor nummern:

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


"""


action_off = np.array([[
    # 0    1    2    3    4    5    6    7
    1, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0, 1.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]])

action_amp = np.array([[
    # 0    1    2    3    4    5    6    7
    1.0, -1.5, 1.0, -1.5, 1.5, 1.5, -0.5, -0.5,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    #0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]])


dt = 0.01
t = 0
start_time = datetime.now()

json_out = {
    'time': [],
    'action': [],
}


maxtime_s = 30
millisec = 0
printcount = 0


while (millisec < (maxtime_s * 1000)):
    millisec = millis()

    t += dt
    action = 5 * action_off + 5 * action_amp * np.cos(t)
    obs, reward, done, image = env.step(action)

    append_obs = obs[0].tolist()

    if obs_handler(append_obs):
        # add to json_out:
        json_out['time'].append(millisec)
        json_out['action'].append(action.tolist())

    printcount += 1
    if printcount > 1000:
        printcount = 0


print('finished')
json_out['observation'] = obs_handler.output_dict

with open('data/latest.json', 'w') as f:
    json.dump(json_out, f, indent=2)

# %%
