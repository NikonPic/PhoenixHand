
# %%
import numpy as np
import sys
import signal
from datetime import datetime
from argmax_gym import TcpRemoteGym
import json


def millis():
    """return the time since start in milliseconds"""
    dt = datetime.now() - start_time
    ms = (dt.days * 24 * 60 * 60 + dt.seconds) * 1000 + dt.microseconds / 1000.0
    return ms

def sigint_handler(signal, frame):
    print('Interrupted')
    sys.exit(0)

signal.signal(signal.SIGINT, sigint_handler)

batch_size = 1

env = TcpRemoteGym(
    "phoenix-v0",
    hosts=["141.39.191.228"],
    target_device="x86_64",
    prefix="phoenix"
)

print('init')
obs, image = env.reset(record=False)
print('reset')

action_off = np.array([[
#       1    2    3    4    5    6    7    8   
        1.5, 1.5, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]])

action_amp = np.array([[
#       1    2    3     4     5    6    7    8  
        1.5, 1.5, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
]])


dt = 0.01
t = 0
start_time = datetime.now()

json_out = {
    'time': [],
    'action' : [],
    'observation' : [],
}


maxtime_s = 10
millisec = 0
printcount = 0


while (millisec < (maxtime_s * 1000)):
    millisec = millis()
    
    t += dt
    action =  5 * action_off +  7 * action_amp * np.cos(t)
    obs, reward, done, image = env.step(action)

    append_obs = obs[0].tolist()

    # add to json_out:
    json_out['time'].append(millisec)
    json_out['action'].append(action.tolist())
    json_out['observation'].append(append_obs)

    printcount += 1
    if printcount > 1000:
        print(millisec)
        print(type(append_obs))
        print(len(append_obs))
        print(len(json_out['observation'][-1]))
        printcount = 0

with open('latest.json', 'w') as f:
    json.dump(json_out, f, indent=2)

# %%