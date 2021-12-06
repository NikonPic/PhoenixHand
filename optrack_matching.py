# %% imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

"""
Goal is to find the Transformation Matrix -> Red, Green, Blue -> Tracker System

1. We need to determine, which Marker is the Red, green and Blue one for each Tracker
2. We need the Coordinate System of the Red Green Blue System
3. We need the Transformation Matrix from the Red Green Blue System to the Tracker System
"""


# %%
path_csv = './data/Take 2021-12-06 03.42.36 PM.csv'
data = pd.read_csv(path_csv, header=2)
data.head()
# %%
# list of all the trackers
switches = [False, False, False, False]
tr_list = ['ZF DP', 'ZF MC', 'DAU DP', 'DAU MC']

# List of all values
tr_data = ['qx', 'qy', 'qz', 'qw', 'x', 'y', 'z']
tr_appd = ['', '.1', '.2', '.3', '.4', '.5', '.6']

# list of x, y, z
mk_data = ['x', 'y', 'z']
mk_appd = ['', '.1', '.2']

markers = ['1', '2', '3']

start_id = 10
end_id = 100


def get_m(data, name, start_id=10, end_id=200):
    arr = np.array([float(d) for d in data[name][start_id:end_id]])
    arr = arr[np.logical_not(np.isnan(arr))]
    return np.mean(arr)


class TrackerOpt(object):
    def __init__(self, name, data, switch=False):

        for tr_d, tr_app in zip(tr_data, tr_appd):
            setattr(self, tr_d, get_m(data, name + tr_app))
        self.name = name
        self.quat = [self.qx, self.qy, self.qz, self.qw]
        self.pos = [self.x, self.y, self.z]

        self.cogs = []
        for marker in markers:
            marker_res = []
            for mk_app in mk_appd:
                res = get_m(data, f'{name}:Marker{marker}{mk_app}')
                marker_res.append(res)
            setattr(self, f'marker{marker}', marker)
            self.cogs.append(np.array(marker_res))

        print(self.cogs)
        self.center = np.mean(self.cogs, axis=0)

        # next calculate differences between trackers to determine the red one:
        self.diffs = [
            np.linalg.norm(self.cogs[1] - self.cogs[2]),
            np.linalg.norm(self.cogs[0] - self.cogs[2]),
            np.linalg.norm(self.cogs[0] - self.cogs[1]),
        ]
        sel1 = self.diffs.index(min(self.diffs))
        sel2 = self.diffs.index(max(self.diffs))
        sel3 = list(set(range(3)).difference([sel1, sel2]))[0]

        self.red = self.cogs[sel1]
        self.green = self.cogs[sel2]
        self.blue = self.cogs[sel3]

        print(self.red, self.green, self.blue)

        if switch:
            self.green = markers[sel3]
            self.blue = markers[sel2]

        self.define_all_axes()

    def define_all_axes(self):
        """calculate the coordinate system"""
        # define the axes
        self.x_axis = (self.blue - self.green)  # from green to blue
        print(np.linalg.norm(self.x_axis))
        self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)

        midpoint = (self.blue + self.green) / 2
        self.y_axis = self.red - midpoint  # from midpoint to red
        self.y_axis = self.y_axis / np.linalg.norm(self.y_axis)

        # finally th z_axis
        self.z_axis = np.cross(self.x_axis, self.y_axis)
        self.z_axis = self.z_axis / np.linalg.norm(self.z_axis)
        self.cosy = np.array([self.x_axis, self.y_axis, self.z_axis])

    def plot_cosys(self, axes):
        self.plot_axis(self.x_axis, axes, 'b', 5)
        self.plot_axis(self.y_axis, axes, 'r', 5)
        self.plot_axis(self.z_axis, axes, 'g', 5)

    def plot_axis(self, axis, axes, color, leng):
        """plot the tracker in the axes"""
        x, y, z = [self.center[0]], [self.center[1]], [self.center[2]]
        u, v, w = [axis[0]], [axis[1]], [axis[2]]
        axes.quiver(x, y, z, u, v, w, color=color, length=leng)

    def plot_scatter(self, axes, r=0.75, alp=0.2):
        """plot the trackers"""
        axes.scatter(self.red[0], self.red[1], self.red[2],
                     color='red', alpha=alp, s=np.pi*r**2*100)
        axes.scatter(self.green[0], self.green[1], self.green[2],
                     color='green', alpha=alp, s=np.pi*r**2*100)
        axes.scatter(self.blue[0], self.blue[1], self.blue[2],
                     color='blue', alpha=alp, s=np.pi*r**2*100)

    def plot(self, axes):
        self.plot_scatter(axes)
        self.plot_cosys(axes)


figure = plt.figure(figsize=(14, 14))
axes = mplot3d.Axes3D(figure)

for tr in tr_list:
    tr_obj = TrackerOpt(tr, data)
    tr_obj.plot(axes)

axes.set_xlabel('x [mm]')
axes.set_ylabel('y [mm]')
axes.set_zlabel('z [mm]')
plt.show()
# %%
tr_obj.red
# %%
