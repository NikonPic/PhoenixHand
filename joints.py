# %%

import json
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import widgets

with open('./data/joints.json') as f:
    data = json.load(f)

data = np.array(data)
data.shape
# %%

color_hand_joints = [[1.0, 0.0, 0.0],
                     [0.0, 0.4, 0.0], [0.0, 0.6, 0.0], [
                         0.0, 0.8, 0.0], [0.0, 1.0, 0.0],  # thumb
                     [0.0, 0.0, 0.6], [0.0, 0.0, 1.0], [
                         0.2, 0.2, 1.0], [0.4, 0.4, 1.0],  # index
                     [0.0, 0.4, 0.4], [0.0, 0.6, 0.6], [
                         0.0, 0.8, 0.8], [0.0, 1.0, 1.0],  # middle
                     [0.4, 0.4, 0.0], [0.6, 0.6, 0.0], [
                         0.8, 0.8, 0.0], [1.0, 1.0, 0.0],  # ring
                     [0.4, 0.0, 0.4], [0.6, 0.0, 0.6], [0.8, 0.0, 0.8], [1.0, 0.0, 1.0]]  # little


class HandJoints(object):
    """
    Data shape:
    0: base
    1: thumb base
    2: thumb joint 1
    3: thumb joint 2
    4: thumb joint 3
    5: index base
    6: index joint 1
    7: index joint 2
    8: index joint 3
    ...
    """

    def __init__(self, data) -> None:
        super().__init__()
        self.data = data
        self.base = data[:, 0, :]
        self.thumb_mcp = data[:, 1, :]
        self.thumb_pip = data[:, 2, :]
        self.thumb_dip = data[:, 3, :]
        self.index_mcp = data[:, 4, :]
        self.index_pip = data[:, 5, :]
        self.index_dip = data[:, 6, :]
        self.filter_data()

    def filter_data(self, tfilt=0.9):
        self.data_filtered = np.zeros(self.data.shape)
        self.data_filtered[0, :, :] = self.data[0, :, :]
        for i in range(1, self.data.shape[0]):
            self.data_filtered[i, :, :] = tfilt * self.data_filtered[i-1, :, :] +(1 - tfilt) * self.data[i, :, :]


    def plot(self, ax, timestamp: int):
        ax.scatter(self.data[timestamp, 0, 0], self.data[timestamp,
                                                         0, 1], self.data[timestamp, 0, 2], c='k')
        self.plot_finger(ax, timestamp, 0, color='r')
        self.plot_finger(ax, timestamp, 1, color='b')
        self.plot_finger(ax, timestamp, 2, color='g')
        self.plot_finger(ax, timestamp, 3, color='m')
        self.define_limits(ax, timestamp)

    def plot_finger(self, ax, timestamp: int, finger: int, color='r'):
        finger_l = list(range(1 + finger * 3, 1 + finger * 3 + 3))
        ax.scatter(self.data[timestamp, finger_l, 0], self.data[timestamp,
                                                                finger_l, 1], self.data[timestamp, finger_l, 2], c=color, s=20)
        finger_l.insert(0, 0)
        ax.plot(self.data[timestamp, finger_l, 0], self.data[timestamp,
                                                             finger_l, 1], self.data[timestamp, finger_l, 2], c=color)

    def define_limits(self, ax, timestamp: int, rang=0.05):
        p0 = self.data[1, 0, :]
        ax.set_xlim3d([p0[0] - rang, p0[0] + rang])
        ax.set_ylim3d([p0[1] - rang, p0[1] + rang])
        ax.set_zlim3d([p0[2] - rang, p0[2] + rang])


idx = widgets.IntSlider(min=0, max=data.shape[0] - 1, value=0)
handj = HandJoints(data)


def draw_3d_skeleton(pose_cam_xyz):
    """
    :param pose_cam_xyz: 21 x 3
    :return:
    """
    assert pose_cam_xyz.shape[0] == 21
    plt.figure(figsize=(16, 16))
    ax = plt.subplot(111, projection='3d')
    marker_sz = 10
    line_wd = 2

    for joint_ind in range(pose_cam_xyz.shape[0]):
        ax.plot(pose_cam_xyz[joint_ind:joint_ind + 1, 0], pose_cam_xyz[joint_ind:joint_ind + 1, 1],
                pose_cam_xyz[joint_ind:joint_ind + 1, 2], '.', c=color_hand_joints[joint_ind], markersize=marker_sz)
        if joint_ind == 0:
            continue
        elif joint_ind % 4 == 1:
            ax.plot(pose_cam_xyz[[0, joint_ind], 0], pose_cam_xyz[[0, joint_ind], 1], pose_cam_xyz[[0, joint_ind], 2],
                    color=color_hand_joints[joint_ind], linewidth=line_wd)
        else:
            ax.plot(pose_cam_xyz[[joint_ind - 1, joint_ind], 0], pose_cam_xyz[[joint_ind - 1, joint_ind], 1],
                    pose_cam_xyz[[joint_ind - 1, joint_ind],
                                 2], color=color_hand_joints[joint_ind],
                    linewidth=line_wd)

    ax.axis('auto')
    x_lim = [-0.1, 0.1, 0.02]
    y_lim = [-0.1, 0.12, 0.02]
    z_lim = [0.0, 0.8, 0.1]
    x_ticks = np.arange(x_lim[0], x_lim[1], step=x_lim[2])
    y_ticks = np.arange(y_lim[0], y_lim[1], step=y_lim[2])
    z_ticks = np.arange(z_lim[0], z_lim[1], step=z_lim[2])
    plt.xticks(x_ticks, [x_lim[0], '', '', '', '',
                         0, '', '', '', x_lim[1]], fontsize=14)
    plt.yticks(y_ticks, [y_lim[0], '', '', '', '', 0,
                         '', '', '', -y_lim[0], ''], fontsize=14)
    ax.set_zticks(z_ticks)
    z_ticks = [''] * (z_ticks.shape[0])
    z_ticks[4] = 0.4
    ax.set_zticklabels(z_ticks, fontsize=14)
    ax.view_init(elev=140, azim=80)
    plt.subplots_adjust(left=-0.06, right=0.98, top=0.93,
                        bottom=-0.07, wspace=0, hspace=0)

handj.filter_data(tfilt=0.95)
def update(idx):
    draw_3d_skeleton(handj.data_filtered[idx, :, :])


widgets.interact(update, idx=idx)

# %%
handj.filter_data(tfilt=0.95)
plt.plot(handj.data_filtered[:, 0, 0])
# %%
