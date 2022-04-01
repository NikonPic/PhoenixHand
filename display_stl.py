# %% show the stl files
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.editor import VideoClip
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from stl import mesh
from pyquaternion import Quaternion
from read_test import data
from optrack_matching import opttr, optdict, TrackerOpt
from ipywidgets import widgets


def get_angle(p, q):
    """get the angle between two rot matrices for evaluation"""
    rot = np.dot(p, q.T)
    theta = (np.trace(rot) - 1)/2
    return np.arccos(theta) * (180/np.pi)


def get_loc_sphere(loc_mesh: mesh.Mesh, vector0, min_dist):
    """get the local distances to all other points and collect the minimum values"""
    all_dist = []
    for vector in loc_mesh.vectors:
        dist = np.linalg.norm(vector - vector0)
        all_dist.append(dist)

    all_idx = np.where([dist > -1 for dist in all_dist])[0]
    idx_list = np.where([dist < min_dist for dist in all_dist])[0]
    idx_list = idx_list.tolist()

    sphere = [loc_mesh.vectors[i] for i in idx_list]

    # create new sphere
    data = np.zeros(len(sphere), dtype=mesh.Mesh.dtype)
    for idx, vector in enumerate(sphere):
        data['vectors'][idx] = vector

    new_sphere = mesh.Mesh(data)

    return new_sphere, all_idx, idx_list


def get_triple_header(loc_mesh: mesh.Mesh, min_dist: float = 15):
    """
    Returns the three heads of the tracker spheres
    1. start random at point 0
    2. get min distances to 0 and collect to 1. sphere
    3. get to next point outside min distance
    """
    # 1. sphere 0
    vector0 = loc_mesh.vectors[0]
    sphere0, all_idx, idx_list0 = get_loc_sphere(loc_mesh, vector0, min_dist)

    # 2 get next point outside
    remain_idx = list(set(all_idx) - set(idx_list0))
    vector0 = loc_mesh.vectors[remain_idx[0]]
    sphere1, _, idx_list1 = get_loc_sphere(loc_mesh, vector0, min_dist)

    # 3 get next point outside
    remain_idx = list(set(all_idx) - set(idx_list0) - set(idx_list1))
    vector0 = loc_mesh.vectors[remain_idx[0]]
    sphere2, _, _ = get_loc_sphere(loc_mesh, vector0, min_dist)

    # finally get the means for each of the received points
    return sphere0, sphere1, sphere2


def get_cog(loc_mesh: mesh.Mesh, as_mean=True):
    """Returns the center of gravity of the tracker spheres"""
    if as_mean:
        return np.mean(np.mean(loc_mesh.vectors, axis=0), axis=0)
    else:
        return loc_mesh.get_mass_properties()[1]


class Tracker(object):
    """Class to handle the tracker spheres"""

    def __init__(self, path, name, finger_mesh: mesh.Mesh, opt_tr: TrackerOpt, my_mesh: mesh.Mesh) -> None:
        super().__init__()
        self.path = path
        self.name = name
        self.my_mesh = my_mesh
        self.loc_mesh = mesh.Mesh.from_file(f'{self.path}_TRACKER {name}.stl')
        sphere0, sphere1, sphere2 = get_triple_header(
            self.loc_mesh)
        spheres = [sphere0, sphere1, sphere2]

        # define the tracker with maximum distance to finger cog
        self.cog_finger = finger_mesh.get_mass_properties()[1]

        self.cogs = [get_cog(sphere0),
                     get_cog(sphere1),
                     get_cog(sphere2)]

        self.d_finger = [np.linalg.norm(self.cogs[0] - self.cog_finger),
                         np.linalg.norm(self.cogs[1] - self.cog_finger),
                         np.linalg.norm(self.cogs[2] - self.cog_finger),
                         ]

        idx_max = np.where(self.d_finger == np.max(self.d_finger))[0][0]
        idx_min = np.where(self.d_finger == np.min(self.d_finger))[0][0]

        self.r_sphere = spheres[idx_max]
        self.g_sphere = spheres[list(
            set([0, 1, 2]) - set([idx_max, idx_min]))[0]]
        self.b_sphere = spheres[idx_min]

        self.calculate_center()
        self.define_all_axes()
        self.rot_matrix = self.cosy.copy()
        self.opt_tr = opt_tr
        self.define_scale()
        self.define_const_rot()

    def define_scale(self):
        self.scale = self.length / self.opt_tr.length

    def define_const_rot(self):
        """define the constant rotation matrix"""
        self.t_tr_ct = self.rot_matrix
        self.t_ct_tr = self.t_tr_ct.T

        self.t_q_tr = self.opt_tr.t_q_tr
        self.t_tr_q = self.t_q_tr.T

    def define_all_axes(self):
        """calculate the coordinate system"""
        # define the axes
        self.x_axis = (self.cogs[2] - self.cogs[1])  # from green to blue
        self.length = np.linalg.norm(self.x_axis)
        self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)

        midpoint = (self.cogs[2] + self.cogs[1]) / 2
        self.y_axis = self.cogs[0] - midpoint  # from midpoint to red
        self.y_axis = self.y_axis / np.linalg.norm(self.y_axis)

        # finally th z_axis
        self.z_axis = np.cross(self.x_axis, self.y_axis)
        self.z_axis = self.z_axis / np.linalg.norm(self.z_axis)
        self.optimize_cosy()

        # finally the rotation matrices
        self.t_tr_ct = self.cosy
        self.t_ct_tr = self.cosy.T

    def calculate_center(self):
        """calculate the center of the tracker"""
        self.cogs = [get_cog(self.r_sphere),
                     get_cog(self.g_sphere),
                     get_cog(self.b_sphere)]
        self.center = np.mean(self.cogs, axis=0)
        self.pos = self.center

    def plot_axis(self, axis, axes, color, leng):
        """plot the tracker in the axes"""
        x, y, z = [self.center[0]], [self.center[1]], [self.center[2]]
        u, v, w = [axis[0]], [axis[1]], [axis[2]]
        axes.quiver(x, y, z, u, v, w, color=color, length=leng)

    def optimize_cosy(self):
        """optimize the tracker"""
        x, y, z = self.x_axis, self.y_axis, self.z_axis

        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)

        y = np.cross(z, x)
        z = np.cross(x, y)
        x = np.cross(y, z)

        x = x / np.linalg.norm(x)
        y = y / np.linalg.norm(y)
        z = z / np.linalg.norm(z)

        self.x_axis, self.y_axis, self.z_axis = x, y, z
        self.cosy = np.array([self.x_axis, self.y_axis, self.z_axis]).T

    def plot_cosys(self, axes):
        self.plot_axis(self.x_axis, axes, 'b', 5)
        self.plot_axis(self.y_axis, axes, 'r', 5)
        self.plot_axis(self.z_axis, axes, 'g', 5)

        axes.text(self.center[0], self.center[1],
                  self.center[2], self.name, color='k')

    def plot_raw(self, axes):
        """plot the trackers"""
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
            self.r_sphere.vectors, color='red', alpha=0.5))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
            self.g_sphere.vectors, color='green', alpha=0.5))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
            self.b_sphere.vectors, color='blue', alpha=0.5))

    def plot_scatter(self, axes, r=0.75, alp=0.2):
        """plot the trackers"""
        axes.scatter(self.cogs[0][0], self.cogs[0][1], self.cogs[0][2],
                     color='red', alpha=alp, s=np.pi*r**2*100)
        axes.scatter(self.cogs[1][0], self.cogs[1][1], self.cogs[1][2],
                     color='green', alpha=alp, s=np.pi*r**2*100)
        axes.scatter(self.cogs[2][0], self.cogs[2][1], self.cogs[2][2],
                     color='blue', alpha=alp, s=np.pi*r**2*100)

        cog_mesh = get_cog(self.my_mesh)
        axes.scatter(cog_mesh[0], cog_mesh[1], cog_mesh[2],
                     color='k', alpha=alp, s=np.pi*r**2*100)

    def plot(self, axes, show_raw=False, show_mesh=False):
        """plot the trackers"""
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
            self.my_mesh.vectors, color='lightblue', alpha=0.5)) if show_mesh else None
        self.plot_raw(axes) if show_raw else None
        self.plot_scatter(axes)
        self.plot_cosys(axes)

    def transform(self, rot_matrix, diff):
        """transform the tracker"""
        tr_mat = np.zeros((4, 4))
        tr_mat[:3, :3] = rot_matrix
        tr_mat[:3, 3] = diff

        self.r_sphere.transform(tr_mat)
        self.g_sphere.transform(tr_mat)
        self.b_sphere.transform(tr_mat)
        self.calculate_center()
        self.define_all_axes()

    def rotate(self, rot_matrix, center, verbose=False):
        """rotate the trackers"""
        self.my_mesh.rotate_using_matrix(rot_matrix, center)
        self.r_sphere.rotate_using_matrix(rot_matrix, center)
        self.g_sphere.rotate_using_matrix(rot_matrix, center)
        self.b_sphere.rotate_using_matrix(rot_matrix, center)
        self.calculate_center()
        self.define_all_axes()

        if verbose:
            print(np.around(self.t_tr_ct, decimals=1))

    def translate(self, diffpos):
        """translate the trackers"""
        self.my_mesh.translate(diffpos)
        self.r_sphere.translate(diffpos)
        self.g_sphere.translate(diffpos)
        self.b_sphere.translate(diffpos)
        self.calculate_center()
        self.define_all_axes()

    def update(self, t_ct_opt, offset, scale, loc_data, verbose=False, ct_sys=False):
        """
        Update the finger system using the offset, the scale and the new loc_data
        loc_data : {
            'pos' : [x,y,z]
            'rot_matrix' : [..]
        }
        - also use the transformation from ct->opt
        - perform updates
        - return the diff and rotation applyed
        """
        # 1. take data from loc-data
        pos = loc_data['pos']
        t_q_opt = loc_data['rot_matrix']

        if ct_sys:
            new_pos = t_ct_opt @ np.array(pos) * scale + offset
            diff_pos = (new_pos - self.center)

            # 3. calculate the difference in rotation
            t_tr_q = self.t_tr_q
            t_ct_old = self.t_ct_tr
            t_neu_old = t_tr_q @ t_q_opt @ t_ct_opt.T @ t_ct_old

            self.rotate(t_neu_old.T, self.center, verbose=verbose)
            self.translate(diff_pos)

        else:  # use optitrack
            # 2. calculate the difference in position
            new_pos = np.array(pos) * scale
            diff_pos = (new_pos - self.center)

            # 3. calculate the difference in rotation
            t_tr_q = self.t_tr_q
            t_ct_old = self.t_ct_tr
            t_neu_base = t_q_opt

            self.rotate(t_ct_old.T, self.center, verbose=verbose)
            self.rotate(t_neu_base.T, self.center, verbose=verbose)
            self.translate(diff_pos)

        # 4. return diffpos and rotation
        if verbose:
            print(self.name)
            print('cur rot:')
            print(np.around(self.t_tr_ct, decimals=1))

            print('des rot:')
            print(np.around(t_neu_base, decimals=1))

            print('req_rot:')
            print(np.around(t_neu_base, decimals=1))

            print('final1:')


class Finger(object):
    """each finger has dp, pip, mcp and two trackers"""

    def __init__(self, name, path, opttr_finger: dict, extra=False) -> None:
        super().__init__()
        self.path = path
        self.name = name

        # phalanxes
        self.dp = mesh.Mesh.from_file(f'{self.path}_{name} DP.stl')
        self.pip = mesh.Mesh.from_file(f'{self.path}_{name} PIP.stl')
        self.mcp = mesh.Mesh.from_file(f'{self.path}_{name} MCP.stl')

        self.extra = extra
        if extra:
            self.back = mesh.Mesh.from_file(
                f'{self.path}_{name} HANDRÜCKEN.stl')

        # trackers
        dp_mesh = self.dp
        self.t_dp = Tracker(
            self.path, f'{name}', self.dp, opttr_finger['t_dp'], dp_mesh)

        mcp_mesh = self.back if self.extra else self.mcp
        self.t_mcp = Tracker(
            self.path, f'{name} MCP', self.mcp, opttr_finger['t_mcp'], mcp_mesh)

    def update(self, loc_data: dict, t_ct_opt, offset, scale, ct_sys=False):
        """
        update the position and rotation of the fingers by applying the optitrack data
        loc_data contains:
        {
            't_dp': {
                'pos': [x, y, z],
                'rot_matrix': [3x3 rotation matrix]
                }
            't_mcp': ..
        }
        """
        self.t_dp.update(t_ct_opt, offset, scale,
                         loc_data['t_dp'], ct_sys=ct_sys)
        self.t_mcp.update(t_ct_opt, offset, scale,
                          loc_data['t_mcp'], ct_sys=ct_sys)

    def plot(self, axes, alp=0.5, plot_extra=True):
        """plot the finger"""

        if plot_extra:
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
                self.pip.vectors, color='grey', alpha=alp))

            if self.extra:
                axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
                    self.mcp.vectors, color='darkgrey', alpha=alp))

        self.t_dp.plot(axes, show_mesh=True)
        self.t_mcp.plot(axes)

    def rotate(self, rot_matrix, center):
        """rotate the finger"""
        self.dp.rotate_using_matrix(rot_matrix, point=center)
        self.pip.rotate_using_matrix(rot_matrix, point=center)
        self.mcp.rotate_using_matrix(rot_matrix, point=center)

        if self.extra:
            self.back.rotate_using_matrix(rot_matrix, point=center)

        self.t_dp.rotate(rot_matrix, center)
        self.t_mcp.rotate(rot_matrix, center)

    def translate(self, diffpos):
        """translate the finger"""
        self.dp.translate(diffpos)
        self.pip.translate(diffpos)
        self.mcp.translate(diffpos)

        if self.extra:
            self.back.translate(diffpos)


class HandMesh(object):
    """general hand class, which should handle all stl files"""

    def __init__(self, opttr: dict,  add_bones=False, ) -> None:
        """
        opttr: dictionary with the information about the trackers and their transformations
        """
        super().__init__()
        self.path = './segmentations/Segmentation'
        self.thumb = Finger('DAU', self.path, opttr['thumb'])
        self.index = Finger('ZF', self.path, opttr['index'], extra=True)

        self.bones = None
        if add_bones:
            self.bones = mesh.Mesh.from_file(f'{self.path}_BONES.stl')

        self.get_scale()

    def rotate(self, rot_matrix, center):
        """rotate the hand"""
        self.thumb.rotate(rot_matrix, center)
        self.index.rotate(rot_matrix, center)

        if self.bones:
            self.bones.rotate(rot_matrix, center)

    def plot_bones(self, axes):
        """plot the bones"""
        if self.bones is not None:
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
                self.bones.vectors, color='gold', alpha=0.05))

    def define_limits(self, ct_sys=False):
        """define the limits of the hand"""
        self.y_min, self.y_max = 20, 220
        self.z_min, self.z_max = -200, 0
        self.x_min, self.x_max = -300, -100

        if ct_sys:
            self.x_min, self.x_max = -200, 0
            self.z_min, self.z_max = -100, 100

    def plot(self, plot_extra=False):
        """plot the hand"""
        figure = plt.figure(figsize=(8, 8))
        axes = mplot3d.Axes3D(figure)

        # plot surrounding bones
        self.plot_bones(axes)

        # plot fingers
        self.thumb.plot(axes, plot_extra=plot_extra)
        self.index.plot(axes, plot_extra=plot_extra)

        # adjust scale
        scale = self.thumb.dp.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # name axis and limit range
        axes.set_xlabel('x [mm]')
        axes.set_ylabel('y [mm]')
        axes.set_zlabel('z [mm]')

        axes.set_xlim3d(self.x_min, self.x_max)
        axes.set_ylim3d(self.y_min, self.y_max)
        axes.set_zlim3d(self.z_min, self.z_max)

        return figure

    def update(self, loc_data, ct_sys=False):
        """
        Update the Hand using the current information of the measurement
        loc_data contains:
        {
            'thumb': {
                't_dp': {
                    'pos': [x, y, z],
                    'rot_matrix': [3x3 rotation matrix]
                }
                't_mcp': ..
            },
            'index': ..
        }
        """
        self.define_limits(ct_sys)

        # define the relevant rotation matrices
        t_ct_opt = self.get_rot_opt_to_ct(
            loc_data['index']['t_mcp']['rot_matrix'])
        self.t_ct_opt = t_ct_opt
        offset = self.get_offset_ct_opt(loc_data)

        # update the hand
        self.thumb.update(loc_data['thumb'], t_ct_opt,
                          offset, self.scale, ct_sys)
        self.index.update(loc_data['index'], t_ct_opt,
                          offset, self.scale, ct_sys)

    def get_scale(self):
        """get the scale between opt and ct sys"""
        self.scale = self.index.t_mcp.scale

    def get_rot_opt_to_ct(self, t_q_opt):
        """get the rotation matrix from opt sys to ct sys - using the current position of the index mcp sys"""
        t_ct_tr = self.index.t_mcp.t_ct_tr
        t_tr_q = self.index.t_mcp.t_tr_q
        t_ct_opt = t_ct_tr @ t_tr_q @ t_q_opt
        return t_ct_opt

    def get_offset_ct_opt(self, loc_data: dict):
        """get the offset between opt and ct sys"""
        offset = self.index.t_mcp.pos - \
            self.t_ct_opt @ loc_data['index']['t_mcp']['pos'] * self.scale
        return offset


def get_base_matrix(data, name, ind, scale=1):
    """get the quaternion base for the given index"""
    loc_dict = getattr(data, name)
    qw, qx, qy, qz = loc_dict['qw'], loc_dict['qx'], loc_dict['qy'], loc_dict['qz']
    x, y, z = loc_dict['x'], loc_dict['y'], loc_dict['z']
    q = Quaternion(qw[ind], qx[ind], qy[ind], qz[ind])
    pos = [x[ind] * scale, y[ind] * scale, z[ind]*scale]
    return q.rotation_matrix, pos


def build_loc_data(data, ind, scale=1000):
    """build the location data for the given index"""
    loc_dict = {}
    # thumb
    loc_dict['thumb'] = {}
    rot_matirx, pos = get_base_matrix(data, 'daumen_dp', ind, scale)
    loc_dict['thumb']['t_dp'] = {
        'pos': pos,
        'rot_matrix': rot_matirx
    }
    rot_matirx, pos = get_base_matrix(data, 'daumen_mc', ind, scale)
    loc_dict['thumb']['t_mcp'] = {
        'pos': pos,
        'rot_matrix': rot_matirx
    }
    # index
    loc_dict['index'] = {}
    rot_matirx, pos = get_base_matrix(data, 'zf_dp', ind, scale)
    loc_dict['index']['t_dp'] = {
        'pos': pos,
        'rot_matrix': rot_matirx
    }
    rot_matirx, pos = get_base_matrix(data, 'zf_pp', ind, scale)
    loc_dict['index']['t_mcp'] = {
        'pos': pos,
        'rot_matrix': rot_matirx
    }
    return loc_dict


def update_all(ind, plotit=False, plot_extra=False, set_scale=False, scale=1.0, ct_sys=False):
    loc_data = build_loc_data(data, ind)
    if set_scale:
        hand.scale = scale
    hand.update(loc_data, ct_sys=ct_sys)

    if plotit:
        hand.plot(plot_extra=plot_extra)
        plt.savefig('./imaging/current.png')


def makre_frame(t):
    """make a frame for the hand"""
    # t in s -> index
    ind = int(round(t * 100))
    loc_data = build_loc_data(data, ind)
    hand.update(loc_data, ct_sys=True)
    fig = hand.plot(plot_extra=False)
    return mplfig_to_npimage(fig)


def generate_video(fps=5, dur=25):
    """generate a video of the hand"""

    # save the frames
    animation = VideoClip(make_frame=makre_frame, duration=dur)
    animation.write_videofile("my_animation.mp4", fps=fps)


# %%
if __name__ == '__main__':
    idx = widgets.IntSlider(value=2000, min=0, max=len(data.time))
    hand = HandMesh(opttr, add_bones=False)
    widgets.interact(update_all, ind=idx)

    generate_video()

# %%