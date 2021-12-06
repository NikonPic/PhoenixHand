# %% show the stl files
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import numpy as np
from stl import mesh
from pyquaternion import Quaternion
from read_test import data


# %%

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


# %%


class Tracker(object):
    """Class to handle the tracker spheres"""

    def __init__(self, path, name, finger_mesh) -> None:
        super().__init__()
        self.path = path
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

        self.calclate_center()
        self.define_all_axes()
        self.rot_matrix = self.cosy.copy()

    def define_all_axes(self):
        """calculate the coordinate system"""
        # define the axes
        self.x_axis = (self.cogs[2] - self.cogs[1])  # from green to blue
        print(np.linalg.norm(self.x_axis))
        self.x_axis = self.x_axis / np.linalg.norm(self.x_axis)

        midpoint = (self.cogs[2] + self.cogs[1]) / 2
        self.y_axis = self.cogs[0] - midpoint  # from midpoint to red
        self.y_axis = self.y_axis / np.linalg.norm(self.y_axis)

        # finally th z_axis
        self.z_axis = np.cross(self.x_axis, self.y_axis)
        self.z_axis = self.z_axis / np.linalg.norm(self.z_axis)
        self.cosy = np.array([self.x_axis, self.y_axis, self.z_axis])

    def calclate_center(self):
        """calculate the center of the tracker"""
        self.cogs = [get_cog(self.r_sphere),
                     get_cog(self.g_sphere),
                     get_cog(self.b_sphere)]
        self.center = np.mean(self.cogs, axis=0)

    def plot_axis(self, axis, axes, color, leng):
        """plot the tracker in the axes"""
        x, y, z = [self.center[0]], [self.center[1]], [self.center[2]]
        u, v, w = [axis[0]], [axis[1]], [axis[2]]
        axes.quiver(x, y, z, u, v, w, color=color, length=leng)

    def plot_cosys(self, axes):
        self.plot_axis(self.x_axis, axes, 'b', 5)
        self.plot_axis(self.y_axis, axes, 'r', 5)
        self.plot_axis(self.z_axis, axes, 'g', 5)

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

    def plot(self, axes, show_raw=False):
        """plot the trackers"""
        self.plot_raw(axes) if show_raw else None
        self.plot_scatter(axes)
        self.plot_cosys(axes)

    def rotate(self, rot_matrix, center):
        """rotate the trackers"""
        self.r_sphere.rotate_using_matrix(rot_matrix, center)
        self.g_sphere.rotate_using_matrix(rot_matrix, center)
        self.b_sphere.rotate_using_matrix(rot_matrix, center)
        self.calclate_center()
        self.define_all_axes()


class Finger(object):
    """each finger has dp, pip, mcp and two trackers"""

    def __init__(self, name, path, extra=False) -> None:
        super().__init__()
        self.path = path

        # phalanxes
        self.dp = mesh.Mesh.from_file(f'{self.path}_{name} DP.stl')
        self.pip = mesh.Mesh.from_file(f'{self.path}_{name} PIP.stl')
        self.mcp = mesh.Mesh.from_file(f'{self.path}_{name} MCP.stl')

        # trackers
        self.t_dp = Tracker(self.path, f'{name}', self.dp)
        self.t_mcp = Tracker(self.path, f'{name} MCP', self.mcp)

        self.extra = extra
        if extra:
            self.back = mesh.Mesh.from_file(
                f'{self.path}_{name} HANDRÜCKEN.stl')

    def plot(self, axes, alp=0.5):
        """plot the finger"""
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
            self.dp.vectors, color='lightblue', alpha=alp))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
            self.pip.vectors, color='grey', alpha=alp))
        axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
            self.mcp.vectors, color='darkgrey', alpha=alp))

        if self.extra:
            axes.add_collection3d(mplot3d.art3d.Poly3DCollection(
                self.back.vectors, color='silver', alpha=alp))

        self.t_dp.plot(axes)
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


class HandMesh(object):
    """general hand class, which should handle all stl files"""

    def __init__(self, add_bones=False) -> None:
        super().__init__()
        self.path = './segmentations/Segmentation'
        self.thumb = Finger('DAU', self.path)
        self.index = Finger('ZF', self.path, extra=True)

        self.bones = None
        if add_bones:
            self.bones = mesh.Mesh.from_file(f'{self.path}_BONES.stl')

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

    def plot(self):
        """plot the hand"""
        figure = plt.figure(figsize=(14, 14))
        axes = mplot3d.Axes3D(figure)

        # plot surrounding bones
        self.plot_bones(axes)

        # plot fingers
        self.thumb.plot(axes)
        self.index.plot(axes)

        # adjust scale
        scale = self.thumb.dp.points.flatten()
        axes.auto_scale_xyz(scale, scale, scale)

        # name axis and limit range
        axes.set_xlabel('x [mm]')
        axes.set_ylabel('y [mm]')
        axes.set_zlabel('z [mm]')
        axes.set_xlim3d(-100, 50)
        axes.set_ylim3d(50, 200)
        axes.set_zlim3d(-100, 50)

        # Show the plot to the screen
        plt.show()

    def update(self, loc_data):
        """
        Update the Hand using the current information of the measurement
        loc_data contains:
        {
            'thumb': {
                't_dp': {
                    'pos': [x, y, z],
                    'rot': [3x3 rotation matrix]
                }
                't_mcp': ..
            },
            'index': ..
        }
        """
        # define the relevant rotation matrices
        r_ct_0 = self.index.t_mcp.rot_matrix
        r_0_ct = np.linalg.inv(r_ct_0)
        r_0_tr = loc_data['index']['t_mcp']['rot']
        r_tr_0 = np.linalg.inv(r_0_tr)

        # define the relevant vectors


# %%
hand = HandMesh(add_bones=False)
# hand.plot()


# %%
# hand.rotate(np.transpose(hand.index.t_dp.cosy), hand.index.t_dp.center)
hand.plot()
# %%


def get_base_matrix(data, name, ind, scale=1000):
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


def get_offset_ct_tr_and_rot(data, hand: HandMesh):
    """get the offset for the given data"""
    loc_data = build_loc_data(data, 0)

    # get the required rot_matrices
    r_ct_0 = hand.index.t_mcp.rot_matrix
    r_0_tr = loc_data['index']['t_mcp']['rot_matrix']

    # get the required vectors
    v_ct_to_0_in_ct = hand.index.t_mcp.center
    v_tr_to_0_in_tr = loc_data['index']['t_mcp']['pos']

    # calculate the offset
    r_ct_tr = r_ct_0 @ r_0_tr
    v_tr_to_ct_in_ct = r_ct_tr @ v_tr_to_0_in_tr - v_ct_to_0_in_ct

    return v_tr_to_ct_in_ct, r_ct_tr


get_base_matrix(data, 'zf_pp', 1)
offset, r_ct_tr = get_offset_ct_tr_and_rot(data, hand)
loc_data = build_loc_data(data, 0, 0)

# %%
hand.index.t_mcp.center
# %%
loc_data['index']['t_mcp']['pos']
# %%
offset
# %%
