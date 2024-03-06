# %%
import json
import os
import imageio
from PIL import Image
from itertools import permutations
from read_measurement_data import TestEvaluator
from markers import Marker, read_pointdata, copy_xml_values, copy_xml_to_yaml, create_control_array
from tqdm import tqdm
from dm_control import mujoco
from config import AxisInfos, HandSettings, JointInfo, TrackSettings
from scipy.spatial.transform import Rotation as R
from hydra.core.config_store import ConfigStore
import hydra
import xml.etree.ElementTree as ET
from pyquaternion import Quaternion
from stl import mesh
from scipy.spatial.distance import pdist, squareform
import pandas as pd
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import csv
from pyvirtualdisplay import Display
display = Display(visible=0, size=(1080, 1080))
display.start()


# os.environ['MUJOCO_GL'] = 'egl'

TENDON = 'tendon'
# required for video rendering on remote server
# make sure to isnall with: sudo apt-get install firefox xvfb


cs = ConfigStore.instance()
cs.store(name='hand_config', node=HandSettings)


def sort_points_relative(list_points_1, list_points_2):
    """sort a list of points based on their relative distances"""
    # Convert lists to numpy arrays
    list_points_1 = np.array(list_points_1)
    list_points_2 = np.array(list_points_2)

    # Compute pairwise distances for each list
    distances_1 = squareform(pdist(list_points_1))
    distances_2 = squareform(pdist(list_points_2))

    # Sum the distances for each point
    sum_distances_1 = np.sum(distances_1, axis=1)
    sum_distances_2 = np.sum(distances_2, axis=1)

    # Get the sorted indices based on the sum of distances
    sorted_indices_1 = np.argsort(sum_distances_1)
    sorted_indices_2 = np.argsort(sum_distances_2)

    # Sort the points based on the computed indices
    sorted_list_points_1 = list_points_1[sorted_indices_1]
    sorted_list_points_2 = list_points_2[sorted_indices_2]

    return sorted_list_points_1.tolist(), sorted_list_points_2.tolist()


def kabsch(p, q):
    """Calculate the optimal rigid transformation matrix from Q -> P using Kabsch algorithm"""

    centroid_p = np.mean(p, axis=0)
    centroid_q = np.mean(q, axis=0)

    p_centered = p - centroid_p
    q_centered = q - centroid_q

    H = np.dot(p_centered.T, q_centered)

    U, _, vt = np.linalg.svd(H)

    R = np.dot(vt.T, U.T)

    if np.linalg.det(R) < 0:
        vt[-1, :] *= -1
        R = np.dot(vt.T, U.T)

    t = centroid_q - np.dot(centroid_p, R.T)

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t

    return T


def get_def_tracker_from_csv(df, tracker_name):
    """
    get the optitrack measure data for the tracker
    -> format:
    x, y, z, qw, qx, qy, qz
    x1, y1, z1, x2, y2, z2, x3, y3, z3
    """
    start_column = df.columns.get_loc(tracker_name)
    data = df.iloc[3, start_column:start_column+16].astype(float)

    qw, qx, qy, qz, x, y, z, x1, y1, z1, x2, y2, z2, x3, y3, z3 = data

    quat = Quaternion(qx=qx, qy=qy, qz=qz, qw=qw)
    v = np.array([x, y, z])
    ms = [np.array([xi, yi, zi, 1]) for xi, yi, zi in zip(
        [x1, x2, x3], [y1, y2, y3], [z1, z2, z3])]

    t_mat = np.eye(4)
    t_mat[:3, :3] = quat.rotation_matrix
    t_mat[:3, 3] = v
    t_mat_inv = np.linalg.inv(t_mat)

    transformed_ms = [t_mat_inv @ mi for mi in ms]

    return [mi[:3] for mi in transformed_ms]


def csv_test_load(df, tracker_name):
    """get the optitrack measure data for the tracker -> format: x, y, z, qw, qx, qy, qz"""
    start_coloum = df.columns.get_loc(tracker_name)
    data = df.values[3:, start_coloum:start_coloum+7]
    data = np.array([list(map(float, i)) for i in data])
    return data


def update_all_pos_quat(element, name, pos, quat):
    """Updates the position and quaternion of an element with the specified name."""
    # Check if the current element is a 'body' with the matching name
    if 'name' in element.attrib and element.attrib['name'] == name:
        element.attrib['pos'] = write_arr_xml(pos)
        element.attrib['quat'] = write_arr_xml(quat)

    # Recursively update children
    for child in element:
        update_all_pos_quat(child, name, pos, quat)


def write_arr_xml(arr):
    """Converts an array of elements to a space-separated string."""
    return ' '.join(str(elem) for elem in arr)


def find_element_parent(element, parent, name):
    if 'name' in element.attrib and element.attrib['name'] == name:
        return parent
    res = None
    for child in element:
        res = find_element_parent(child, element, name)
        if res is not None:
            return res
    return res


def update_all_joint(element, name, pos, axis):
    """Updates the position and quaternion of an element with the specified name."""
    # Check if the current element is a 'body' with the matching name
    if 'name' in element.attrib and element.attrib['name'] == name:
        element.attrib['pos'] = write_arr_xml(pos)
        element.attrib['axis'] = write_arr_xml(axis)

    # Recursively update children
    for child in element:
        update_all_joint(child, name, pos, axis)


def update_all_wrap(element, name, p1, p2, size=0.003):
    p1p2_arr = list(p1) + list(p2)
    if 'name' in element.attrib and element.attrib['name'] == name:
        element.attrib['fromto'] = write_arr_xml(p1p2_arr)
        element.attrib['size'] = write_arr_xml([size])
        element.attrib['density'] = "0"
        element.attrib['contype'] = "0"
        element.attrib['conaffinity'] = "0"

    # Recursively update children
    for child in element:
        update_all_wrap(child, name, p1, p2, size)


def load_all_tendons_to_dict(path):
    """laod all available tendons and assign to dict"""
    files = os.listdir(path)
    tendon_dict = {}

    # go trough all files in dir
    for filename in files:

        if filename.endswith('.mrk.json'):
            filename = f'{path}/{filename}'
            with open(filename) as fp:
                loc_data = json.load(fp)

            pointlist = loc_data['markups'][0]['controlPoints']

            for point in pointlist:
                pointname = point['label']
                position = point['position']
                tendon_dict[pointname] = position

    return tendon_dict


def recursive_site(elem, name, tendon_name, pos, scale=0.001):
    if 'name' in elem.attrib.keys() and elem.attrib['name'] == name:
        for loc_elem in elem:
            if 'name' in loc_elem.attrib.keys() and loc_elem.attrib['name'] == tendon_name:
                loc_elem.attrib['pos'] = write_arr_xml(pos)
                loc_elem.attrib['type'] = 'sphere'
                loc_elem.attrib['size'] = f'{scale}'
                return
        # we looped over all so the site does not exist yet
        attrib = {
            'name': tendon_name,
            'pos': write_arr_xml(pos),
            'type': 'sphere',
            'size': f'{scale}',
        }
        elem.makeelement('site', attrib)
        ET.SubElement(elem, 'site', attrib)

    for loc_elem in elem:
        recursive_site(loc_elem, name, tendon_name, pos, scale=scale)


class Tracker:

    def __init__(self, cfg: TrackSettings, df: pd.DataFrame, data, off_xyz, off_wxyz,
                 scale, verbose=False, use_gym=False, tendon_dict={}):
        """init a tracker object"""
        self.cfg = cfg
        self.mj_name = cfg.mj_name
        self.def_path = cfg.def_path
        self.ct_path = cfg.ct_path
        self.axis_path = cfg.axis_path
        self.your_mesh = mesh.Mesh.from_file(cfg.stl_path)
        self.off_xyz = off_xyz
        self.off_wxyz = off_wxyz
        self.scale = scale
        self.q_offset = Quaternion(self.off_wxyz)
        self.use_gym = use_gym
        self.tendon_names = cfg.point_list
        self.init_tendon_points(tendon_dict)

        if verbose:
            plt.figure(cfg.def_path)
            plt.subplot(211)
            plt.plot(self.measure_data[:, :4])
            plt.subplot(212)
            plt.plot(self.measure_data[:, 4:])
            plt.show()

        # read the point lists
        self.marker_pos_def = self.read_markerdata(self.def_path, df)
        self.read_ctdata()
        self.read_axis_data()

        if self.cfg.resort:
            # sort the point lists by relative distance
            self.marker_pos_def, self.marker_pos_ct = sort_points_relative(
                self.marker_pos_def, self.marker_pos_ct)
        else:
            self.marker_pos_ct = np.array(self.marker_pos_ct)[
                np.array(self.cfg.resort_list)]

        # now calculate transformation matrix
        self.t_def_ct = kabsch(
            self.marker_pos_ct, self.marker_pos_def)

        # and its inverse
        self.t_ct_def = np.linalg.inv(self.t_def_ct)

        # get the measurement
        if self.use_gym:
            self.measure_data = np.array(data)
        else:
            self.measure_data = csv_test_load(df, cfg.csv_tracker_name)

    def init_tendon_points(self, tendon_dict: dict):
        self.tendon_points = {}
        for tendon_name in self.tendon_names:
            self.tendon_points[tendon_name] = tendon_dict[tendon_name]

    def read_markerdata(self, path, df):
        """read the data from the csv file"""

        if path == './':
            coordinates = get_def_tracker_from_csv(
                df, self.cfg.csv_tracker_name)

        else:
            with open(path, 'r') as file:
                reader = csv.reader(file)
                for _ in range(6):  # Skip the first six rows
                    next(reader)
                coordinates = [[float(row[2]), float(
                    row[3]), float(row[4])] for row in reader]

        return coordinates

    def update_tmat(self, t_index):
        rel_line = self.measure_data[t_index, :]
        qx, qy, qz, qw, x, y, z = rel_line

        # only update tmat if quaternion exists!
        if not np.isnan(qx):
            quat = Quaternion(qx=qx, qy=qy, qz=qz, qw=qw)

            if quat.w < 0:
                quat *= -1

            v = [x, y, z]

            t_mat = np.eye(4)
            t_mat[:3, :3] = quat.rotation_matrix
            t_mat[:3, 3] = v

            self.t_opt_def = t_mat
            self.t_def_opt = np.linalg.inv(t_mat)
            self.tmat_t = self.t_opt_def @ self.t_def_ct

            # update axis points:
            self.update_axis()

        return self.get_quaternion_pos()

    def update_axis(self):
        self.new_axis = []
        for pointx4 in self.axisx4:
            new_pointx4 = self.tmat_t @ pointx4
            self.new_axis.append(new_pointx4[:3])

    def read_ctdata(self):
        """read the data of the marker points from the ct scan"""
        self.marker_pos_ct = read_pointdata(self.ct_path)

    def read_axis_data(self):
        """read data of the axis"""
        self.axis_pos_ct = read_pointdata(self.axis_path)
        self.axisx4 = []

        for point in self.axis_pos_ct:
            pointx4 = np.ones(4)
            pointx4[:3] = point
            self.axisx4.append(pointx4)

    def get_quaternion_pos(self):
        quat = (Quaternion._from_matrix(
            self.tmat_t[:3, :3]) * self.q_offset).elements
        vec = self.tmat_t[:3, 3] * self.scale + self.off_xyz
        return quat, vec, self.tmat_t


def get_qaut_vec(tmat_t, off_xyz, q_offset: Quaternion, scale=0.001):
    quat = (Quaternion._from_matrix(tmat_t[:3, :3]) * q_offset).elements
    vec = tmat_t[:3, 3] * scale + off_xyz
    return quat, vec


def display_video(frames, videoname, framerate=120, dpi=600):
    height, width, _ = frames[0].shape
    fig, ax = plt.subplots(1, 1, figsize=(width / dpi, height / dpi), dpi=dpi)
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_position([0, 0, 1, 1])
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_data(frame)
        return [im]

    interval = 1000 / framerate
    anim = animation.FuncAnimation(fig, update, frames=frames,
                                   interval=interval, blit=True, repeat=False)
    anim.save(videoname, dpi=dpi, writer='ffmpeg')


def find_between(s: str, first: str, last: str):
    """helper for string preformatting"""
    try:
        start = s.index(first) + len(first)
        start_pos = s.index(first)
        end = s.index(last, start)
        return s[start:end].replace('"', ''), start_pos, end
    except ValueError:
        return "", "", ""


def recursive_loading(xml_path, path_ext='../', st_s='<include file=', end_s='/>', template_mode=False, model_path=''):
    """recursively load subfiles"""

    with open(f'{model_path}/{xml_path}', "r") as stream:
        xml_string = stream.read()

    xml_string = xml_string.replace("./", path_ext)
    xml_string = xml_string.replace("../", path_ext)

    extra_file, start_p, end_p = find_between(xml_string, st_s, end_s)

    if extra_file:

        if template_mode:
            filename = extra_file.split('.')[0]
            extra_file = f'{filename}_template.xml'

        extra_string = recursive_loading(
            f'{extra_file}', path_ext=path_ext, model_path=model_path)

        spos = extra_string.index('<mujoco model=')
        end = extra_string.index('>', spos)
        extra_string = extra_string[:spos] + extra_string[end:]
        extra_string = extra_string.replace('</mujoco>', '')

        xml_string = xml_string[:start_p] + extra_string + xml_string[end_p:]

    return xml_string


class Joint():
    def __init__(self, cfg: JointInfo):
        self.path = cfg.path
        self.name = cfg.name
        self.points = read_pointdata(self.path)
        self.calc_midpoint_and_axis()

    def calc_midpoint_and_axis(self):
        assert len(
            self.points) == 2, "The points list must contain exactly two points."
        self.p1 = np.array(self.points[0])
        self.p2 = np.array(self.points[1])

        # Calculate the midpoint and the axis
        self.midpoint = 0.5 * (self.p1 + self.p2)
        axis = self.p2 - self.p1
        self.axis = axis / np.linalg.norm(axis)  # normalize

        # Convert to homogeneous coordinates
        self.p1_hom = np.append(self.p1, 1)
        self.p2_hom = np.append(self.p2, 1)
        self.midpoint_hom = np.append(self.midpoint, 1)
        # append 0 since axis is a direction vector
        self.axis_hom = np.append(self.axis, 0)

    def update(self, root, tmat, scale=0.001):
        # Transform the points
        p1_transformed = tmat @ self.p1_hom
        p2_transformed = tmat @ self.p2_hom
        midpoint_transformed = tmat @ self.midpoint_hom
        axis_transformed = tmat @ self.axis_hom

        self.midpoint_transformed = (
            midpoint_transformed[:3] / midpoint_transformed[3]) * scale
        self.axis_transformed = axis_transformed[:3]

        p1, p2 = p1_transformed[:3] * scale, p2_transformed[:3] * scale
        dist = np.linalg.norm(p2-p1)
        # update mujoco
        update_all_joint(
            root, self.name, self.midpoint_transformed, self.axis_transformed)
        update_all_wrap(root, f'{self.name}_wrap', p1, p2, 0.3*dist)


class Joints():

    def __init__(self, cfg: AxisInfos):
        self.cfg = cfg
        self.dau_cmc = Joint(cfg.dau_cmc)
        self.dau_mcp = Joint(cfg.dau_mcp)
        self.dau_pip = Joint(cfg.dau_pip)
        self.dau_dip = Joint(cfg.dau_dip)
        self.zf_mcp = Joint(cfg.zf_mcp)
        self.zf_pip = Joint(cfg.zf_pip)
        self.zf_dip = Joint(cfg.zf_dip)


class Hand:

    def __init__(self, cfg: HandSettings):
        """init all configs, Markers, Trackers and static point lists"""
        # manage configs
        self.cfg = cfg
        self.model_path = cfg.model_path
        self.mujoco_file = cfg.mujoco_file
        self.scene_file = cfg.scene_file
        self.use_rel = cfg.create_relative
        self.mujoco_path = f'{self.model_path}/{self.mujoco_file}'
        self.tree = ET.parse(self.mujoco_path)
        self.root = self.tree.getroot()
        self.measure_path = cfg.optitrack_file
        self.threshold = cfg.gen.marker_filt_threshold
        self.off_xyz = cfg.gen.off_xyz
        self.off_wxyz = cfg.gen.off_wxyz
        self.scale = cfg.gen.scale
        self.use_gym = cfg.use_gym
        self.videoname = cfg.gen.videoname
        self.number_opt_iters = cfg.gen.number_opt_iters
        self.use_thumb_marker = cfg.gen.use_thumb_marker
        self.df = pd.read_csv(self.measure_path, header=2, low_memory=False)
        self.fps = cfg.gen.fps
        self.start_pos = cfg.gen.start_pos
        self.end_pos = cfg.gen.end_pos
        self.q_normal = Quaternion(1, 0, 0, 0)
        self.data = TestEvaluator(
            cfg.motor_assign, cfg.body_assign, name=cfg.gym_file, path='', dt=cfg.gen.dt, calib_path=cfg.gen.calib_path)
        self.joints = Joints(cfg.axis_infos)
        self.trafos = {}
        self.trafos_rel = {}
        self.init_tendons()

        # thumb config
        self.init_thumb()

        # index finger config
        self.init_index()

        # get static points from ct
        self.get_thumb_pip_ct()
        self.get_index_1_ct()
        self.get_index_1_ct_v2()
        self.get_index_2_ct()
        self.marker_estimate_static()

        _, _, self.t_mcp_const = self.tracker_zf_mcp.update_tmat(0)
        self.inv_t_mcp_const = np.linalg.inv(self.t_mcp_const)

        self.bodies = [
            self.tracker_dau_dip,
            self.marker_dau_pip,
            self.tracker_dau_mcp,
            self.tracker_zf_dip,
            self.marker_zf_mid1,
            self.marker_zf_mid2,
            self.tracker_zf_mcp,
        ]

        self.add_tendon_paths()

    def init_tendons(self):
        self.tendon_dict = load_all_tendons_to_dict(self.cfg.tendon_path)

    def init_thumb(self):
        self.tracker_dau_dip = Tracker(
            self.cfg.dau_dip, self.df, self.data.daumen_dp, self.off_xyz,
            self.off_wxyz, self.scale, use_gym=self.use_gym, tendon_dict=self.tendon_dict)
        self.marker_dau_pip = Marker(
            self.measure_path, self.threshold, self.cfg.dau_pip, tendon_dict=self.tendon_dict)
        self.tracker_dau_mcp = Tracker(
            self.cfg.dau_mcp, self.df, self.data.daumen_mc, self.off_xyz,
            self.off_wxyz, self.scale, use_gym=self.use_gym, tendon_dict=self.tendon_dict)

    def init_index(self):
        self.tracker_zf_dip = Tracker(
            self.cfg.zf_dip, self.df, self.data.zf_dp, self.off_xyz,
            self.off_wxyz, self.scale, use_gym=self.use_gym, tendon_dict=self.tendon_dict)
        self.marker_zf_mid1 = Marker(
            self.measure_path, self.threshold, self.cfg.zf_mid1, tendon_dict=self.tendon_dict)
        self.marker_zf_mid2 = Marker(
            self.measure_path, self.threshold, self.cfg.zf_mid2, tendon_dict=self.tendon_dict)
        self.tracker_zf_mcp = Tracker(
            self.cfg.zf_mcp, self.df, self.data.zf_pp, self.off_xyz,
            self.off_wxyz, self.scale, use_gym=self.use_gym, tendon_dict=self.tendon_dict)

    def get_thumb_pip_ct(self):
        """static point list defining the position of the thumb middle"""
        ct_pos = []
        ct_pos.extend(self.marker_dau_pip.ct_axis_distal)
        ct_pos.extend(self.marker_dau_pip.ct_axis_proximal)
        ct_pos.extend(
            self.marker_dau_pip.marker_ct) if self.use_thumb_marker else None
        self.ct_dau_pip_pos = ct_pos

    def get_index_1_ct(self):
        """
        static point list defining the position of the index middle 1 v1
        -> dip axis + marker point
        """
        ct_pos = []
        ct_pos.extend(self.marker_zf_mid1.ct_axis_distal)
        ct_pos.extend(self.marker_zf_mid1.marker_ct)
        self.ct_zf_mid1_pos = ct_pos

    def get_index_1_ct_v2(self):
        """
        static point list defining the position of the index middle 1 v2
        -> dip axis + pip axis
        """
        ct_pos = []
        ct_pos.extend(self.marker_zf_mid1.ct_axis_distal)
        ct_pos.extend(self.marker_zf_mid1.ct_axis_proximal)
        self.ct_zf_mid1_pos2 = ct_pos

    def get_index_2_ct(self):
        """static point list defining the position of the index middle 2"""
        ct_pos = []
        ct_pos.extend(self.marker_zf_mid2.ct_axis_distal)
        ct_pos.extend(self.marker_zf_mid2.ct_axis_proximal)

        # also store x4 values
        self.axisx4_zf1 = []
        for point in self.marker_zf_mid2.ct_axis_distal:
            pointx4 = np.ones(4)
            pointx4[:3] = point
            self.axisx4_zf1.append(pointx4)

        self.ct_zf_mid2_pos = ct_pos

    def update_axis_zf1(self, tmat_t):
        """update the axis of the index finger pip joint"""
        self.new_axis_zf1 = []
        for pointx4 in self.axisx4_zf1:
            new_pointx4 = tmat_t @ pointx4
            self.new_axis_zf1.append(new_pointx4[:3])

    def update_tendon_points(self):
        """use the trafos to update the points"""

        for body in self.bodies:
            bodyname = body.mj_name
            trafo = self.trafos_rel[bodyname]
            points = body.tendon_points

            for point_name in points:
                pointx4 = np.ones(4)
                pointx4[:3] = points[point_name]
                new_point = (trafo @ pointx4)[:3] * self.scale
                recursive_site(self.root, bodyname, point_name,
                               new_point, 0.5*self.scale)

    def add_tendon_paths(self, stiffness=1000, damping=200):
        """
        for this we need "self.tendon_dict" -> the logic is that each tendon runs run NAME-1 to NAME-n.
        the names can be obtained from tendon_dict.keys()
        but also consider if the site has a joint, we need to add the tendon wrap as well!
        """

        # pepare tendon
        if self.root.find(TENDON) != None:
            self.root.remove(self.root.find(TENDON))
        self.root.makeelement(TENDON, {})
        ET.SubElement(self.root, TENDON, {})
        tendon_root = self.root.find(TENDON)

        keylist = list(self.tendon_dict.keys())
        tendon_dict = {}
        for key in keylist:
            tendon_name = key.split('-')[0]
            if tendon_name in list(tendon_dict.keys()):
                tendon_dict[tendon_name].append(key)
            else:
                tendon_dict[tendon_name] = [key]

        tendon_root = self.root.find(TENDON)

        for tendon in tendon_dict:

            # use local attribute
            loc_attrib = {
                'name': tendon,
                'stiffness': f"{stiffness * self.scale * 1000}",
                'damping': f"{damping}",
                'width': f'{self.scale * 0.5}',
                'rgba': "0.9 0.2 0.2 0.3",
            }

            tendon_root.makeelement('spatial', loc_attrib)
            ET.SubElement(tendon_root, 'spatial', loc_attrib)

        for tendon in tendon_dict:
            tendon_points = sorted(tendon_dict[tendon])
            for pos_tendon in tendon_root:
                if pos_tendon.attrib['name'] == tendon:
                    for tendon_point in tendon_points:
                        pos_tendon.makeelement('site', {'site': tendon_point})
                        ET.SubElement(pos_tendon, 'site', {
                                      'site': tendon_point})
                        self.add_joint_wrap(tendon_point, pos_tendon)

    def add_joint_wrap(self, tendon_point, pos_tendon):
        """check the position of the current tendon point -> if it has a cylinder add the cyl as well
        tendon_point -> name of tendon point, pos_tendon -> name of the actual overall tendon
        """
        # 1. find the body which contains the site
        parent = find_element_parent(self.root, None, tendon_point)

        # check if there is a zyl
        rel_wrap = None
        for child in parent:
            if 'name' in child.attrib and '_wrap' in child.attrib['name']:
                rel_wrap = child.attrib['name']

        # if there is, include the cyl to the list
        if rel_wrap is not None:
            ET.SubElement(pos_tendon, 'geom', {'geom': rel_wrap})

    def update_thumb(self, t_index, t_mcp_index):
        """construct the required calculations for the thumb and update"""
        t_mcp_index_inv = np.linalg.inv(t_mcp_index)
        # first get the positions and quaternions of the trackers for the thumb
        _, _, t_mcp = self.tracker_dau_mcp.update_tmat(t_index)
        tmat_mcp_relative = t_mcp_index_inv @ t_mcp
        quat, vec = get_qaut_vec(
            tmat_mcp_relative, [0, 0, 0], self.q_normal, scale=self.scale)
        update_all_pos_quat(self.root, self.tracker_dau_mcp.mj_name, vec, quat)
        self.joints.dau_cmc.update(
            self.root, tmat_mcp_relative, scale=self.scale)
        self.joints.dau_mcp.update(
            self.root, tmat_mcp_relative, scale=self.scale)
        self.tmat_dau_mcp = t_mcp @ t_mcp_index_inv @ self.t_mcp_const
        self.trafos[self.tracker_dau_mcp.mj_name] = tmat_mcp_relative if self.use_rel else self.tmat_dau_mcp

        _, _, t_dip = self.tracker_dau_dip.update_tmat(t_index)

        # now we need to construct the required list of points in ct and opt sys:
        opt_pos = []
        opt_pos.extend(self.tracker_dau_dip.new_axis)
        opt_pos.extend(self.tracker_dau_mcp.new_axis)

        if self.use_thumb_marker:
            # now get the current position of the thumb marker:
            pos_marker = self.marker_dau_pip.get_position(t_index)
            opt_pos.extend([pos_marker])

        # having the list of ct and opt_pos, we now can calculate kabsch:
        tmat_marker = kabsch(self.ct_dau_pip_pos, opt_pos)
        tmat_m = np.linalg.inv(t_mcp) @ tmat_marker
        quat1, vec1 = get_qaut_vec(
            tmat_m, [0, 0, 0], self.q_normal, scale=self.scale)
        update_all_pos_quat(
            self.root, self.marker_dau_pip.mj_name, vec1, quat1)
        self.joints.dau_pip.update(self.root, tmat_m, scale=self.scale)
        self.tmat_dau_pip = tmat_marker @ t_mcp_index_inv @ self.t_mcp_const
        self.trafos[self.marker_dau_pip.mj_name] = tmat_m if self.use_rel else self.tmat_dau_pip

        t_dip_relative = np.linalg.inv(tmat_marker) @ t_dip
        quat2, vec2 = get_qaut_vec(
            t_dip_relative, [0, 0, 0], self.q_normal, scale=self.scale)
        update_all_pos_quat(
            self.root, self.tracker_dau_dip.mj_name, vec2, quat2)
        self.joints.dau_dip.update(self.root, t_dip_relative, scale=self.scale)
        self.tmat_dau_dip = t_dip @ t_mcp_index_inv @ self.t_mcp_const
        self.trafos[self.tracker_dau_dip.mj_name] = t_dip_relative if self.use_rel else self.tmat_dau_dip

        self.trafos_rel[self.tracker_dau_dip.mj_name] = t_dip_relative
        self.trafos_rel[self.marker_dau_pip.mj_name] = tmat_m
        self.trafos_rel[self.tracker_dau_mcp.mj_name] = tmat_m

    def marker_estimate_static(self):
        marker_pos = self.marker_zf_mid1.marker_ct[0]
        self.marker_zf1_pointx4 = np.ones(4)
        self.marker_zf1_pointx4[:3] = marker_pos

    def calculate_marker_estimate(self, tmat_t):
        marker_estimate = tmat_t @ self.marker_zf1_pointx4
        self.marker_estimate = marker_estimate[:3]

    def update_index(self, t_index):
        """construct the required calculations for the index finger and update"""
        _, _, t_dip = self.tracker_zf_dip.update_tmat(
            t_index)  # from ct to opt
        _, _, t_mcp = self.tracker_zf_mcp.update_tmat(t_index)
        t_mcp_inv = np.linalg.inv(t_mcp)
        self.tmat_zf_mcp = t_mcp @ t_mcp_inv @ self.t_mcp_const
        self.trafos[self.tracker_zf_mcp.mj_name] = self.tmat_zf_mcp
        quat_mcp, vec_mcp = get_qaut_vec(
            self.tmat_zf_mcp, self.off_xyz, self.q_normal, scale=self.scale)

        # now we need to construct the required list of points in ct and opt sys:
        opt_pos = []
        opt_pos.extend(self.tracker_zf_dip.new_axis)

        self.calculate_marker_estimate(t_dip)
        opt_pos.extend([self.marker_estimate])

        # having the list of ct and opt_pos, we now can calculate kabsch:
        tmat_marker1 = kabsch(self.ct_zf_mid1_pos, opt_pos)

        # idea: iterate to increase acc
        for _ in range(self.number_opt_iters):

            # now we can update the axis
            self.update_axis_zf1(tmat_marker1)

            # finally we need the last position and therefore calculate the axis points
            opt_pos = []
            opt_pos.extend(self.new_axis_zf1)
            opt_pos.extend(self.tracker_zf_mcp.new_axis)
            tmat_marker2 = kabsch(self.ct_zf_mid2_pos, opt_pos)

            self.update_axis_zf1(tmat_marker2)

            # reiterate on zf pip1 -> hopefully increased accuracy
            opt_pos = []
            opt_pos.extend(self.tracker_zf_dip.new_axis)
            opt_pos.extend(self.new_axis_zf1)
            tmat_marker1 = kabsch(self.ct_zf_mid1_pos2, opt_pos)

        update_all_pos_quat(
            self.root, self.tracker_zf_mcp.mj_name, vec_mcp, quat_mcp)

        tmat_m2 = np.linalg.inv(t_mcp) @ tmat_marker2
        quat_m2, vec_m2 = get_qaut_vec(
            tmat_m2, [0, 0, 0], self.q_normal, scale=self.scale)
        update_all_pos_quat(
            self.root, self.marker_zf_mid2.mj_name, vec_m2, quat_m2)

        self.joints.zf_mcp.update(self.root, tmat_m2, scale=self.scale)
        self.tmat_zf_pip2 = tmat_marker2 @ t_mcp_inv @ self.t_mcp_const
        self.trafos[self.marker_zf_mid2.mj_name] = tmat_m2 if self.use_rel else self.tmat_zf_pip2

        tmat_m1 = np.linalg.inv(tmat_marker2) @ tmat_marker1
        quat_m1, vec_m1 = get_qaut_vec(
            tmat_m1, [0, 0, 0], self.q_normal, scale=self.scale)
        update_all_pos_quat(
            self.root, self.marker_zf_mid1.mj_name, vec_m1, quat_m1)
        self.joints.zf_pip.update(self.root, tmat_m1, scale=self.scale)
        self.tmat_zf_pip1 = tmat_marker1 @ t_mcp_inv @ self.t_mcp_const
        self.trafos[self.marker_zf_mid1.mj_name] = tmat_m1 if self.use_rel else self.tmat_zf_pip1

        t_dip_relative = np.linalg.inv(tmat_marker1) @ t_dip
        quat_dip, vec_dip = get_qaut_vec(
            t_dip_relative, [0, 0, 0], self.q_normal, scale=self.scale)
        update_all_pos_quat(
            self.root, self.tracker_zf_dip.mj_name, vec_dip, quat_dip)
        self.joints.zf_dip.update(self.root, t_dip_relative, scale=self.scale)
        self.tmat_zf_dip = t_dip @ t_mcp_inv @ self.t_mcp_const
        self.trafos[self.tracker_zf_dip.mj_name] = t_dip_relative if self.use_rel else self.tmat_zf_dip

        zero_t = np.eye(4)
        zero_t[:3, 3] = -np.array(self.off_xyz)
        self.trafos_rel[self.tracker_zf_dip.mj_name] = t_dip_relative
        self.trafos_rel[self.marker_zf_mid1.mj_name] = t_dip_relative
        self.trafos_rel[self.marker_zf_mid2.mj_name] = zero_t
        self.trafos_rel[self.tracker_zf_mcp.mj_name] = zero_t

        return t_mcp

    def update(self, t_index):
        """central update function handling index and thumb"""
        t_mcp = self.update_index(t_index)
        self.update_thumb(t_index, t_mcp)
        if t_index == 0:
            self.update_tendon_points()
        self.tree.write(self.mujoco_path)

    def load(self):
        """load model from path"""
        self.model = recursive_loading(
            self.scene_file, template_mode=False, path_ext='./', model_path=self.model_path)
        self.physics = mujoco.MjModel.from_xml_string(self.model)
        data = mujoco.MjData(self.physics)
        renderer = mujoco.Renderer(self.physics, 1024, 1024)

        scene_option = mujoco.MjvOption()
        # scene_option.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = True

        mujoco.mj_step(self.physics, data)
        renderer.update_scene(
            data, scene_option=scene_option, camera="camera0")
        pixels = renderer.render()
        image = Image.fromarray((pixels).astype(np.uint8))
        return image

    def make_video(self):
        """make a whole video based on the settings from config"""
        frames = []

        for index in tqdm(range(self.start_pos, self.end_pos)):
            self.update(index)
            self.load()
            frames.append(self.get_pixels())

        display_video(frames, framerate=self.fps, videoname=self.videoname)

    def return_tmats(self):
        tmats = [
            self.tmat_zf_pip2,
            self.tmat_zf_pip1,
            self.tmat_zf_dip,
            self.tmat_dau_mcp,
            self.tmat_dau_pip,
            self.tmat_dau_dip,
        ]
        return np.array(tmats)

    def make_dataset_dict(self):
        dlen = len(range(self.start_pos, self.end_pos))
        self.update(self.start_pos)

        dset_numpy = {}  # For saving as a NumPy file
        dset_json = {}   # For saving as a JSON file
        num_tmats = len(self.return_tmats())
        dataset = np.zeros([dlen, num_tmats, 4, 4], dtype=np.float32)

        # Process transformation matrices
        for loc_trafo in self.trafos:
            dset_numpy[loc_trafo] = np.zeros([dlen, 4, 4], dtype=np.float32)

        for index in tqdm(range(self.start_pos, self.end_pos)):
            self.update(index)
            for loc_trafo in self.trafos:
                tmats = self.return_tmats()
                tmats[:, :3, 3] = tmats[:, :3, 3] * self.scale
                # tmats[:, :3, 3] = tmats[:, :3, 3] * self.scale
                dataset[index, :, :, :] = tmats

                loc_mat = np.array(self.trafos[loc_trafo], dtype=np.float32)
                loc_mat[:3, 3] = loc_mat[:3, 3] * self.scale + self.off_xyz
                dset_numpy[loc_trafo][index, :, :] = loc_mat

        # save the flat dataset
        np.save(self.cfg.dataset_name, dataset)

        self.dset_numpy = dset_numpy.copy()

        # Save as NumPy file
        np.save(self.cfg.dataset_dict_name, dset_numpy)

        # Process force signals
        for field_name in self.cfg.motor_assign:
            force_arr = self.data.force_data[self.cfg.motor_assign[field_name]]
            dset_numpy[field_name] = force_arr[self.start_pos: self.end_pos]

        dset_numpy['f_norm'] = self.data.f_norm
        dset_numpy['time'] = self.data.new_time - self.data.new_time[0]

        # Convert numpy arrays to lists for the JSON version
        for key in dset_numpy:
            dset_json[key] = dset_numpy[key].tolist()

        # Save as JSON file
        json_name = self.cfg.dataset_dict_name.replace('.npy', '')
        with open(json_name + '.json', 'w') as file:
            json.dump(dset_json, file, indent=4, sort_keys=True)

        # also prepare the simple xml and the init params yaml for training of the kinematics later
        copy_xml_values(f'{self.cfg.model_path}/{self.cfg.mujoco_file}',
                        f'{self.cfg.model_path}/{self.cfg.mujoco_copy_file}', tag='body', subtags=['pos', 'quat'])
        copy_xml_to_yaml(f'{self.cfg.model_path}/{self.cfg.mujoco_file}',
                         f'{self.cfg.model_path}/{self.cfg.joint_store_file}', tag='joint', subtags=['pos', 'axis'])

        self.plot_trafos()
        self.plot_all_signals()
        create_control_array(
            f'{self.cfg.model_path}/{self.cfg.mujoco_file}', json_name + '.json')

    def plot_all_signals(self):
        # Number of keys in motor_assign
        keylen = len(self.cfg.motor_assign)

        # Create a figure with shared x-axis
        fig, axes = plt.subplots(
            keylen + 1, 1, figsize=(10, keylen*2 + 2), sharex=True)
        fig.suptitle('All Signals', fontsize=16)

        # If there's only one subplot, axes is not a list, so we wrap it in a list
        if keylen == 1:
            axes = [axes]

        for idx, (field_name, motor) in enumerate(self.cfg.motor_assign.items()):
            force = self.data.force_data[motor]
            axes[idx].plot(self.data.new_time[self.start_pos: self.end_pos],
                           force[self.start_pos: self.end_pos])
            axes[idx].set_ylabel(f'Force ({field_name})')
            axes[idx].grid(True)

        axes[-1].plot(self.data.new_time[self.start_pos: self.end_pos],
                      self.data.f_norm[self.start_pos: self.end_pos])
        axes[-1].set_ylabel(f'FT Sensor')
        axes[-1].grid(True)

        # Labeling the x-axis on the last subplot
        axes[-1].set_xlabel('Time')

        # Adjust layout for better fit
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Show the plot
        plt.savefig('./signal_infos.png')

    def plot_trafos(self):
        num_objects = len(self.dset_numpy)
        fig, axs = plt.subplots(num_objects, 2, figsize=(15, 5 * num_objects))
        fig.suptitle('Quaternion and Position Vectors over Time', fontsize=16)
        time = self.data.new_time[self.start_pos: self.end_pos]

        for idx, (obj_name, matrices) in enumerate(self.dset_numpy.items()):
            # Preallocate arrays for positions and quaternions
            positions = np.empty((self.end_pos - self.start_pos, 3))
            quaternions = np.empty((self.end_pos - self.start_pos, 4))

            for i, matrix in enumerate(matrices[self.start_pos:self.end_pos]):
                # Extracting position vector
                positions[i, :] = matrix[:3, 3]

                # Converting rotation matrix to quaternion
                rot = R.from_matrix(matrix[:3, :3])
                quaternions[i, :] = rot.as_quat()  # [x, y, z, w]

            # Plot position vectors
            axs[idx, 0].plot(time, positions)
            axs[idx, 0].set_title(f'{obj_name} Position Vectors')
            axs[idx, 0].set_xlabel('Time')
            axs[idx, 0].set_ylabel('Position')
            axs[idx, 0].legend(['X', 'Y', 'Z'])
            axs[idx, 0].grid(True)

            # Plot quaternion components
            axs[idx, 1].plot(time, quaternions)
            axs[idx, 1].set_title(f'{obj_name} Quaternion')
            axs[idx, 1].set_xlabel('Time')
            axs[idx, 1].set_ylabel('Quaternion Component')
            axs[idx, 1].legend(['X', 'Y', 'Z', 'W'])
            axs[idx, 1].grid(True)

        plt.tight_layout()
        plt.savefig('./trafos.png')

    def run(self):
        if self.cfg.create_dataset:
            self.make_dataset_dict()

# %%


@hydra.main(version_base=None, config_path="../config", config_name="config_hand_2021")
def main_sort(cfg: HandSettings):
    # Define the possible configurations for each resort_list
    possible_configs = list(permutations([0, 1, 2]))

    # Iterate over all possible configurations
    for i, config_zf_mcp in enumerate(possible_configs):
        for j, config_zf_dip in enumerate(possible_configs):
            # Update the resort_list for each tracker
            cfg.zf_dip.resort_list = config_zf_dip
            cfg.zf_mcp.resort_list = config_zf_mcp

            # Update the video name to capture the current configuration
            cfg.gen.videoname = f"./videos/calib/config_zf_mcp_{i}_config_zf_dip_{j}.mp4"

            # Run the code with the updated configuration
            hand = Hand(cfg)
            hand.run()


@hydra.main(version_base=None, config_path="../config", config_name="config_hand_2023")
def main(cfg: HandSettings):
    hand = Hand(cfg)
    hand.run()

    if cfg.create_video:
        hand_safety(cfg)


def hand_safety(cfg: HandSettings, lim_zf=5000, lim_dau=5000):
    hand_1 = Hand(cfg)
    hand_2 = Hand(cfg)

    frames = []
    hand_2.update(hand_1.start_pos)
    old_vec_zf = np.array(hand_2.tmat_zf_dip)
    old_vec_dau = np.array(hand_2.tmat_dau_dip)

    err_dau_list = []
    err_zf_list = []

    for index in tqdm(range(hand_1.start_pos, hand_2.end_pos)):

        hand_1.update(index)
        hand_1.load()

        new_vec_zf = np.array(hand_1.tmat_zf_dip)
        new_vec_dau = np.array(hand_2.tmat_dau_dip)

        err_zf = (np.square(old_vec_zf - new_vec_zf)).mean(axis=None)
        err_dau = (np.square(old_vec_dau - new_vec_dau)).mean(axis=None)

        print()
        print(np.round(err_zf, decimals=4))
        print(np.round(err_dau, decimals=4))
        err_dau_list.append(err_dau)
        err_zf_list.append(err_zf)

        if err_zf < lim_zf and err_dau < lim_dau:
            old_vec_zf = new_vec_zf
            old_vec_dau = new_vec_dau
            hand_2.update(index)

        frames.append(hand_2.load())

    image_arrays = [np.array(img) for img in frames]
    with imageio.get_writer(hand_1.videoname, mode='I', fps=hand_1.fps) as writer:
        for image_array in image_arrays:
            writer.append_data(image_array)

    fig = plt.figure()
    plt.plot(err_zf_list)
    plt.plot(err_dau_list)
    fig.savefig('./err_plot.png')
    plt.close()


if __name__ == '__main__':
    main()
# %%
