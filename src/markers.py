# %%
import os
from stl import mesh
from config import MarkerSettings
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import yaml


def create_control_array(xml_path, json_path):
    # Read the XML file and find the order of actuators
    tree = ET.parse(xml_path)
    root = tree.getroot()
    actuators = [actuator.get('name') for actuator in root.find('actuator')]

    # Read the JSON file with actions
    with open(json_path, 'r') as json_file:
        json_data = json.load(json_file)

    # Check if the JSON file contains all the actuators
    if not all(actuator in json_data for actuator in actuators):
        raise ValueError(
            "JSON file does not contain all the actuators defined in the XML file.")

    # Determine the number of timesteps based on the first actuator's data
    num_timesteps = len(json_data[actuators[0]])

    # Create the control array
    control_array = [[json_data[actuator][timestep]
                      for actuator in actuators] for timestep in range(num_timesteps)]

    # Add the control array to the JSON data
    json_data['ctrl'] = control_array

    # Save the updated JSON data back to the file
    with open(json_path, 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

    return json_data


def create_modified_xml_and_yaml(tags_list, load_xml, save_xml, yaml_dir):
    """
    Prepare the XML and YAML for dynamic training.
    """
    # Parse the XML file
    tree = ET.parse(load_xml)
    root = tree.getroot()

    # Initialize a dictionary for YAML content
    yaml_content = {}

    # Iterate over elements in the XML
    for elem in root.iter():
        # Handle 'site' elements separately
        if elem.tag == 'site':
            name = elem.get('name')
            pos_str = elem.get('pos')

            # Ensure pos_str is not None before processing
            if pos_str:
                pos_array = np.array(pos_str.split(), dtype=float)
                x, y, z = pos_array

                # Store in the YAML content
                yaml_content[name] = {'x': str(x), 'y': str(y), 'z': str(z)}

                # Update the 'pos' attribute in XML with placeholder
                placeholder = "{{" f"{elem.get('name')}.x" + "}} " + \
                    "{{" f"{elem.get('name')}.y" + "}} " + \
                    "{{" f"{elem.get('name')}.z" + "}}"
                elem.set('pos', placeholder)

        # Process other tags
        for tag in tags_list:
            if tag in elem.attrib:
                # Construct the placeholder and original value
                placeholder = "{{" + f"{elem.get('name')}.{tag}" + "}}"
                original_value = elem.get(tag)

                # Update the element attribute in XML
                elem.set(tag, placeholder)

                # Update the YAML content
                if elem.get('name') not in yaml_content:
                    yaml_content[elem.get('name')] = {}
                yaml_content[elem.get('name')][tag] = str(original_value)

    # Write the modified XML
    tree.write(save_xml)

    # Write the YAML file
    with open(yaml_dir, 'w') as yaml_file:
        yaml.dump(yaml_content, yaml_file, default_flow_style=False)


def copy_xml_values(source_file, target_file, tag, subtags):
    """
    Copies specified attributes from one XML file to another.

    :param source_file: Path to the source XML file.
    :param target_file: Path to the target XML file.
    :param tag: The tag to be considered for copying.
    :param subtags: List of subtags whose values need to be copied.
    """
    # Load source XML
    source_tree = ET.parse(source_file)
    source_root = source_tree.getroot()

    # Load target XML
    target_tree = ET.parse(target_file)
    target_root = target_tree.getroot()

    # Iterate over each specified tag in source XML
    for element in source_root.findall(f'.//{tag}'):
        name = element.get('name')

        # Find the corresponding tag in target XML and update
        for target_element in target_root.findall(f".//{tag}[@name='{name}']"):
            for subtag in subtags:
                value = element.get(subtag)
                if value:
                    target_element.set(subtag, value)

    # Save the changes to the target file
    target_tree.write(target_file)


def copy_xml_to_yaml(xml_file, yaml_file, tag, subtags):
    """
    Copies specified attributes from an XML file to a YAML file.

    :param xml_file: Path to the source XML file.
    :param yaml_file: Path to the target YAML file.
    :param tag: The tag to be considered for copying.
    :param subtags: List of subtags whose values need to be copied.
    """
    # Load and parse the XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Load the YAML file
    with open(yaml_file, 'r') as file:
        yaml_data = yaml.safe_load(file) or {}

    # Iterate over each specified tag in the XML file
    for element in root.findall(f'.//{tag}'):
        name = element.get('name')
        if name:
            yaml_data[name] = yaml_data.get(name, {})
            for subtag in subtags:
                value = element.get(subtag)
                if value:
                    yaml_data[name][subtag] = value

    # Write the updated data to the YAML file
    with open(yaml_file, 'w') as file:
        yaml.safe_dump(yaml_data, file, default_flow_style=False)


def csv_test_load(testrun_path, tracker_designation_motive):
    """This function is suppose to read in the trakcing data of a single tracker
    of a specified testrun from an motive .csv export."""
    df = pd.read_csv(testrun_path, header=2, low_memory=False)
    start_column = df.columns.get_loc(tracker_designation_motive)
    data = df.iloc[3:, start_column:start_column+8].applymap(float).values
    return data


def json_test_load(path='./Data/optitrack-20230130-234800.json', initial_id=''):
    """This function is suppose to read in the trakcing data of a single tracker"""
    with open(path) as f:
        df = pd.DataFrame(json.load(f))

    header = [
        f"{df[i][0]['name']}.{j}"
        for i in range(df.shape[1])
        for j in range(7)
        if 'name' in df[i][0]
    ]

    keys = ['qx', 'qy', 'qz', 'qw', 'x', 'y', 'z']
    xyz = np.array([
        [df[j][i][k] if k in df[j][i] else np.nan for j in range(
            len(header) // 7) for k in keys]
        for i in range(df.shape[0])
    ])

    df = pd.DataFrame(xyz, columns=header)

    if initial_id:
        idx = df.columns.get_loc(initial_id)
        df = df.iloc[:, idx:idx+7]

    return df


def marker_variable_id_linewise(testrun_path, initial_id=None, dtype="csv", d_max=30):
    """read marker data linewise and find potential new id if marker was lost and redefined"""

    initial_id = str(initial_id)

    if dtype == "json":
        print("rebuilding json to be formated as csv")
        df = json_test_load(testrun_path, initial_id)
    else:
        df = pd.read_csv(testrun_path, header=2, low_memory=False)

    idx = df.columns.get_loc(initial_id)
    start_line = np.where(pd.notna(df.iloc[:, idx]))[
        0][3] if np.isnan(float(df.iloc[3, idx])) else 3

    data = np.zeros((df.shape[0]-3, 3))
    data[0, :] = df.values[start_line, idx:idx+3]
    last_signal = data[0, :]

    for i in tqdm(range(start_line, data.shape[0])):
        values_to_add, min_dis = find_min_distance(i, df, idx, last_signal)
        if not np.isnan(values_to_add[0]):
            last_signal = values_to_add

            if min_dis >= d_max:
                data[i, :] = [np.nan, np.nan,
                              np.nan]
            else:
                data[i, :] = values_to_add

    return data


def find_min_distance(line, df, idx, last_signal):
    min_dis = np.inf
    values_to_add = [np.nan, np.nan, np.nan]

    for j in range(idx, len(df.columns), 3):
        values = df.values[line, j:j+3]
        values = [float(value) for value in values]
        if not np.isnan(values).any():
            current_dis = np.abs(np.linalg.norm(
                np.array(last_signal) - np.array(values)))
            if current_dis < min_dis:
                min_dis = current_dis
                values_to_add = values

    return values_to_add, min_dis


def filter_jumps(positions, threshold):
    """filter the occuring jumps defined by a thershold margin -> keep old position if jump occurs"""
    cur_pos = positions[0, :]
    filt_positions = positions.copy()

    for i, pos in enumerate(positions):

        cur_dis = np.linalg.norm(pos - cur_pos)

        if cur_dis < threshold:
            cur_pos = pos
            filt_positions[i, :] = cur_pos
        else:
            filt_positions[i, :] = [np.nan, np.nan, np.nan]

    return filt_positions


def interpolate_nan_2d(arr):
    """perform 2d interpolation for missing values"""
    for i in range(arr.shape[1]):
        valid_mask = ~np.isnan(arr[:, i])
        valid_indices = np.where(valid_mask)[0]
        invalid_indices = np.where(~valid_mask)[0]

        if len(valid_indices) == 0:
            continue  # Skip if there are no valid indices

        valid_values = arr[valid_mask, i]

        interpolated_values = np.interp(
            invalid_indices, valid_indices, valid_values)
        arr[invalid_indices, i] = interpolated_values

    return arr


def load_marker(test_path, marker_id, measurement_type, threshold=0.1, verbose=False):
    """perform the task to prepare marker data"""
    marker_data = marker_variable_id_linewise(
        test_path, marker_id, measurement_type, threshold)
    marker_data = interpolate_nan_2d(marker_data)
    marker_data = filter_jumps(marker_data, threshold=threshold)
    marker_data = interpolate_nan_2d(marker_data)

    if verbose:
        plt.figure()
        plt.plot(marker_data)
        plt.show()

    return marker_data


def get_path(filename: str) -> str:
    return os.path.dirname(filename)


def read_pointdata(path):
    """func for loading points from the slicer based json"""
    with open(path) as jsonfile:
        data = json.load(jsonfile)
    # extract point infos
    point_data = data['markups'][0]['controlPoints']
    return [point['position'] for point in point_data]


class Marker:

    def __init__(self, measure_path: str, threshold: float, cfg: MarkerSettings, tendon_dict: dict):
        """init a marker object"""
        self.cfg = cfg
        self.mj_name = cfg.mj_name
        self.name = cfg.name
        self.start_id = cfg.start_id
        self.ct_path_axis_distal = cfg.ct_path_axis_distal
        self.ct_path_axis_proximal = cfg.ct_path_axis_proximal
        self.ct_path_marker = cfg.ct_path_marker
        self.stl_path = cfg.stl_path
        self.your_mesh = mesh.Mesh.from_file(cfg.stl_path)
        self.threshold = threshold
        self.measure_path = measure_path
        self.tendon_names = cfg.point_list
        self.init_tendon_points(tendon_dict)

        if self.measure_path.endswith('.csv'):
            self.measure_type = 'csv'
        else:
            self.measure_type = 'json'

        self.path = get_path(self.measure_path)
        self.filename = f'{self.path}/{self.name}.npy'

        # if startid not found we have no tracking
        if len(self.start_id) <= 2:
            self.has_tracking = False
        else:
            self.has_tracking = True
            self.load_file_or_recalculate()

        self.load_ct_positions()

    def init_tendon_points(self, tendon_dict: dict):
        self.tendon_points = {}
        for tendon_name in self.tendon_names:
            self.tendon_points[tendon_name] = tendon_dict[tendon_name]

    def load_file_or_recalculate(self):
        """try to find the corresponding precalculated marker file"""
        if os.path.isfile(self.filename):
            self.load_positions()
        else:
            self.positions = load_marker(
                self.measure_path, self.start_id, self.measure_type, self.threshold)
            self.save_positions()
            print(len(self.positions))

    def save_positions(self):
        """Save the positions array to a file."""
        np.save(self.filename, self.positions)

    def load_positions(self):
        """Load the positions array from a file."""
        self.positions = np.load(self.filename)

    def load_ct_positions(self):
        """get all positions of the relevant points in ct"""
        self.points_ct = []
        self.ct_axis_distal = read_pointdata(self.ct_path_axis_distal)
        self.ct_axis_proximal = read_pointdata(self.ct_path_axis_proximal)

        if len(self.ct_path_marker) > 2:
            self.marker_ct = read_pointdata(self.ct_path_marker)

    def get_position(self, t_index):
        if self.has_tracking:
            position = self.positions[t_index, :]
            return np.array(position)
        return []


# %%
if __name__ == '__main__':
    tstart = 2000
    tend = 4000
    test_path = '../measurement/23_01_31/Take 2023-01-31 06.11.42 PM.csv'
    marker_trace = marker_variable_id_linewise(
        test_path, 'Unlabeled 2016', 'csv', 40)

# %%


# %%
