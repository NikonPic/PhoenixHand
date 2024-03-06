# %%
import os
from matplotlib import pyplot as plt
from pyquaternion import Quaternion
import json
from config import FingerAssignment, RigidBodyAssignment
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
import numpy as np


def interpolate_positions(original_time_vector, new_time_vector, rigid_b):
    # Interpolate each coordinate separately
    interpolated_x = np.interp(
        new_time_vector, original_time_vector, rigid_b['x'])
    interpolated_y = np.interp(
        new_time_vector, original_time_vector, rigid_b['y'])
    interpolated_z = np.interp(
        new_time_vector, original_time_vector, rigid_b['z'])

    # Combine the interpolated coordinates
    interpolated_positions = np.column_stack(
        (interpolated_x, interpolated_y, interpolated_z))

    return interpolated_positions


class TestEvaluator():

    def __init__(self, finger_a: FingerAssignment, body_a: RigidBodyAssignment, name='pincer_highscore.json', path='./', dt=0.01, calib_path='./'):

        # read the json
        self.name = f'{path}{name}'
        with open(self.name, encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        # extract the main vectors
        self.obs = data['observation']
        self.act = data['action']
        self.time = [t / 1000 for t in data['time']]

        self.body_a = body_a
        self.finger_a = finger_a
        self.generate_new_time_vector(dt=dt)
        self.assign_rigid_bodies()
        self.clear_quaternions()
        self.calibrate(calib_path, data)
        self.add_force_sensor(data)

    def add_force_sensor(self, data):
        fx = np.array(data['observation']['force_torques'][0]['fx'])
        fy = np.array(data['observation']['force_torques'][0]['fy'])
        fz = np.array(data['observation']['force_torques'][0]['fz'])

        self.fx = np.interp(self.new_time, self.time, fx)
        self.fy = np.interp(self.new_time, self.time, fy)
        self.fz = np.interp(self.new_time, self.time, fz)

        self.f_norm = np.sqrt(self.fx**2 + self.fy**2 + self.fz**2)

    def calibrate(self, calib_path, data):
        calib_dict = calibrate_force_sensor_data(calib_path, num_sensors=8)
        force_list = arrange_force_sensor_data(data)
        self.force_data = []

        scales = calib_dict['scale']
        offsets = calib_dict['offset']
        for index, (scale, offset) in enumerate(zip(scales, offsets)):
            # calibrate
            calibrated_data = scale * force_list[index] + offset
            # interpolate
            calibrated_interpolated_data = np.interp(
                self.new_time, self.time, calibrated_data)
            resc_calib_interp = (calibrated_interpolated_data / max(calibrated_interpolated_data)) * -1
            self.force_data.append(resc_calib_interp)

    def generate_new_time_vector(self, dt):
        start_time = min(self.time)
        end_time = max(self.time)
        new_time = np.arange(start_time, end_time, dt)
        self.new_time = [t for t in new_time]

    def assign_rigid_bodies(self):
        """assign the rigid bodies acco to the names"""
        for attribute in dir(self.body_a):

            if ('_' in attribute[0]) == False:
                setattr(self, attribute,
                        self.obs['rigid_bodies'][getattr(self.body_a, attribute)])

        # catch inverse flips
    def inverse_quat2(self, call_name, eps=0.1):
        """Inverse the quaternions and their jumps and interpolate them to a new time vector."""
        rigid_b = getattr(self, call_name)

        # Original time vector
        original_time_vector = self.time  # assuming time is in ms

        # new time vector
        new_time_vector = self.new_time

        # Detect inverse jumps
        quaternions = [Quaternion(w, x, y, z) for w, x, y, z in zip(
            rigid_b['qw'], rigid_b['qx'], rigid_b['qy'], rigid_b['qz'])]
        quaternions = [q if q.w >= 0 else -q for q in quaternions]

        # Detect remaining jumps
        quaternions_new = []
        for i in range(len(quaternions)):
            if i > 0 and abs(abs(quaternions[i-1].w / quaternions[i].w) - 1) > eps:
                quaternions_new.append(quaternions[i-1])
            else:
                quaternions_new.append(quaternions[i])

        # Convert quaternions to format for interpolation
        quats_for_interpolation = np.array(
            [[q.x, q.y, q.z, q.w] for q in quaternions_new])

        # Perform interpolation
        slerp = Slerp(original_time_vector,
                      R.from_quat(quats_for_interpolation))
        interpolated_quats = slerp(new_time_vector).as_quat()

        # Interpolate positions (assuming linear interpolation is sufficient)
        interpolated_positions = interpolate_positions(
            original_time_vector, new_time_vector, rigid_b)

        # Create the formatted output
        formatted_output = [[iq[0], iq[1], iq[2], iq[3], ip[0]*1000, ip[1]*1000, ip[2]*1000]
                            for iq, ip in zip(interpolated_quats, interpolated_positions)]

        setattr(self, call_name, formatted_output)

    def clear_quaternions(self):
        """clear the quaternions and their jumps"""
        for attribute in dir(self.body_a):
            if ('_' in attribute[0]) == False:
                self.inverse_quat2(attribute)


def filter_array(arr, t=0.4, decimal_places=7):
    """Filter the array using the given threshold t and round the filtered values to the specified number of decimal places"""
    filtered_arr = []
    loc_value = arr[0]
    filtered_arr.append(round(loc_value, decimal_places))
    for value in arr[1:]:
        loc_value = t * loc_value + (1 - t) * value
        filtered_arr.append(round(loc_value, decimal_places))
    for i in range(len(arr)-1, 0, -1):
        loc_value = t * loc_value + (1 - t) * filtered_arr[i]
        filtered_arr[i] = round(loc_value, decimal_places)
    return filtered_arr


def arrange_force_sensor_data(data, num_sensors=8):
    force_data = data['observation']['analogs']
    force_list = []
    for i in range(num_sensors):
        force_list.append(np.array(filter_array(force_data[i]['force'])))
        force_list.append(np.array(filter_array(force_data[i]['dforce'])))
    return force_list


def calibrate_force_sensor_data(calib_dir, num_sensors=8, verbose=False):
    """This function calibrates the force sensor data.

    Args:
        calib_dir (str): directory of calibration files
    """
    calib_filenames = os.listdir(calib_dir)

    calib_dict = {
        'scale': np.zeros(num_sensors),
        'offset': np.zeros(num_sensors),
    }

    for file in calib_filenames:
        filename = f'{calib_dir}/{file}'

        with open(filename, encoding='utf-8', errors='ignore') as f:
            data = json.load(f)
            force_list = arrange_force_sensor_data(data, num_sensors)

            fx = np.array(data['observation']['force_torques'][0]['fx'])
            fy = np.array(data['observation']['force_torques'][0]['fy'])
            fz = np.array(data['observation']['force_torques'][0]['fz'])

            f_norm = np.sqrt(fx**2 + fy**2 + fz**2)
            f_norm = fz

            # Finding the matching pair
            max_corr = -1
            matched_sensor = None
            for i, sensor_force in enumerate(force_list):
                coefficients = np.polyfit(sensor_force, f_norm, 1)
                calibrated_sensor = coefficients[0] * \
                    sensor_force + coefficients[1]
                corr = np.corrcoef(f_norm, calibrated_sensor)[0, 1]
                if corr > max_corr:
                    max_corr = corr
                    matched_sensor = i

            # Calibrating the sensor
            if matched_sensor is not None:
                # Linear fitting (assuming linear relationship)
                # Replace with specific calibration process if different
                coefficients = np.polyfit(
                    force_list[matched_sensor], f_norm, 1)
                calibrated_sensor = coefficients[0] * \
                    force_list[matched_sensor] + coefficients[1]

                calib_dict['scale'][matched_sensor] = coefficients[0]
                calib_dict['offset'][matched_sensor] = coefficients[1]

                # Optionally: Save or return the calibrated data
                if verbose:
                    plt.figure(file)
                    plt.title(f'{file[:10]}: {matched_sensor}, {max_corr}')
                    plt.plot(f_norm)
                    plt.plot(calibrated_sensor)
                    plt.xlabel(f'{coefficients[0]}; {coefficients[1]}')

            else:
                print("No matching sensor found for file:", filename)

    return calib_dict


# %%
if __name__ == '__main__':
    calibrate_force_sensor_data('../measurement/calibration')
    finger_a = FingerAssignment(4, 6, 3, 2, 7, 5, 0, 1)
    body_a = RigidBodyAssignment(0, 1, 2, 3, 4)
    tmin = 500
    tlim = 4400
    fs = 12
    path = '../measurement/23_01_31/'

    data = TestEvaluator(
        finger_a, body_a, name='2023_01_31_18_12_48.json', path=path, calib_path='./')


# %%
