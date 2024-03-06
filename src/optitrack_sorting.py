# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_tracker_from_csv(df, tracker_name):
    """
    get the optitrack measure data for the tracker
    -> format:
    x, y, z, qw, qx, qy, qz
    x1, y1, z1, x2, y2, z2, x3, y3, z3
    """
    start_column = df.columns.get_loc(tracker_name)
    data = df.iloc[3, start_column:start_column+16].astype(float)

    qw, qx, qy, qz, x, y, z, x1, y1, z1, x2, y2, z2, x3, y3, z3 = data
    ms = [np.array([xi, yi, zi]) for xi, yi, zi in zip(
        [x1, x2, x3], [y1, y2, y3], [z1, z2, z3])]

    return ms


def plot_3d_points(names, df):
    """
    Function to plot 3d points returned by func for each name in names.

    Args:
        names (list of str): List of names.
        func (function): Function that takes a name as input and returns three 3D points.

    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Generate list of colors based on number of points
    colors = ['r', 'g', 'b']
    labels = ['point 1', 'point 2', 'point 3']

    for name in names:
        points = get_tracker_from_csv(df, name)
        for point, color, label in zip(points, colors, labels):
            ax.scatter(*point, c=color, label=f'{name}: {label}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    df = pd.read_csv(
        '../measurement/21_12_06/Take 2021-12-06 03.42.36 PM.csv', header=2, low_memory=False)
    tracker_names = ['ZF MC', 'DAU MC', 'DAU DP']
    plot_3d_points(tracker_names, df)

# %%