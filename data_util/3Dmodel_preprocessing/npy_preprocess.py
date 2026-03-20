import os
import numpy as np
import argparse

"""Utilities for inspecting `.npy` point-cloud files in a directory.

This script scans a target folder, loads each `.npy` file, creates an output
folder with the same stem name, and prints the shapes of the full array, point
coordinates, and normals. It is mainly used as a lightweight preprocessing or
data-inspection helper.
"""

def main(directory_path):
    """Process all `.npy` files under ``directory_path``.

    For each file, the function creates a sibling folder with the same base
    name, splits the loaded array into point coordinates and normals, and
    prints their shapes for quick inspection.

    Args:
        directory_path: Directory that contains `.npy` files to inspect.
    """
    # 1. List all `.npy` files in the target directory.
    npy_files = [f for f in os.listdir(directory_path) if f.endswith('.npy')]

    # 2. Iterate through the files and load each array with NumPy.
    for file in npy_files:
        file_path = os.path.join(directory_path, file)
        data = np.load(file_path)
        
        # 3. Create an output folder that shares the same stem as the file.
        folder_name = file.split('.')[0]  # Remove the extension and use the stem as the folder name.
        folder_path = os.path.join(directory_path, folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Split the array into XYZ coordinates and normal vectors.
        points, normals = data[:, :3], data[:, 3:]
        print(data.shape)
        print(points.shape)
        print(normals.shape)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'tool')
    parser.add_argument('--directory_path', type = str, default = '../吉他/')
    args = parser.parse_args()

    main(args.directory_path)
