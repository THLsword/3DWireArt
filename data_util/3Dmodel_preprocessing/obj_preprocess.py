import os
import trimesh
import numpy as np
# import pymeshlab
import argparse
import open3d as o3d
import json
import shutil
import open3d as o3d

"""Preprocess `.obj` meshes into normalized point-cloud training assets.

This script loads every OBJ mesh in a directory, samples surface points,
derives normals from the sampled faces, normalizes the point cloud into a unit
scale around the origin, and writes the result to an `.npz` file. It also
stores a placeholder area JSON file and copies the original OBJ into the
generated output folder.
"""

def load_mesh(file_path):
    """Load a mesh from an OBJ file."""
    mesh = trimesh.load(file_path)
    return mesh

def sample_points_from_mesh(mesh, num_points=1000):
    """Sample points uniformly from the mesh surface."""
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    return points, face_indices

def get_normals_from_face_indices(mesh, face_indices):
    """Get normals for the sampled faces."""
    normals = mesh.face_normals[face_indices]
    return normals

def normalize_points(points):
    """Center and scale points into a unit sphere."""
    centrroid = np.mean(points, axis  = 0)
    points -= centrroid

    max_dist = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    scale = 1. / max_dist
    points = points * scale

    return points

def copy_obj_file(source_path, destination_folder):
    """Copy the original OBJ file to the output folder."""
    print(source_path)
    # if not source_path.endswith('.obj') or not source_path.endswith('.OBJ'):
    #     raise ValueError("Source file is not an .obj file")
    
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    file_name = os.path.basename(source_path)
    destination_path = os.path.join(destination_folder, file_name)
    
    shutil.copy2(source_path, destination_path)

def main(directory_path):
    """Process all OBJ files in a directory."""
    obj_file = [f for f in os.listdir(directory_path) if f.endswith('.obj') or f.endswith('.OBJ')]
    print(os.listdir(directory_path))

    for file in obj_file:
        print(file)
        file_path = os.path.join(directory_path, file)
        num_points = 4096
        mesh = load_mesh(file_path)
        
        points, face_indices = sample_points_from_mesh(mesh, num_points)
        points = normalize_points(points)
        normals = get_normals_from_face_indices(mesh, face_indices)
        file_name = file.split('.')[0]
        out_model_dir = os.path.join(directory_path, f"{file_name}")
        os.makedirs(out_model_dir, exist_ok=True)
        out_npz_file = os.path.join(out_model_dir, "model_normalized_4096.npz")
        out_area_file = os.path.join(out_model_dir, "model_normalized_area.json")
        np.savez(out_npz_file, points=points, normals=normals)
        out_area = {"area": 1.0} # The area JSON is currently unused, but it is kept to avoid breaking downstream code.
        with open(out_area_file, "w") as f:
                json.dump(out_area, f, indent=4)
        copy_obj_file(file_path, out_model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'tool')
    parser.add_argument('--directory_path', type = str, default = '../下載的模型/penguin')
    args = parser.parse_args()
    print("1")
    main(args.directory_path)
