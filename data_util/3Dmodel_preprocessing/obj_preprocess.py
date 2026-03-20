import os
import trimesh
import numpy as np
# import pymeshlab
import argparse
import open3d as o3d
import json
import shutil
import open3d as o3d

# 1. 读取 .obj mesh 模型
def load_mesh(file_path):
    mesh = trimesh.load(file_path)
    return mesh

# 2. 在mesh上采样指定数量的点作为点云
def sample_points_from_mesh(mesh, num_points=1000):
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    return points, face_indices

# 3. 获取采样点的法向量
def get_normals_from_face_indices(mesh, face_indices):
    normals = mesh.face_normals[face_indices]
    return normals

def normalize_points(points):
    centrroid = np.mean(points, axis  = 0)
    points -= centrroid

    max_dist = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
    scale = 1. / max_dist
    points = points * scale

    return points

# 複製文件
def copy_obj_file(source_path, destination_folder):
    # 检查源文件是否为 .obj 文件
    print(source_path)
    # if not source_path.endswith('.obj') or not source_path.endswith('.OBJ'):
    #     raise ValueError("Source file is not an .obj file")
    
    # 检查目标文件夹是否存在，如果不存在则创建
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    
    # 生成目标路径（将文件复制到目标文件夹中）
    file_name = os.path.basename(source_path)  # 获取源文件名
    destination_path = os.path.join(destination_folder, file_name)
    
    # 执行文件复制操作
    shutil.copy2(source_path, destination_path)

def main(directory_path):
    # 1. 列出指定路径文件夹下所有的.npy文件
    obj_file = [f for f in os.listdir(directory_path) if f.endswith('.obj') or f.endswith('.OBJ')]
    print(os.listdir(directory_path))

    # 2. 使用for循环+numpy读取这些.npy文件
    for file in obj_file:
        print(file)
        file_path = os.path.join(directory_path, file)
        num_points = 4096
        mesh = load_mesh(file_path)
        
        # 采样点
        points, face_indices = sample_points_from_mesh(mesh, num_points)
        points = normalize_points(points)
        # 获取法向量
        normals = get_normals_from_face_indices(mesh, face_indices)
        
        # # mesh重建
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        # # 使用 Alpha Shapes 重建网格
        # alpha = 0.1  # 选择适当的 alpha 值
        # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        # print("Mesh created using Alpha Shapes")
        # # 保存生成的网格
        # o3d.io.write_triangle_mesh("reconstructed_mesh_poisson.obj", mesh)


        # 輸出地址
        file_name = file.split('.')[0]
        out_model_dir = os.path.join(directory_path, f"{file_name}")
        os.makedirs(out_model_dir, exist_ok=True)
        # 文件名+保存
        out_npz_file = os.path.join(out_model_dir, "model_normalized_4096.npz")
        out_area_file = os.path.join(out_model_dir, "model_normalized_area.json")
        np.savez(out_npz_file, points=points, normals=normals)
        out_area = {"area": 1.0} # area 和 json是無用的。暫時先不刪除免得後面出錯
        with open(out_area_file, "w") as f:
                json.dump(out_area, f, indent=4)
        # 複製.obj到文件夾中
        copy_obj_file(file_path, out_model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'tool')
    parser.add_argument('--directory_path', type = str, default = '../下載的模型/penguin')
    args = parser.parse_args()
    print("1")
    main(args.directory_path)