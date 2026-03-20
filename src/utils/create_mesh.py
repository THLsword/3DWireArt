import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
import json

def create_cylinder_mesh(point_a, point_b, radius, segments=32):
    direction = point_b - point_a
    length = np.linalg.norm(direction)

    # 创建一个沿 z 轴的标准圆柱体（不包含顶面和底面）
    cylinder = trimesh.creation.cylinder(radius, length, sections=segments)
    cylinder.vertices[:, 2] += length / 2  # 将圆柱体中心移动到起点

    # 计算方向向量的旋转矩阵
    direction_normalized = direction / length
    z_axis = np.array([0, 0, 1])
    rotation_vector = np.cross(z_axis, direction_normalized)
    rotation_angle = np.arccos(np.dot(z_axis, direction_normalized))

    if np.linalg.norm(rotation_vector) > 1e-6:
        rotation_vector /= np.linalg.norm(rotation_vector)
        rotation = Rotation.from_rotvec(rotation_vector * rotation_angle)
        rotation_matrix = rotation.as_matrix()

        # 将 3x3 旋转矩阵扩展为 4x4 齐次变换矩阵
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation_matrix

        # 应用旋转变换
        cylinder.apply_transform(transform_matrix)

    # 移动到正确的位置
    cylinder.apply_translation(point_a)

    return cylinder

def create_bspline_mesh(bspline_points, radius, segments=32):
    bspline_mesh = trimesh.Trimesh()

    # 遍历相邻的点，使用 create_cylinder_mesh 创建部分圆柱
    for i in range(len(bspline_points) - 1):
        point_a = bspline_points[i]
        point_b = bspline_points[i + 1]
        segment_mesh = create_cylinder_mesh(point_a, point_b, radius, segments)
        bspline_mesh = trimesh.util.concatenate(bspline_mesh, segment_mesh)

    # 在每个采样点生成球体并添加到 mesh 中
    for point in bspline_points:
        sphere = trimesh.creation.icosphere(radius=radius, subdivisions=1)
        sphere.apply_translation(point)
        bspline_mesh = trimesh.util.concatenate(bspline_mesh, sphere)

    return bspline_mesh

def create_mesh(bspline, radius, output_path, object_curve_num: int=None):
    '''
    Inputs:
        bspline: [N, sample num, 3]
    '''
    if object_curve_num==None:
        # bspline [N, M, 3]
        bspline_mesh = trimesh.Trimesh()
        for i in bspline:
            one_curve = create_bspline_mesh(i, radius, 10)
            bspline_mesh = trimesh.util.concatenate(bspline_mesh, one_curve)
        bspline_mesh.export(f'{output_path}/bspline_mesh.obj')


        # save json
        array_list = bspline.tolist()
        data = {'wires': array_list}
        with open(f"{output_path}/spline_data.json", "w") as file:
            json.dump(data, file, indent=4)
    else:
        # bspline [N, M, 3]
        bspline_mesh = trimesh.Trimesh()
        for i in bspline:
            one_curve = create_bspline_mesh(i, radius, 10)
            bspline_mesh = trimesh.util.concatenate(bspline_mesh, one_curve)
        bspline_mesh.export(f'{output_path}/bspline_mesh_{object_curve_num}.obj')


        # save json
        array_list = bspline.tolist()
        data = {'wires': array_list}
        with open(f"{output_path}/spline_data_{object_curve_num}.json", "w") as file:
            json.dump(data, file, indent=4)