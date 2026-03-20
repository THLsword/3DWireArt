import os
import sys
import json
import numpy as np
import open3d as o3d

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.neighbors import NearestNeighbors

def estimate_normals_pca(points: np.ndarray, k: int = 30) -> np.ndarray:
    """
    用 PCA 在局部邻域拟合法向量
    Args:
        points: np.array, shape=(N,3)
        k:     int, 邻居数（不含自身）
    Returns:
        normals: np.array, shape=(N,3)，每行单位化后的法向量
    """
    N = points.shape[0]
    normals = np.zeros((N, 3), dtype=np.float64)

    # 构建邻居搜索结构
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='kd_tree').fit(points)
    dists, idxs = nbrs.kneighbors(points)

    for i in range(N):
        # 排除自身（第一个邻居距离为 0）
        neigh_pts = points[idxs[i, 1:]]  
        # 计算邻域重心
        centroid = neigh_pts.mean(axis=0)
        # 构造 3x3 协方差矩阵
        cov = (neigh_pts - centroid).T @ (neigh_pts - centroid) / k
        # 特征分解
        eigvals, eigvecs = np.linalg.eigh(cov)
        # 最小特征值方向为法向量
        normal = eigvecs[:, np.argmin(eigvals)]
        # 单位化
        normals[i] = normal / np.linalg.norm(normal)

    return normals


def load_npz(file_path: str) -> tuple[torch.Tensor, torch.Tensor]:
    """
    input: file path  
    outputs: points: Tensor[n,3]  
             normals: Tensor[n,3]  
    """
    npz_path = ''
    if os.path.exists(os.path.join(file_path, "model_normalized_4096.npz")):
        npz_path = os.path.join(file_path, "model_normalized_4096.npz")
    elif os.path.exists(os.path.join(file_path, "model_normalized_5000.npz")):
        npz_path = os.path.join(file_path, "model_normalized_5000.npz")
    else:
        raise FileNotFoundError(f"model.npz文件不存在。")
    
    with np.load(npz_path) as npz_data:
        points = npz_data["points"]
        normals = npz_data["normals"]

    # normals = estimate_normals_pca(points, k=15)

    points = torch.tensor(points, dtype=torch.float32)
    normals = torch.tensor(normals, dtype=torch.float32)

    return points, normals

def load_obj(file_path):
    # 定义一个空列表来存储顶点坐标
    vertices = []

    # 打开.obj文件进行读取
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('v '):
                _, x, y, z = line.split()
                vertices.append((float(x), float(y), float(z)))
    return torch.tensor(vertices)


if __name__ == '__main__':
    # ` python src/dataset/load_npz.py `
    file_path = "data/models/cat"
    points, normals = load_npz(file_path)
    print(points.shape) # (4096,3)
    print(normals.shape) # (4096,3)