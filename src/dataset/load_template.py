import os
import sys
import numpy as np
import ast
from typing import List, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import transforms

def load_template(file_path: str):
    """
    以24face球為例:  
    tpl_params: [366]  
    tpl_vertex_idx: [122,3]   
    tpl_face_idx: [24,12]   
    tpl_sym_idx: [2,49]   
    tpl_curve_idx: [96,4] <- (其中有一半是重復的, 實際只有48條curve)
    """
    topology_path = os.path.join(file_path, 'topology.txt')
    vertices_path = os.path.join(file_path, 'vertices.txt')
    topology = ast.literal_eval(open(topology_path, 'r').read()) # topology is a list (face number,vertex number per face):(30,12)
    control_point_num = len(topology[0])/4 + 1

    # Process template data
    parameters = []
    vertex_idxs = np.zeros([len(open(vertices_path, 'r').readlines()), 3],dtype=np.int64)
    # vertices
    for i, l in enumerate(open(vertices_path, 'r')):
        value = l.strip().split(' ')
        _, a, b, c = value
        vertex_idxs[i] = [len(parameters), len(parameters)+1, len(parameters)+2]
        parameters.extend([float(a), float(b), float(c)])
    parameters = torch.tensor(parameters).squeeze()
    vertex_idxs = torch.from_numpy(vertex_idxs.astype(np.int64))

    # faces
    face_idxs = np.empty([len(topology), len(topology[0])])
    for i, patch in enumerate(topology):
        for j, k in enumerate(patch):
            face_idxs[i, j] = k
    face_idxs = torch.from_numpy(face_idxs.astype(np.int64))

    # curves
    n_curve = len(topology) * 4
    if len(topology[0]) == 12:
        curve_idxs = np.empty([n_curve, 4])
        for i, patch in enumerate(topology):
            curve_idxs[i*4, :] = patch[:4]
            curve_idxs[i*4+1, :] = patch[3:7]
            curve_idxs[i*4+2, :] = patch[6:10]
            curve_idxs[i*4+3, :] = [patch[9], patch[10], patch[11], patch[0]]
    elif len(topology[0]) == 28:
        curve_idxs = np.empty([n_curve, 8])
        for i, patch in enumerate(topology):
            curve_idxs[i*4, :] = patch[:8]
            curve_idxs[i*4+1, :] = patch[7:15]
            curve_idxs[i*4+2, :] = patch[14:22]
            curve_idxs[i*4+3, :] = [patch[21], patch[22], patch[23], patch[24], patch[25], patch[26], patch[27], patch[0]]
    curve_idxs = torch.from_numpy(curve_idxs.astype(np.int64))

    # symmetry
    xs, ys = [], []
    for line in open(os.path.join(file_path, 'symmetries.txt'), 'r'):
        x, y = line.strip().split(' ')
        xs.append(int(x))
        ys.append(int(y))
    symmetriy_idx = (xs, ys)
    symmetriy_idx = torch.tensor(symmetriy_idx)

    return parameters, vertex_idxs, face_idxs, symmetriy_idx, curve_idxs

def compute_adjacency_from_idx(
    idx: torch.Tensor
) -> List[Tuple[int, int, int, int]]:
    """
    计算基于控制点 idx (Tensor[n,12]) 的邻接关系列表。

    Args:
        idx (Tensor): shape [n,12]，每个元素是一个整数，表示该 patch
                      上对应位置的控制点全局索引。

    Returns:
        adjacency (List of (i1,e1,i2,e2)):
            所有 i1 < i2 且 patch i1 的第 e1 条边与 patch i2 的 第 e2 条边
            共享同一条几何边（索引相同但参数方向相反）。
    """
    EDGE_CP_IDS = {
        0: [0, 1, 2, 3],
        1: [3, 4, 5, 6],
        2: [6, 7, 8, 9],
        3: [9, 10, 11, 0],
    }
    
    n = idx.shape[0]
    adjacency = []

    for i1 in range(n):
        for e1, ids1 in EDGE_CP_IDS.items():
            cp_idx1 = idx[i1, ids1]               # Tensor[4]

            for i2 in range(i1 + 1, n):
                for e2, ids2 in EDGE_CP_IDS.items():
                    cp_idx2 = idx[i2, ids2]       # Tensor[4]

                    # 参数方向相反时，共用边的控制点序列正好是反向一致
                    if torch.equal(cp_idx1, cp_idx2.flip(0)) or torch.equal(cp_idx1, cp_idx2):
                        adjacency.append((i1, e1, i2, e2))
    return adjacency


if __name__ == '__main__':
    # ` python src/dataset/load_template.py `
    file_path = "data/templates/sphere24"
    parameters, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(file_path)
    print(parameters.shape)