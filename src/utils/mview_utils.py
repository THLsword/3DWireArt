from xml.dom import INDEX_SIZE_ERR
import numpy as np
import torch
from torch.nn import functional as F
from multipledispatch import dispatch
from .curve_utils import bezier_sample

def multiview_sample(pcd_points: torch.Tensor, mview_weights) -> torch.Tensor:
    """
    Inputs: point cloud [N, 3], weights [N]
    
    Output: [M, 3], M 表示权重 > 0.5的点
    """
    mask = mview_weights > 0.5
    sampled_points = pcd_points[mask]
    return sampled_points

def curve_probability(pcd_s, curves, sample_num):
    '''
    似乎是用來計算出一個值, 決定是否保留這條curve. 但是棄用了.
    '''
    # 計算multi view points 到 curve上採樣點的最短距離
    # 計算rate ：mv points<->curve points / curve points
    curves_s = torch.linspace(0, 1, sample_num).to(curves).flatten()[..., None]
    curve_points = bezier_sample(curves_s, curves)
    # curve_chamfer = curve_chamfer_loss(curve_points, pcd_points)

    # x1 is curve sample points
    # x2 is mesh sample points
    x1 = curve_points.unsqueeze(0)
    x1 = x1.view(1,-1,3)
    x2 = pcd_s.unsqueeze(0)

    x1_norm = x1.pow(2).sum(-1, keepdim=True) # [1, curve_n * sample_n, 1]
    x2_norm = x2.pow(2).sum(-1, keepdim=True) # [1, multi_view points , 1]

    res = torch.baddbmm(
        x2_norm.transpose(-2, -1),
        x1,
        x2.transpose(-2, -1),
        alpha=-2
    ).add_(x1_norm).clamp_min_(1e-10).sqrt_()
    # ||(x1 - x2)|| = ((x1 ^ 2) - 2 * (x1 * x2) + (x2 ^ 2)) ^ (1/2)
    # [b, n_sample_points, n_mesh_points]

    chamferloss_a, idx_a = res.min(2)  # [1, curve_n * n_sample_points]
    chamferloss_b, idx_b = res.min(1)  # [1, multi_view_points] multi view点到curve的距离，idx是所有curve上采样点的idx
    chamferloss_a = chamferloss_a.mean(1)

    idx_b = idx_b.squeeze(0)
    mask = torch.zeros(x1.shape[1], dtype=torch.bool)
    mask[idx_b] = True
    mask = mask.view(-1, sample_num) # (curve_n, sample_num)

    # each curves' min distance rate
    min_distance_rate = []
    for i, mask_ in enumerate(mask):
        min_dis_num = mask_.sum()
        rate = min_dis_num/sample_num
        min_distance_rate.append(rate)
    min_distance_rate = torch.tensor(min_distance_rate)

    # threshold
    mean = min_distance_rate.mean()
    std = min_distance_rate.std()
    threshold = mean + std
    rate_mask = min_distance_rate > 0.5

    topk_curves = curves[rate_mask]

    return topk_curves, rate_mask