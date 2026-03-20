from xml.dom import INDEX_SIZE_ERR
import numpy as np
import torch
from torch.nn import functional as F
from multipledispatch import dispatch

def cal_tangent_vector(t, params):
    A = np.array([[-1, 1, 0, 0],
                [2, -4, 2, 0],
                [-1, 3, -3, 1],
                [0, 0, 0, 0]])
    dim = np.array([0, 1, 2, 3]) 
    dt = np.power(t, dim) @ A @ params
    tangent_v = dt / np.sqrt(np.sum(dt**2, axis=-1, keepdims=True))
    
    return tangent_v

@dispatch(np.ndarray, np.ndarray)
def bezier_sample(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    # row: coefficient for all control points with different power of t
    A = np.array([[1, 0, 0, 0],
                [-3, 3, 0, 0],
                [3, -6, 3, 0],
                [-1, 3, -3, 1]])
    t = np.power(t, np.array([0, 1, 2, 3])) # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points

@dispatch(torch.Tensor, torch.Tensor)
def bezier_sample(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    # row: coefficient for all control points with different power of t
    A = params.new_tensor([[1, 0, 0, 0],
                           [-3, 3, 0, 0],
                           [3, -6, 3, 0],
                           [-1, 3, -3, 1]])
    

    t = t.pow(t.new_tensor([0, 1, 2, 3]))  # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points

def bezier_sample_8(t, params):
    """Sample points from cubic Bezier curves defined by params at t values."""
    A = params.new_tensor([[1, 0, 0, 0, 0, 0, 0, 0],
                    [-7, 7, 0, 0, 0, 0, 0, 0],
                    [21, -42, 21, 0, 0, 0, 0, 0],
                    [-35, 105, -105, 35, 0, 0, 0, 0],
                    [35, -140, 210, -140, 35, 0, 0, 0],
                    [-21, 105, -210, 210, -105, 21, 0, 0],
                    [7, -42, 105, -140, 105, -42, 7, 0],
                    [-1, 7, -21, 35, -35, 21, -7, 1]
                    ])

    t = t.pow(t.new_tensor([0, 1, 2, 3, 4, 5, 6, 7]))  # [n_samples, 4]

    points = t @ A @ params  # [..., n_samples, 3]
    return points

def process_curves(params, vertex_idxs, curve_idxs):
    """ Process params to curve control points.
    Args: 
        params: (batch_size, output_dim)
        vertex_idxs: (point#, 3)
        curve_idx: (curve#, 4)
    Return:
        vertices: (batch_size, point#, 3)
        curves: (batch_size, curve#, 4, 3)
    """
    vertices = params.clone()[:, vertex_idxs]
    curves = vertices[:, curve_idxs] #(b, curve#, 4, 3)

    return vertices, curves

def process_FoldNet_curves(params, vertex_idxs, curve_idxs):
    """ Process params to curve control points.
    Args: 
        params: Decoder's output (batch_size, num_points(122) ,3)
        vertex_idxs: (point#, 3)
        curve_idx: (curve#, 4)
    Return:
        vertices: (batch_size, point#, 3)
        curves: (batch_size, curve#, 4, 3)
    """
    vertices = params.clone()
    curves = vertices[:, curve_idxs] #(b, curve#, 4, 3)

    return vertices, curves

def write_curve_points(file, curves, control_point_num = 4, res=300):
    """Write Bezier curve points to an obj file."""
    r_linspace = torch.linspace(0, 1, res).to(curves).flatten()[..., None]
    if control_point_num == 4:
        points = bezier_sample(r_linspace, curves)
    elif control_point_num == 8:
        points = bezier_sample_8(r_linspace, curves)

    c = [0.6, 0.6, 0.6] # (r, g, b)

    with open(file, 'w') as f:
        for p, vertice in enumerate(points):
            for x, y, z in vertice:
                f.write(f'v {x} {y} {z} {c[0]} {c[1]} {c[2]}\n')

def bezier_curve(control_points: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    计算 N 阶 Bézier 曲线在参数 t 处的点。
    ctrl_pts: Tensor of shape [N, 3]
    t: Tensor of shape [num_samples], 取值范围 [0,1]
    返回: Tensor of shape [num_samples, 3]
    """
    # 使用 De Casteljau 递归插值
    # 将 ctrl_pts 扩展到 [num_samples, N, 3]
    num_pts = control_points.shape[0]
    # 先将 ctrl_pts 复制 num_samples 份
    pts = control_points.unsqueeze(0).expand(t.shape[0], -1, -1).clone()  # [S, N, 3]
    # t 扩展为 [S, 1, 1] 以便广播
    u = t.view(-1, 1, 1)
    # 迭代进行线性插值
    for k in range(1, num_pts):
        pts = pts[:, :-1, :] * (1 - u) + pts[:, 1:, :] * u
    return pts[:, 0, :]  # [S,3]

def sample_near_bezier(control_points: torch.Tensor,
                       num_samples: int,
                       max_radius: float=0.005) -> torch.Tensor:
    """
    在 Bézier 曲线附近的三维空间中采样点（带可导偏移）。
    Args:
        control_points: (n+1,3) Tensor, requires_grad=True
        num_samples: 采样点个数 N
        max_radius: 控制最大偏移半径 R
    Returns:
        sampled_pts: (N,3) Tensor，可用于后续计算 loss，梯度可传导到 control_points
    """
    device = control_points.device
    # 1) 在 [0,1] 上均匀采样 t
    t = torch.rand(num_samples, device=device)

    # 2) 计算曲线上对应的点
    pts_on_curve = bezier_curve(control_points, t)  # (N,3)

    # 3) 在三维球体内均匀采样偏移向量
    #    先从标准正态采噪声，再归一化成单位向量（各方向等概率）
    noise = torch.randn(num_samples, 3, device=device)
    noise = noise / noise.norm(dim=1, keepdim=True).clamp(min=1e-8)
    u = torch.rand(num_samples, 1, device=device) * max_radius
    offsets = noise * u  # (N,3)

    # 4) 最终采样点
    sampled_pts = pts_on_curve + offsets  # (N,3)
    return sampled_pts

def batch_sample_near_bezier(curves: torch.Tensor,
                             num_samples: int,
                             max_radius: float = 0.005) -> torch.Tensor:
    """
    对 curves 中的每条 Bézier 曲线批量采样。
    Args:
        curves: Tensor of shape (B, C, P, 3)，requires_grad=True
        num_samples: 每条曲线的采样点数量 S
        max_radius: 最大偏移半径 R
    Returns:
        samples: Tensor of shape (B, C, S, 3)
    """
    B, C, P, _ = curves.shape
    # 预分配输出
    samples = torch.zeros((B, C, num_samples, 3),
                          device=curves.device,
                          dtype=curves.dtype)
    for b in range(B):
        for c in range(C):
            # 取出单条曲线的控制点集，调用采样函数
            ctrl = curves[b, c]  # (P, 3)
            samples[b, c] = sample_near_bezier(ctrl, num_samples, max_radius)  # (S, 3)
    return samples