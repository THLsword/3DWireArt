from ast import List
from dataclasses import dataclass
import torch


@dataclass
class LoadDataResult:
    pcd_points: torch.Tensor
    pcd_normals: torch.Tensor
    prep_weights_scaled: torch.Tensor
    prep_points: torch.Tensor
    tpl_params: torch.Tensor
    tpl_vertex_idx: torch.Tensor
    tpl_face_idx: torch.Tensor
    tpl_sym_idx: torch.Tensor
    tpl_curve_idx: torch.Tensor
    tpl_adjacency: List
    sample_num: int

@dataclass
class LossInputs:
    cp_coord: torch.Tensor     # [B, N, 3]
    patches: torch.Tensor      # [B, F, M, 3]
    curves: torch.Tensor       # [B, C, M, 3]
    pcd_points: torch.Tensor   # [B, N, 3]
    pcd_normals: torch.Tensor  # [B, N, 3]
    prep_points: torch.Tensor  # [B, N, 3]
    sample_num: int
    tpl_sym_idx: torch.Tensor
    prep_weights_scaled: torch.Tensor  

@dataclass
class TrainState:
    step: int  
    epoch: int

def lr_lambda(epoch: int) -> float:
    warm_epoch = 50
    if epoch < warm_epoch:
        # return math.exp((epoch - warm_epoch) / k)
        return 0.1
    else:
        # return 0.99 ** (epoch - warm_epoch)
        return 1.0