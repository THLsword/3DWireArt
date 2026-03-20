import os
import sys
from matplotlib.image import imread
from einops import rearrange, repeat
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
from tqdm import tqdm
from dataclasses import dataclass
from common import LoadDataResult, LossInputs, TrainState, lr_lambda
from loss_utils import compute_loss
from utils import write_curve_obj, write_mesh_obj, save_pcd_obj, save_loss_fig, save_lr_fig
from torchinfo import summary
from .apes_model import APESSegmentor, MLPHead, Model


def load_model(pretrain_path: str = 'apes_model/best_val_instance_mIoU_epoch_146.pth', device:str='cpu'):
    model = APESSegmentor()
    print(pretrain_path)
    checkpoint = torch.load(pretrain_path)
    model.load_state_dict(checkpoint['state_dict']) 
    return model.to(device)

def apes_training(data: LoadDataResult, **kwargs):
    '''
    主训练函数, 使用预训练好的APES模型
    '''
    device = kwargs.get("device", torch.device("cpu"))
    batch_size = kwargs.get("batch_size", 1)
    apes_model = load_model()
    # 冻结APESSegmentor参数
    for param in apes_model.parameters():
        param.requires_grad = False
        
    model = Model(data.tpl_params.repeat(batch_size, 1, 1), apes_model).to(device)
    pcd_points = data.pcd_points.to(device)
    prep_points = data.prep_points.to(device)
    tpl_f_idx = data.tpl_face_idx
    tpl_c_idx = data.tpl_curve_idx
    # 只优化Model中非APESSegmentor部分的参数
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        kwargs.get('learning_rate'),
        betas=(0.9, 0.99)
    )
    scheduler = LambdaLR(optimizer, lr_lambda)
    epoch = kwargs.get('epoch', 201)
    loop = tqdm(range(epoch + 1))
    loss_list = []
    lr_list = []

    # mkdir
    output_path = kwargs.get("output_path", "./output")
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    fitting_save_path = os.path.join(output_path, "save_apes")
    os.makedirs(fitting_save_path, exist_ok=True)
    
    for i in loop:
        model.train()
        indices = torch.randperm(pcd_points.size(0))  # 生成一个从 0 到 1023 的随机排列索引
        shuffled_pcd_points = pcd_points[indices].repeat(batch_size, 1, 1)
        shuffled_pcd_points = torch.transpose(shuffled_pcd_points, 1, 2) # (B, N, 3) -> (B, 3, N)
        cp_coord = model(shuffled_pcd_points) # (B, -1, 3)
        patches = cp_coord[:,tpl_f_idx] # (B, face_num, cp_num, 3)
        curves = cp_coord[:,tpl_c_idx] # (B, curve_num, cp_num, 3)
        prep_weights_scaled = 1 + (data.prep_weights_scaled * i / kwargs['epoch']) # 0.75~1.25
        inputs = LossInputs(
            cp_coord=cp_coord.to(device),
            patches=patches.to(device),
            curves=curves.to(device),
            pcd_points=data.pcd_points.unsqueeze(0).to(device),
            pcd_normals=data.pcd_normals.unsqueeze(0).to(device),
            prep_points=data.prep_points.unsqueeze(0).to(device),
            sample_num=data.sample_num,
            tpl_sym_idx=data.tpl_sym_idx.to(device),
            # prep_weights_scaled=prep_weights_scaled.to(device),
            prep_weights_scaled=None,
        )
        state = TrainState(step=i, epoch=epoch)
        loss = compute_loss(inputs, state)
        loop.set_description('Loss: %.4f' % (loss.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # save data
            # record loss and learning rate
        with torch.no_grad():
            loss_list.append(loss.item())
            current_lr = optimizer.param_groups[0]['lr']
            lr_list.append(current_lr)
        
        if i % 50 == 0 or i == kwargs['epoch'] - 1:
            with torch.no_grad():
                # save result as mesh
                write_curve_obj(f"{fitting_save_path}/{i}_curve.obj", curves[0])
                write_mesh_obj(f"{fitting_save_path}/{i}_mesh.obj", patches[0])
                # save control points
                save_pcd_obj(f"{output_path}/control_points.obj", cp_coord[0])

                if i == kwargs['epoch'] - 1:
                    # save nn model
                    torch.save(model, f"{output_path}/model_weights.pth")
                    '''
                    # save apes ds
                    ds1_idx = model.apes.backbone.ds1.idx.cpu()
                    ds2_idx = torch.gather(ds1_idx, dim=1, index=model.apes.backbone.ds2.idx.cpu())
                    ds1_xyz = torch.gather(pcd_points.unsqueeze(0).cpu(), dim=1,index=ds1_idx.unsqueeze(-1).expand(-1, -1, 3)) # (B,1024,3)
                    ds2_xyz = torch.gather(pcd_points.unsqueeze(0).cpu(), dim=1,index=ds2_idx.unsqueeze(-1).expand(-1, -1, 3)) # (B,512,3)
                    save_pcd_obj(f"{output_path}/ds1.obj", ds1_xyz[0])
                    save_pcd_obj(f"{output_path}/ds2.obj", ds2_xyz[0])
                    '''

    # save point cloud
    save_pcd_obj(f"{fitting_save_path}/pcd.obj", data.pcd_points)
    # save loss and lr logs as images
    save_loss_fig(loss_list, log_path) # save loss image
    save_lr_fig(lr_list, log_path) # save learning rate image

def apes_downsample(data: LoadDataResult, **kwargs):
    '''
    測試apes在我的data上的下采樣結果
    '''
    device = kwargs.get("device", torch.device("cpu"))
    batch_size = kwargs.get("batch_size", 1)
    apes_model = load_model(device=device)
    pcd_points = data.pcd_points.to(device)
    pcd_points = pcd_points.unsqueeze(0) # (B,N,3)
    
    # get idx
    input_pcd = rearrange(pcd_points, 'B N C->B C N')
    outputs = apes_model(input_pcd)
    ds1_idx = apes_model.backbone.ds1.idx.cpu()
    ds2_idx = torch.gather(ds1_idx, dim=1, index=apes_model.backbone.ds2.idx.cpu())
    ds1_xyz = torch.gather(pcd_points.cpu(), dim=1,index=ds1_idx.unsqueeze(-1).expand(-1, -1, 3)) # (B,1024,3)
    ds2_xyz = torch.gather(pcd_points.cpu(), dim=1,index=ds2_idx.unsqueeze(-1).expand(-1, -1, 3)) # (B,512,3)

    # mkdir
    output_path = kwargs.get("output_path", "./output")
    os.makedirs(output_path, exist_ok=True)
    save_pcd_obj(f"{output_path}/ds1.obj", ds1_xyz[0])
    save_pcd_obj(f"{output_path}/ds2.obj", ds2_xyz[0])
    