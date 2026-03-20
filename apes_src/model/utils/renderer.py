import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Util function for loading point clouds|
import numpy as np

# Data structures and functions for rendering
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras, 
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRenderer,
    PulsarPointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from PIL import Image

from alpha_shapes import Alpha_Shaper, plot_alpha_shape
from alpha_shapes.boundary import Boundary, get_boundaries
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from torch import Tensor
from typing import List, Dict
import os
import time
import sys
from einops import repeat, pack, rearrange
from pytorch_lightning import LightningModule

class Renderer(nn.Module):
    def __init__(self, device='cuda:0', batch_size=8):
        super().__init__()
        # 设置渲染参数
        self.device = device
        self.batch_size = batch_size
        # self.views = [55,145,235,325] # degree of 360
        self.views = [55,145,235,325] # degree of 360
        self.batch_repeat_views = []
        for _ in range(self.batch_size):
            for num in self.views:
                self.batch_repeat_views.append(num)
        # self.batch_repeat_views = [num for num in self.views for _ in range(self.batch_size)]
        # print(self.batch_repeat_views)
        # input("...")
        self.view_num = len(self.views)
        self.R, self.T = look_at_view_transform(1.5, 15, self.batch_repeat_views) 
        self.raster_settings = PointsRasterizationSettings(
            image_size=256, 
            radius = 0.01,
            points_per_pixel = 5
        )
        # self.cameras = PerspectiveCameras(R=self.R, T=self.T, device=device)
        self.cameras = FoVOrthographicCameras(device=device, R=self.R, T=self.T, znear=0.01)
        self.rasterizer=PointsRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.renderer = PointsRenderer(
            rasterizer=self.rasterizer,
            # Pass in background_color to the alpha compositor, setting the background color 
            # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
            compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0))
        )

    def forward(self, points, colors): # input:(B,3,N)
        points = rearrange(points, 'B C N -> B N C')
        # points = points.reshape(self.batch_size, -1, 3) #(B,3,N) -> (B,N,3) 
        points = points.repeat_interleave(4, dim=0) #(B,N,3) -> (B*4,N,3) 
        colors = colors.repeat_interleave(4, dim=0)

        point_cloud = Pointclouds(points=[points[i] for i in range(points.shape[0])], 
                                features=[colors[i] for i in range(colors.shape[0])])
        images = self.renderer(point_cloud)
        images = images.reshape(self.batch_size, 4,256,256,3)
        return images



if __name__ == '__main__':
    # 'python src/model/utils/renderer.py '
    from argparse import ArgumentParser
    parser = ArgumentParser()
    # Basic Training Control
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--loss', default='MSEloss', type=str)
    parser.add_argument('--lr', default=1e-4, type=float)

    # dataset 
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--dataset', type=str, default="modelnet")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--pcd_train_path', type=str, default='data/modelnet/pcd/train/')
    parser.add_argument('--cls_label_train_path', type=str, default='data/modelnet/label/train/')
    parser.add_argument('--pcd_val_path', type=str, default='data/modelnet/pcd/test/')
    parser.add_argument('--cls_label_val_path', type=str, default='data/modelnet/label/test/')

    # LR Scheduler
    parser.add_argument('--lr_scheduler', choices=['step', 'cosine'], default='cosine', type=str)
    parser.add_argument('--lr_decay_steps', default=20, type=int)
    parser.add_argument('--lr_decay_rate', default=0.5, type=float)
    parser.add_argument('--lr_decay_min_lr', default=1e-5, type=float)

    #train
    parser.add_argument('--which_ds', default="local", type=str)
    parser.add_argument('--max_epochs', default=200, type=int)

    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    parser.add_argument('--device', default=device, type=str)

    args = parser.parse_args()

    renderer = Renderer(args.device, args.batch_size)
    img_data=torch.rand((3,2048)).to(device).unsqueeze(0).repeat(4,1,1)
    img_color=torch.rand((2048,3)).to(device).unsqueeze(0).repeat(4,1,1)
    print(img_data.shape)
    print(img_color.shape)

    output=renderer(img_data,img_color)
    print(output.shape)


