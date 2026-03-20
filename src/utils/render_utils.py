import torch
import torch.nn as nn
from torch import Tensor
import os
from tqdm import tqdm
import numpy as np
import argparse
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
import random
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

class PointcloudRenderer:
    '''
    Init:
        view_angles: List[azimuth angles]
        device: str
    '''
    def __init__(self, view_angles, device: str = "cpu", elevations_reverse: bool=False, img_size: int=128):
        # 增加俯视视角 (azimuth=0, elevation=90)
        self.view_angles = view_angles + [0.0]  # append to end
        self.device = device
        self.view_num = len(self.view_angles)

        # elevation = 15 for all views except top-down
        if elevations_reverse:
            elevations = [-15.0] * (len(self.view_angles) - 1) + [-90.0]
        else:
            elevations = [15.0] * (len(self.view_angles) - 1) + [90.0]

        # 计算多视角下的相机变换
        self.R, self.T = look_at_view_transform(
            dist=1.5,
            elev=elevations,
            azim=self.view_angles
        )

        self.raster_settings = PointsRasterizationSettings(
            image_size=img_size,
            radius=0.03,
            points_per_pixel=5
        )

        # 渲染器与合成器设置（cameras 按视角在 render() 中动态生成）
        self.compositor = AlphaCompositor(background_color=(0, 0, 0))

    def render(self, pointcloud: torch.Tensor, step: int=0, random_bool: bool=False) -> torch.Tensor:
        """
        Inputs:
            pointcloud: [B, N, C]  # C >= 3, 用前三维做坐标
        Outputs:
            images: [B, V, H, W, 3]  # 每个 batch 的 V 个视角图像
        """
        B, N, _ = pointcloud.shape
        pts = pointcloud[..., :3].to(self.device)
        colors = torch.ones((B, N, 3), device=self.device)

        if step > 0 and step % 200 == 0 and random_bool:
            print(f"Step {step}: Randomizing all view angles and elevations.")
            start_angle = random.random() * 360.0
            angle_increment = 360.0 / self.view_num
            self.view_angles = [(start_angle + i * angle_increment) % 360.0 for i in range(self.view_num)]

            # self.view_angles = [random.random() * 360.0 for _ in range(self.view_num)]
            self.elevations = [random.uniform(-90.0, 90.0) for _ in range(self.view_num)]

            # 使用新的随机方位角和仰角重新计算相机变换矩阵 R 和 T
            self.R, self.T = look_at_view_transform(
                dist=1.5, # 相机距离保持不变
                elev=self.elevations, # 使用新生成的随机仰角列表
                azim=self.view_angles # 使用新生成的随机方位角列表
            )

        # 构建 batched Pointclouds
        pcds = Pointclouds(points=pts, features=colors)

        images_all_views = []

        for i in range(self.view_num):
            R_i = self.R[i].unsqueeze(0).expand(B, -1, -1).to(self.device)  # [B, 3, 3]
            T_i = self.T[i].unsqueeze(0).expand(B, -1).to(self.device)      # [B, 3]
            cameras_i = FoVOrthographicCameras(device=self.device, R=R_i, T=T_i, znear=0.01)

            rasterizer = PointsRasterizer(cameras=cameras_i, raster_settings=self.raster_settings)
            renderer = PointsRenderer(rasterizer=rasterizer, compositor=self.compositor)

            images_i = renderer(pcds)  # [B, H, W, 3]
            images_all_views.append(images_i)

        # [B, V, H, W, 3]
        images = torch.stack(images_all_views, dim=1)
        return images
    
