import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from tqdm import tqdm

from pytorch3d.renderer import (
    AlphaCompositor,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds


def save_img(img, file_name):
    """Save a float image in [0, 1] to disk."""
    img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)
    image.save(file_name)


def save_obj(filename, cps):
    """Write point coordinates to a simple OBJ vertex list."""
    with open(filename, 'w') as file:
        for point in cps:
            file.write("v {} {} {}\n".format(*point))


class Model(nn.Module):
    def __init__(self, pcd, view_angels, device, point_num: int = 4096, radius=0.01):
        super().__init__()
        self.device = device
        self.pcd = pcd  # [N, 3]
        self.register_buffer('init_colors', torch.zeros(point_num))

        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.init_colors)))

        self.views = view_angels
        self.view_num = len(self.views)
        self.R, self.T = look_at_view_transform(1.5, 15, self.views)
        self.raster_settings = PointsRasterizationSettings(
            image_size=128,
            radius=radius,
            points_per_pixel=5,
        )
        self.cameras = FoVOrthographicCameras(
            device=self.device,
            R=self.R,
            T=self.T,
            znear=0.01,
        )
        self.rasterizer = PointsRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.renderer = PointsRenderer(
            rasterizer=self.rasterizer,
            compositor=AlphaCompositor(background_color=(0, 0, 0)),
        )

    def forward(self):
        base = self.init_colors.to(self.device)
        colors_ = torch.sigmoid(base + self.displace)  # [N]

        points = self.pcd.unsqueeze(0).repeat(self.view_num, 1, 1)
        colors = colors_.unsqueeze(1).repeat(1, 3)
        colors = colors.unsqueeze(0).repeat(self.view_num, 1, 1)

        point_cloud = Pointclouds(
            points=[points[i] for i in range(points.shape[0])],
            features=[colors[i] for i in range(colors.shape[0])],
        )
        images = self.renderer(point_cloud)
        return images, colors_


def weighted_L1_loss(pred, target):
    """Penalize under-coverage more strongly than over-coverage."""
    pred = pred.sum(dim=-1) / 3
    target = target.sum(dim=-1) / 3

    count_all = torch.sum(target > 0)
    l1_loss = torch.where(pred <= target, (pred - target) * 2, (pred - target) * 0.5)
    l1_loss = l1_loss.abs()
    return l1_loss.sum() / count_all


def visual_training(input_pcd, contour_imgs, epoch, view_angels, device, point_num: int = 4096, radius=0.01) -> tuple[Tensor, Tensor]:
    pcd_tensor = torch.as_tensor(input_pcd, dtype=torch.float32, device=device)
    contour_imgs_tensor = torch.as_tensor(contour_imgs / 255.0, dtype=torch.float32, device=device)

    model = Model(pcd_tensor, view_angels, device, point_num, radius).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.5, 0.99))

    loop = tqdm(range(epoch))
    images, colors = torch.empty(0), torch.empty(0)
    for _ in loop:
        images, colors = model()
        loss = weighted_L1_loss(images, contour_imgs_tensor)

        loop.set_description(f'Loss: {loss.item():.4f}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return images, colors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default="./data/models/cat")
    parser.add_argument('--FILENAME', type=str, default="model_normalized_4096.npz")

    parser.add_argument('--GT_DIR', type=str, default="./render_utils/alpha_outputs")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/train_outputs")

    parser.add_argument('--EPOCH', type=int, default=50)
    parser.add_argument('--VIEW_ANGELS', type=float, nargs='+', default=[45, 90, 135, 225, 270, 315])

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    os.makedirs(args.SAVE_DIR, exist_ok=True)

    pcd_path = os.path.join(args.DATA_DIR, args.FILENAME)
    with np.load(pcd_path) as npzfile:
        pointcloud = npzfile['points']

    images = []
    for filename in os.listdir(args.GT_DIR):
        if filename.endswith('.png'):
            image_path = os.path.join(args.GT_DIR, filename)
            with Image.open(image_path) as img:
                images.append(np.array(img.convert('RGB')))
    contour_imgs = np.array(images, dtype=np.uint8)

    training_outputs = visual_training(pointcloud, contour_imgs, args.EPOCH, args.VIEW_ANGELS, device)
