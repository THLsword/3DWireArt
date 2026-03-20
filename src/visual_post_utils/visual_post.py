import torch
import torch.nn as nn
from torch import Tensor
from einops import rearrange
import os
import sys
sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')
))
import os
from tqdm import tqdm
import numpy as np
import argparse
from dataset import load_npz, load_obj
from dataset import load_template
from utils.curve_utils import * 

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

def save_img(img,file_name):
    # img (256,256,3) np array
    img = img*255
    img = (img).astype(np.uint8)
    image = Image.fromarray(img)
    image.save(file_name)

def save_obj(filename, cps):
    # cps (p_num, 3)
    with open(filename, 'w') as file:
        # 遍历每个点，将其写入文件
        for point in cps:
            # 格式化为OBJ文件中的顶点数据行
            file.write("v {} {} {}\n".format(*point))

class Model(nn.Module):
    def __init__(self, pcd, view_angels, device):
        super(Model, self).__init__()
        self.device = device
        self.pcd = pcd # [n, 3]
        self.register_buffer('init_colors', torch.zeros(4096))

        # optimize for displacement map and center
        self.register_parameter('displace', nn.Parameter(torch.zeros_like(self.init_colors)))

        # render
        self.views = view_angels # degree of 360
        self.view_num = len(self.views)
        self.R, self.T = look_at_view_transform(1.5, 15, self.views) 
        self.raster_settings = PointsRasterizationSettings(
            image_size=128, 
            radius = 0.01,
            points_per_pixel = 5
        )
        self.cameras = FoVOrthographicCameras(device=self.device, R=self.R, T=self.T, znear=0.01)
        self.rasterizer=PointsRasterizer(cameras=self.cameras, raster_settings=self.raster_settings)
        self.renderer = PointsRenderer(
            rasterizer=self.rasterizer,
            # Pass in background_color to the alpha compositor, setting the background color 
            # to the 3 item tuple, representing rgb on a scale of 0 -> 1, in this case blue
            compositor=AlphaCompositor(background_color=(0, 0, 0))
        )

    def forward(self):
        base = self.init_colors.to(self.device)
        colors_ = torch.sigmoid(base+self.displace) # [n]

        points = self.pcd.unsqueeze(0).repeat(self.view_num,1,1)
        colors = colors_.unsqueeze(1).repeat(1,3) # (n) -> (n, 3)
        colors = colors.unsqueeze(0).repeat(self.view_num,1,1)

        # point_cloud = Pointclouds(points=[self.pcd], features=[colors])
        point_cloud = Pointclouds(points=[points[i] for i in range(points.shape[0])], 
                                features=[colors[i] for i in range(colors.shape[0])])
        images = self.renderer(point_cloud) # (self.view_num,256,256,3)
        return images, colors_

def weighted_L1_loss(pred, target):
    pred = pred.sum(dim=-1)/3     # (B,4,256,256,3) -> (B,4,256,256)
    target = target.sum(dim=-1)/3 # (B,4,256,256,3) -> (B,4,256,256)

    # count_all = torch.sum((pred <= target) & (target > 0))
    count_all = torch.sum(target > 0)
    L1_loss = torch.where(pred <= target, (pred - target)*2, (pred - target)*0.5)
    # L1_loss = pred - target
    L1_loss = L1_loss.abs()
    L1_loss = L1_loss.sum()/count_all

    return L1_loss

def visual_training(input_pcd, contour_imgs, epoch, view_angels, device) -> tuple[Tensor, Tensor]:
    # data to tensor
    pcd_tensor = torch.tensor(input_pcd).to(torch.float32).to(device)
    contour_imgs_tensor = torch.tensor(contour_imgs / 255.0, dtype=torch.float32).to(device)

    # model init
    model = Model(pcd_tensor, view_angels, device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), 0.1, betas=(0.5, 0.99))

    loop = tqdm(list(range(0, epoch)))
    images, colors = torch.empty(0), torch.empty(0)
    for i in loop:
        images, colors = model.forward()
        loss = weighted_L1_loss(images, contour_imgs_tensor)

        loop.set_description('Loss: %.4f' % (loss.item()))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return images, colors


if __name__ == '__main__':
    model_name = 'car6'
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default=f"./data/models/{model_name}")
    parser.add_argument('--FILENAME', type=str, default="model_normalized_4096.npz")

    parser.add_argument('--GT_DIR', type=str, default="./render_utils/alpha_outputs")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/train_outputs")
    
    parser.add_argument('--EPOCH', type=int, default=50)
    parser.add_argument('--VIEW_ANGELS', type=float, default=[45,90,135,225,270,315])

    parser.add_argument('--output_path', type=str, default=f"outputs/{model_name}")
    parser.add_argument('--object_curve_num', type=float, default=25) 
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")

    args = parser.parse_args()

    # device setup
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # create folder
    if not os.path.exists(args.SAVE_DIR):
        os.makedirs(args.SAVE_DIR, exist_ok=True)

    # load pointcloud
    PCD_PATH = os.path.join(args.DATA_DIR, args.FILENAME)
    npzfile = np.load(PCD_PATH)
    pointcloud = npzfile['points'] # [N, 3]

    # load template
    _, vertex_idx, face_idx, symmetriy_idx, curve_idx = load_template(args.template_path)
    sample_num = int(np.ceil(np.sqrt(4096/face_idx.shape[0])))

    # load control points -> curves 
    output_path = args.output_path
    control_points = load_obj(f"{output_path}/control_points.obj")
    curves = control_points[curve_idx] # (curve_num, cp_num, 3)
    linspace = torch.linspace(0, 1, 16).to(curves).flatten()[..., None]
    curves_points = bezier_sample(linspace, curves) # [2*curve num, sample num, 3]
    curves_points = rearrange(curves_points, 'C N K -> (C N) K')
    print(curves_points.shape)

    # # load contour images
    # images = []
    # for filename in os.listdir(args.GT_DIR):
    #     if filename.endswith(".png"): 
    #         image_path = os.path.join(args.GT_DIR, filename)
    #         with Image.open(image_path) as img:
    #             img = img.convert("RGB")
    #             images.append(np.array(img))
    # contour_imgs = np.array(images, dtype=np.uint8)

    # training_outputs = visual_training(pointcloud, contour_imgs, args.EPOCH, args.VIEW_ANGELS, device)