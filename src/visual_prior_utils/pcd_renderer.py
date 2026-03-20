import argparse
import os

import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import (
    AlphaCompositor,
    FoVOrthographicCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer,
    look_at_view_transform,
)
from pytorch3d.structures import Pointclouds
from tqdm import tqdm


def pcd_renderer(input_pcd, view_angels, device, radius=0.01):
    """Render a point cloud from multiple predefined azimuth angles."""
    verts = torch.as_tensor(input_pcd, dtype=torch.float32, device=device)
    rgb = torch.ones_like(verts)
    point_cloud = Pointclouds(points=[verts], features=[rgb])

    rotations = []
    translations = []
    for angle in view_angels:
        rotation, translation = look_at_view_transform(1.5, 15, angle)
        rotations.append(rotation)
        translations.append(translation)

    raster_settings = PointsRasterizationSettings(
        image_size=128,
        radius=radius,
        points_per_pixel=5,
    )

    rendered_images = []
    for rotation, translation in tqdm(list(zip(rotations, translations))):
        cameras = FoVOrthographicCameras(
            device=device,
            R=rotation,
            T=translation,
            znear=0.01,
        )
        rasterizer = PointsRasterizer(cameras=cameras, raster_settings=raster_settings)
        renderer = PointsRenderer(
            rasterizer=rasterizer,
            compositor=AlphaCompositor(background_color=(0.0, 0.0, 0.0)),
        )
        output = renderer(point_cloud)
        image_data = output[0, ..., :3].cpu().numpy()
        rendered_images.append((image_data * 255).astype(np.uint8))

    return rendered_images


if __name__ == '__main__':
    model_name = 'car6'
    parser = argparse.ArgumentParser()
    parser.add_argument('--DATA_DIR', type=str, default=f"./data/models/{model_name}")
    parser.add_argument('--SAVE_DIR', type=str, default="./render_utils/render_outputs")
    parser.add_argument('--filename', type=str, default="model_normalized_4096.npz")

    parser.add_argument('--output_path', type=str, default=f"outputs/{model_name}")
    parser.add_argument('--object_curve_num', type=float, default=25)
    parser.add_argument('--template_path', type=str, default="data/templates/sphere24")

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    os.makedirs(args.SAVE_DIR, exist_ok=True)

    file_path = os.path.join(args.DATA_DIR, args.filename)
    with np.load(file_path) as npzfile:
        pointcloud = npzfile['points']  # [N, 3]

    view_angels = [45, 90, 135, 225, 270, 315]
    rendered_images = pcd_renderer(pointcloud, view_angels, device)

    for index, data in enumerate(rendered_images):
        image = Image.fromarray(data)
        save_filename = f'{os.path.splitext(args.filename)[0]}_{index}.png'
        print(save_filename)
        image.save(os.path.join(args.SAVE_DIR, save_filename))
