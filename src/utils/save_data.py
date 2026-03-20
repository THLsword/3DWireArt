import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from .curve_utils import bezier_sample
from .patch_utils import coons_points


def save_img(img: np.ndarray, file_name):
    """Save an RGB image array in [0, 1] to disk."""
    img = (img * 255).astype(np.uint8)
    image = Image.fromarray(img)
    image.save(file_name)


def save_pcd_obj(filename, points):
    """Save point coordinates as OBJ vertices."""
    with open(filename, 'w') as file:
        for point in points:
            file.write("v {} {} {}\n".format(*point))


def save_loss_fig(loss_list, save_dir):
    """Save a training-loss curve."""
    plt.figure(figsize=(6, 4))
    plt.plot(loss_list, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'train_loss.png'))
    plt.close()


def save_lr_fig(lr_list, save_dir):
    """Save a learning-rate curve."""
    plt.figure(figsize=(6, 4))
    plt.plot(lr_list, label='Learning Rate', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('LR')
    plt.title('Learning Rate')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'learning_rate.png'))
    plt.close()


def save_curves(save_dir, data):
    """Save piecewise-linear curve samples to an OBJ file."""
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    num_interp_points = 8
    interpolated_curves = []
    for curve in data:
        total_points = (curve.shape[0] - 1) * num_interp_points + 1
        interpolated_data = np.zeros((total_points, 3))

        idx = 0
        for point_index in range(len(curve) - 1):
            for t in np.linspace(0, 1, num_interp_points, endpoint=False):
                interpolated_data[idx] = (1 - t) * curve[point_index] + t * curve[point_index + 1]
                idx += 1
        interpolated_data[idx] = curve[-1]
        interpolated_curves.append(interpolated_data)
    interpolated_curves = np.array(interpolated_curves)
    obj_points = interpolated_curves.reshape((-1, 3))

    save_pcd_obj(save_dir, obj_points)


def write_curve_obj(file: str, curves: torch.Tensor, res=100):
    """Write Bezier curve points to an OBJ file."""
    r_linspace = torch.linspace(0, 1, res).to(curves).flatten()[..., None]
    points = bezier_sample(r_linspace, curves)

    color = [0.6, 0.6, 0.6]

    with open(file, 'w') as f:
        for _, vertice in enumerate(points):
            for x, y, z in vertice:
                f.write(f'v {x} {y} {z} {color[0]} {color[1]} {color[2]}\n')


def write_mesh_obj(file: str, patches: torch.Tensor, res=30):
    """Write Coons patches to an OBJ file."""
    linspace = torch.linspace(0, 1, res).to(patches)
    s_grid, t_grid = torch.meshgrid(linspace, linspace)

    verts = coons_points(s_grid.flatten(), t_grid.flatten(), patches).cpu().numpy()

    n_verts = verts.shape[-2]
    colors = np.random.rand(10000, 3)
    with open(file, 'w') as f:
        for patch_index, patch in enumerate(verts):
            color = colors[patch_index]
            for x, y, z in patch:
                f.write(f'v {x} {y} {z} {color[0]} {color[1]} {color[2]}\n')
            for i in range(res - 1):
                for j in range(res - 1):
                    f.write(
                        f'f {i * res + j + 2 + patch_index * n_verts} {i * res + j + 1 + patch_index * n_verts} {(i + 1) * res + j + 1 + patch_index * n_verts}\n'
                    )
                    f.write(
                        f'f {(i + 1) * res + j + 2 + patch_index * n_verts} {i * res + j + 2 + patch_index * n_verts} {(i + 1) * res + j + 1 + patch_index * n_verts}\n'
                    )
