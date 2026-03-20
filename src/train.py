import os
import argparse
import numpy as np
import torch

from dataset.load_pcd import load_npz
from dataset.load_template import load_template, compute_adjacency_from_idx
from utils.mview_utils import multiview_sample
from optimization import opt_training
from common import LoadDataResult


def str2bool(value):
    """Parse common string representations of booleans for argparse."""
    if isinstance(value, bool):
        return value

    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def load_data(model_path, prep_output_path, template_path):
    """Load the point cloud, preprocessing output, and template data."""
    # Load point cloud and normals.
    pcd_points, pcd_normals = load_npz(model_path)

    # Load preprocessing output: `weights.pt`.
    weight_path = os.path.join(prep_output_path, "weights.pt")
    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"Missing preprocessing weights: {weight_path}. "
            "Run the preprocessing step before training."
        )

    prep_weights = torch.load(weight_path, map_location="cpu").detach()
    # 0~1 -> -0.25~0.25
    prep_weights_scaled = (prep_weights - 0.5) / 2
    prep_weights_scaled.requires_grad_(False)

    # Sample multi-view points from the point cloud.
    prep_points = multiview_sample(pcd_points, prep_weights)  # (M, 3)

    # Load template.
    tpl_params, tpl_vertex_idx, tpl_face_idx, tpl_sym_idx, tpl_curve_idx = load_template(template_path)
    # Number of samples on one face of the template.
    sample_num = int(np.ceil(np.sqrt(pcd_points.shape[0] / tpl_face_idx.shape[0])))
    tpl_params = tpl_params.view(-1, 3)
    tpl_adjacency = compute_adjacency_from_idx(tpl_face_idx)  # 24-face template -> list length 48

    return LoadDataResult(
        pcd_points=pcd_points,
        pcd_normals=pcd_normals,
        prep_weights_scaled=prep_weights_scaled,
        prep_points=prep_points,
        tpl_params=tpl_params,
        tpl_vertex_idx=tpl_vertex_idx,
        tpl_face_idx=tpl_face_idx,
        tpl_sym_idx=tpl_sym_idx,
        tpl_curve_idx=tpl_curve_idx,
        tpl_adjacency=tpl_adjacency,
        sample_num=sample_num,
    )


def main(**kwargs):
    # Set device.
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # Load data.
    data = load_data(
        model_path=kwargs["model_path"],
        prep_output_path=kwargs["prep_output_path"],
        template_path=kwargs["template_path"],
    )
    opt_training(data, device=device, **kwargs)


if __name__ == "__main__":
    # `python src/train.py`
    model = "rabbit"
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=f"data/models/{model}")
    # parser.add_argument("--template_path", type=str, default="data/templates/cube24")
    parser.add_argument("--template_path", type=str, default="data/templates/sphere24")
    # parser.add_argument("--template_path", type=str, default="data/templates/donut")
    parser.add_argument("--output_path", type=str, default=f"outputs/{model}")
    parser.add_argument("--prep_output_path", type=str, default=f"outputs/{model}/prep_outputs/train_outputs")

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1)  # 不要改，就是 1
    parser.add_argument("--learning_rate", type=float, default=0.0005)

    parser.add_argument("--d_curve", type=str2bool, default=False)  # 是否删掉不需要的 curve
    parser.add_argument("--k", type=int, default=2)  # 曲线采样点最近的 k 个点
    parser.add_argument("--match_rate", type=float, default=0.2)

    parser.add_argument("--prep_bool", type=str2bool, default=True)  # 是否加入前处理的点云作为 cross attention
    parser.add_argument("--view_angels", type=float, nargs="+", default=[45, 90, 135, 225, 270, 315])
    args = parser.parse_args()

    main(**vars(args))
