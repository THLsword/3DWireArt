import os

import torch
import torch.nn as nn
from einops import rearrange
from tqdm import tqdm

from apes import APESSegmentor
from common import LoadDataResult, LossInputs, TrainState
from loss_utils import ComputeLoss
from utils import save_loss_fig, save_lr_fig, save_pcd_obj, write_curve_obj, write_mesh_obj


def load_model(model_path: str = "apes_model/best_val_instance_mIoU_epoch_146.pth", **kwargs):
    """Load the pretrained APES segmentor used for point weighting."""
    device = kwargs.get("device", "cpu")
    model = APESSegmentor().to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    return model

class Optimizer:
    """Minimal optimizer wrapper for directly updating point parameters."""

    def __init__(self, points: torch.Tensor, step_size: float = 1e-3) -> None:
        if points.ndim != 2:
            raise ValueError("points must have shape [N, C]")

        self.points = points
        self.step_size = step_size

    def zero_grad(self) -> None:
        if self.points.grad is not None:
            self.points.grad.zero_()

    def step(self, step_size: float = None) -> None:
        if self.points.grad is None:
            return

        with torch.no_grad():
            if step_size is None:
                step_size = self.step_size
            self.points -= step_size * self.points.grad

    def reset(self) -> None:
        """Reset optimizer state if a subclass stores moments."""
        pass

class VectorAdam(Optimizer):
    """Adam-style optimizer for vector-valued control point offsets."""

    def __init__(
        self,
        points: torch.Tensor,
        step_size: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.99,
        eps: float = 1e-12,
    ) -> None:
        super().__init__(points, step_size)

        if not 0.0 < beta1 < 1.0:
            raise ValueError(f"beta1 must be in (0, 1), got {beta1}")
        if not 0.0 < beta2 < 1.0:
            raise ValueError(f"beta2 must be in (0, 1), got {beta2}")
        if eps <= 0.0:
            raise ValueError(f"eps must be > 0, got {eps}")

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.beta1_acc = 1.0
        self.beta2_acc = 1.0
        self.m1: torch.Tensor = None
        self.m2: torch.Tensor = None
        self.current_epoch = 0
        self.base_step_size = step_size

    def reset(self) -> None:
        self.beta1_acc = 1.0
        self.beta2_acc = 1.0
        self.m1 = None
        self.m2 = None

    def step(self, step_size: float = None) -> None:
        if self.points.grad is None:
            return

        if self.m1 is None or self.m2 is None:
            self.m1 = torch.zeros_like(self.points)
            self.m2 = torch.zeros_like(self.points)

        self.beta1_acc *= self.beta1
        self.beta2_acc *= self.beta2

        with torch.no_grad():
            if step_size is None:
                if self.current_epoch < 50:
                    step_size = self.base_step_size * 0.1
                else:
                    decay_factor = 1.0 - (self.current_epoch - 50) / 3000
                    step_size = self.base_step_size * max(decay_factor, 0.01)

            self.m1 = self.beta1 * self.m1 + (1.0 - self.beta1) * self.points.grad
            self.m2 = self.beta2 * self.m2 + (1.0 - self.beta2) * torch.sum(
                torch.square(self.points.grad), dim=1, keepdim=True
            )

            m1_corr = self.m1 / (1.0 - self.beta1_acc)
            m2_corr = self.m2 / (1.0 - self.beta2_acc)
            grad = m1_corr / (torch.sqrt(m2_corr) + self.eps)
            self.points -= step_size * grad

        self.current_epoch += 1

class Model(nn.Module):
    """Optimize control points as learnable offsets from the template."""

    def __init__(self, template_params: torch.Tensor):
        super().__init__()
        self.register_buffer("template_params", template_params)
        self.offset = nn.Parameter(torch.zeros_like(template_params))

    def forward(self) -> torch.Tensor:
        return self.template_params + self.offset

def opt_training(data: LoadDataResult, **kwargs):
    """Fit template control points to one target point cloud by direct optimization."""
    device = kwargs.get("device", torch.device("cpu"))
    model = Model(data.tpl_params.to(device)).to(device)
    pcd_points = data.pcd_points.to(device)
    tpl_f_idx = data.tpl_face_idx
    tpl_c_idx = data.tpl_curve_idx
    tpl_adjacency = data.tpl_adjacency

    epoch = kwargs.get("epoch", 201)
    loop = tqdm(range(epoch + 1))
    loss_list = []
    lr_list = []

    # Prepare output directories.
    output_path = kwargs.get("output_path", "./output")
    os.makedirs(output_path, exist_ok=True)
    log_path = os.path.join(output_path, "logs")
    os.makedirs(log_path, exist_ok=True)
    fitting_save_path = os.path.join(output_path, "save_opt")
    os.makedirs(fitting_save_path, exist_ok=True)

    # Build APES-guided point weights for later stages.
    apes_model = load_model(device=device).to(device)
    for param in apes_model.parameters():
        param.requires_grad = False
    apes_model.eval()

    input_pcd = rearrange(data.pcd_points.unsqueeze(0).to(device), "B N C -> B C N")
    _ = apes_model(input_pcd)
    ds1_idx = apes_model.backbone.ds1.idx.cpu()
    ds2_idx = torch.gather(ds1_idx, dim=1, index=apes_model.backbone.ds2.idx.cpu())

    batch_size = data.pcd_points.unsqueeze(0).size(0)
    point_count = data.pcd_points.size(0)
    weights_ds1 = torch.full((batch_size, point_count), 1.0, device="cpu")
    weights_ds1.scatter_(1, ds1_idx, 1.1)
    weights_ds2 = torch.full((batch_size, point_count), 1.0, device="cpu")
    weights_ds2.scatter_(1, ds2_idx, 1.1)

    loss_computer = ComputeLoss(
        pcd_points.unsqueeze(0).to(device),
        data.pcd_normals.unsqueeze(0).to(device),
        data.tpl_sym_idx.to(device),
        data.sample_num,
        epoch,
        tpl_adjacency,
        kwargs.get("view_angels"),
    )

    optimizer = VectorAdam(
        points=model.offset,
        step_size=2.0 * kwargs.get("learning_rate"),
    )

    for i in loop:
        model.train()
        cp_coord = model()
        cp_coord = cp_coord * (1.01 ** int(i / 200))
        cp_coord = cp_coord.unsqueeze(dim=0)
        patches = cp_coord[:, tpl_f_idx]
        curves = cp_coord[:, tpl_c_idx]

        prep_weights_scaled = 1 + (data.prep_weights_scaled * i / epoch) / 2
        if i <= 500:
            effective_prep_weights = prep_weights_scaled
        elif i <= 1000:
            effective_prep_weights = prep_weights_scaled * weights_ds1
        else:
            effective_prep_weights = prep_weights_scaled * weights_ds2

        inputs = LossInputs(
            cp_coord=cp_coord.to(device),
            patches=patches.to(device),
            curves=curves.to(device),
            pcd_points=data.pcd_points.unsqueeze(0).to(device),
            pcd_normals=data.pcd_normals.unsqueeze(0).to(device),
            prep_points=data.prep_points.unsqueeze(0).to(device),
            sample_num=data.sample_num,
            tpl_sym_idx=data.tpl_sym_idx.to(device),
            prep_weights_scaled=effective_prep_weights.to(device),
        )
        state = TrainState(step=i, epoch=epoch)
        loss_computer.updata_data(inputs, state)

        if i <= 500:
            loss = (
                loss_computer.compute_area_weighted_chamfer_loss(weight_norm=0.02)
                + loss_computer.compute_planar_loss()
                + loss_computer.compute_patch_symmetry_loss(0.1)
                + loss_computer.compute_concavity_enhancement_loss(0.1)
                + loss_computer.silhouette_loss(0.2)
                + loss_computer.compute_patch_rectangular_loss(0)
                + loss_computer.compute_patch_grad_uniform_loss(0)
            ).mean()
        elif i <= 1000:
            loss = (
                loss_computer.compute_area_weighted_chamfer_loss(weight_norm=0.02)
                + loss_computer.compute_planar_loss()
                + loss_computer.compute_patch_symmetry_loss()
                + loss_computer.compute_concavity_enhancement_loss(0.4)
                + loss_computer.silhouette_loss(0.2)
                + loss_computer.compute_patch_rectangular_loss(0)
            ).mean()
        else:
            loss = (
                loss_computer.compute_area_weighted_chamfer_loss(weight_norm=0.02)
                + loss_computer.compute_planar_loss()
                + loss_computer.compute_patch_symmetry_loss()
                + loss_computer.compute_concavity_enhancement_loss(0.6)
                + loss_computer.silhouette_loss(0.0)
                + loss_computer.compute_patch_rectangular_loss(0.01)
            ).mean()

        loop.set_description("Loss: %.4f" % loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            loss_list.append(loss.item())

        if i % 300 == 0 or i == epoch:
            with torch.no_grad():
                write_curve_obj(f"{fitting_save_path}/{i}_curve.obj", curves[0])
                write_mesh_obj(f"{fitting_save_path}/{i}_mesh.obj", patches[0])
                save_pcd_obj(f"{output_path}/control_points.obj", cp_coord[0])

                if i == epoch:
                    torch.save(model, f"{output_path}/model_weights.pth")

    save_pcd_obj(f"{fitting_save_path}/pcd.obj", data.pcd_points)
    save_loss_fig(loss_list, log_path)
    save_lr_fig(lr_list, log_path)
