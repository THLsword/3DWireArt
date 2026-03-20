from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from torchvision.transforms import v2
from pytorch3d.ops.knn import knn_gather, knn_points
from common import LossInputs, TrainState
from utils import coons_points, coons_normals, coons_mtds
from utils import bezier_sample
from utils import PointcloudRenderer, save_img
from .loss_functions import area_weighted_chamfer_loss, compute_batch_chamfer, planar_patch_loss, patch_overlap_loss
from .loss_functions import curve_perpendicular_loss, flatness_area_loss, patch_symmetry_loss, curve_curvature_loss, compute_beam_gap_loss
from .loss_functions import curvature_loss, patch_rectangular_loss, chamfer_and_grad_uniformity, compute_G1_loss
import math
from einops import rearrange
import numpy as np
import clip
import collections
from PIL import Image
import os
from typing import Optional

# def compute_loss(inputs: LossInputs, state: TrainState):
#     """
#     参数:  
#         inputs: GeometryInputs 数据结构，封装了所有模型输入数据  
#         state: TrainState 结构, 包含当前步骤和總epoch數  
#     """
#     # 取出參數
#     cp_coord = inputs.cp_coord
#     patches = inputs.patches
#     curves = inputs.curves
#     pcd_points = inputs.pcd_points
#     pcd_normals = inputs.pcd_normals
#     prep_points = inputs.prep_points
#     sample_num = inputs.sample_num
#     tpl_sym_idx = inputs.tpl_sym_idx
#     prep_weights_scaled = inputs.prep_weights_scaled
#     i = state.step
#     epoch = state.epoch

#     device = patches.device
#     batch_size = patches.shape[0]
#     face_num = patches.shape[1]
    
#     st = torch.empty(batch_size, face_num, sample_num**2, 2).uniform_().to(device) # [b, patch_num, sample_num, 2]

#     # preprocessing
#     # patches (B,face_num,cp_num,3)
#     points  = coons_points(st[..., 0], st[..., 1], patches) # [b, patch_num, sample_num, 3]
#     normals = coons_normals(st[..., 0], st[..., 1], patches)
#     mtds    = coons_mtds(st[..., 0], st[..., 1], patches)   # [b, patch_num, sample_num] 

#     # area-weighted chamfer loss (position + normal)
#     chamfer_loss, normal_loss = area_weighted_chamfer_loss(
#         mtds, points, normals, 
#         pcd_points, pcd_normals, 
#         prep_weights_scaled,
#     )

#     # # curve (b, curve_num, cp_num, 3) chamfer loss
#     # curves_sample_num = 16
#     # linspace = torch.linspace(0, 1, curves_sample_num).to(curves).flatten()[..., None]
#     # curve_points = bezier_sample(linspace, curves)
#     # curve_chamfer_loss, _, _, _ = compute_batch_chamfer(curve_points, pcd_points*1.01)
#     # curve_chamfer_loss = curve_chamfer_loss.mean(1)

#     # flatness loss
#     planar_loss = planar_patch_loss(st, points, mtds)

#     # symmetry loss
#     symmetry_loss = torch.zeros(1).to(patches)
#     symmetry_loss = patch_symmetry_loss(tpl_sym_idx[0], tpl_sym_idx[1], cp_coord)

#     # beam gap loss
#     thres = 0.9
#     beam_gap_loss = compute_beam_gap_loss(points, normals, pcd_points, pcd_points, thres)
#     # beam_gap_loss = compute_beam_gap_loss(points, normals, pcd_points, prep_points, thres)

#     # # curvature loss
#     # curv_loss = curvature_loss(curves) # =0.002

#     # patch rectangular loss
#     rectangular_loss = patch_rectangular_loss(patches)

#     # stable_loss = chamfer_loss + 2*planar_loss + 0.1*symmetry_loss + normal_loss + 0.2 * beam_gap_loss
#     stable_loss =  chamfer_loss + 2.0 * planar_loss + 0.1 * symmetry_loss + 0.005*normal_loss + 0.20 * beam_gap_loss + 0.02 * rectangular_loss# + 0.1 * curv_loss 
#     loss = stable_loss

#     return loss.mean()

# def compute_loss_stage2(inputs: LossInputs, state: TrainState):
#     """
#     参数:  
#         inputs: GeometryInputs 数据结构，封装了所有模型输入数据  
#         state: TrainState 结构, 包含当前步骤和總epoch數  
#     """
#     # 取出參數
#     cp_coord = inputs.cp_coord
#     patches = inputs.patches
#     curves = inputs.curves
#     pcd_points = inputs.pcd_points
#     pcd_normals = inputs.pcd_normals
#     prep_points = inputs.prep_points
#     sample_num = inputs.sample_num
#     tpl_sym_idx = inputs.tpl_sym_idx
#     prep_weights_scaled = inputs.prep_weights_scaled
#     i = state.step
#     epoch = state.epoch

#     device = patches.device
#     batch_size = patches.shape[0]
#     face_num = patches.shape[1]
    
#     # preprocessing
#         # patches (B,face_num,cp_num,3)
#     st = torch.empty(batch_size, face_num, sample_num**2, 2).uniform_().to(device) # [b, patch_num, sample_num, 2]
#     points  = coons_points(st[..., 0], st[..., 1], patches) # [b, patch_num, sample_num, 3]
#     normals = coons_normals(st[..., 0], st[..., 1], patches)
#     mtds    = coons_mtds(st[..., 0], st[..., 1], patches)   # [b, patch_num, sample_num] 

#     # area-weighted chamfer loss (position + normal)
#     chamfer_loss, normal_loss = area_weighted_chamfer_loss(
#         mtds, points, normals, 
#         pcd_points, pcd_normals, 
#         prep_weights_scaled,
#     )

#     # curve chamfer loss
#         # curves (b, curve_num, cp_num, 3)
#     curves_sample_num = 16
#     linspace = torch.linspace(0, 1, curves_sample_num).to(curves).flatten()[..., None]
#     curve_points = bezier_sample(linspace, curves)
#     curve_chamfer_loss, _, _, _ = compute_batch_chamfer(curve_points, pcd_points*1.01)
#     curve_chamfer_loss = curve_chamfer_loss.mean(1)

#     # multi view curve loss
#     _, _, mv_curve_loss, _ = compute_batch_chamfer(curve_points, prep_points)
#     mv_curve_loss = mv_curve_loss.mean(1)

#     # flatness loss
#     planar_loss = planar_patch_loss(st, points, mtds)

#     # # Orthogonality loss
#     # perpendicular_loss = curve_perpendicular_loss(patches)

#     # # flatness loss
#     # FA_loss = flatness_area_loss(st, points, mtds)

#     # symmetry loss
#     symmetry_loss = torch.zeros(1).to(patches)
#     symmetry_loss = patch_symmetry_loss(tpl_sym_idx[0], tpl_sym_idx[1], cp_coord)

#     # curvature loss
#     curvature_loss = curve_curvature_loss(curves, linspace)

#     # beam gap loss
#     thres = 0.9
#     beam_gap_loss = compute_beam_gap_loss(points, normals, pcd_points, pcd_points, thres)
#     # beam_gap_loss = compute_beam_gap_loss(points, normals, pcd_points, prep_points, thres)

#     # test loss
#     # stable_loss = chamfer_loss + 2*planar_loss + 0.1*symmetry_loss + normal_loss + 0.2 * beam_gap_loss
#     stable_loss =  chamfer_loss + 10.0 * planar_loss + 0.1 * symmetry_loss + 0.005*normal_loss + 0.20 * beam_gap_loss# + 0.003 * curvature_loss# + 0.005*overlap_loss
#     loss = stable_loss

#     return loss.mean()

class CLIPVisualEncoder(nn.Module):
    def __init__(self, clip_model, device):
        super().__init__()
        self.clip_model = clip_model
        self.featuremaps = collections.OrderedDict()
        self.device = device
        self.n_channels = 3
        self.kernel_h = 32
        self.kernel_w = 32
        self.step = 32
        self.num_patches = 49

        for i in range(12):  # 12 resblocks in VIT visual transformer
            self.clip_model.visual.transformer.resblocks[i].register_forward_hook(
                self.make_hook(i))

    def make_hook(self, name):
        def hook(module, input, output):
            if len(output.shape) == 3:
                self.featuremaps[name] = output.permute(
                    1, 0, 2)  # LND -> NLD bs, smth, 768
            else:
                self.featuremaps[name] = output

        return hook

    def forward(self, x, mode="train"):
        self.featuremaps.clear()
        fc_features = self.clip_model.encode_image(x).float()
        # fc_features = self.clip_model.encode_image(x, attn_map, mode).float()
        # Each featuremap is in shape (5,50,768) - 5 is the batchsize(augment), 50 is cls + 49 patches, 768 is the dimension of the features
        # for each k (each of the 12 layers) we only take the vectors
        featuremaps = [self.featuremaps[k] for k in range(12)]

        return fc_features, featuremaps

class ComputeLoss:
    def __init__(self, pcd_points, pcd_normals, tpl_sym_idx, sample_num, epoch, tpl_adjacency, view_angles=[45,90,135,225,270,315]):
        self.pcd_points = pcd_points
        self.pcd_normals = pcd_normals
        self.tpl_sym_idx = tpl_sym_idx
        self.view_angles = view_angles
        self.tpl_adjacency = tpl_adjacency
        self.gaussian_blur = v2.GaussianBlur(kernel_size=71, sigma=7.0)

        self.sample_num = sample_num
        self.batch_size = self.pcd_points.shape[0]
        self.epoch = epoch
        self.device = self.pcd_points.device

        # from https://github.com/kenji-tojo/fab3dwire/blob/main/wiregrad/visual_loss.py#L178
        self.clip_conv_layer = 11 

        self.renderer = PointcloudRenderer(self.view_angles, self.device)
        self.model, preprocess = clip.load("ViT-B/32", device=self.device)
        self.visual_encoder = CLIPVisualEncoder(self.model, self.device)
        # self.img_size = preprocess.transforms[1].size
        for param in self.model.parameters():
            param.requires_grad = False
        self.preprocess = transforms.Compose(
            [preprocess.transforms[-1]])
        self.target_transform = transforms.Compose([
            transforms.ToTensor(),
        ])  # clip normalisation
        self.normalize_transform = transforms.Compose([
            preprocess.transforms[0],  # Resize
            preprocess.transforms[1],  # CenterCrop
            preprocess.transforms[-1],  # Normalize
        ])
        self.num_augs = 0
        augemntations = []
        if self.num_augs > 0:
            augemntations.append(transforms.RandomPerspective(
                fill=0, p=1.0, distortion_scale=0.5))
            augemntations.append(transforms.RandomResizedCrop(
                224, scale=(0.8, 0.8), ratio=(1.0, 1.0)))
            # augemntations.append(transforms.RandomResizedCrop(
                # 224, scale=(0.4, 0.9), ratio=(1.0, 1.0)))
        augemntations.append(
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)))
        self.augment_trans = transforms.Compose(augemntations)
        self.model.eval()
    
    def updata_data(self, inputs: LossInputs, state: TrainState):
        self.inputs = inputs
        self.state = state
        self.face_num = inputs.patches.shape[1]
        self.prep_weights_scaled = inputs.prep_weights_scaled

        with torch.no_grad():
            self.st = torch.empty(self.batch_size, self.face_num, self.sample_num**2, 2).uniform_().to(self.device) # [b, patch_num, sample_num, 2]
        self.points  = coons_points(self.st[..., 0], self.st[..., 1], self.inputs.patches) # [b, patch_num, sample_num, 3]
        self.normals = coons_normals(self.st[..., 0], self.st[..., 1], self.inputs.patches)
        self.mtds    = coons_mtds(self.st[..., 0], self.st[..., 1], self.inputs.patches)   # [b, patch_num, sample_num] 

        # # 計算gt點雲和patch采樣點的chamfer distance
        # self.chamfer_a, self.idx_a, self.chamfer_b, self.idx_b = self.compute_batch_chamfer_(self.pcd_points, self.points.view(self.batch_size, -1, 3))

    def compute_batch_chamfer_(self, input_a, input_b):
        '''
        Inputs:
            x1 [B,N,3] 
            x2 [B,M,3] 
        Outputs:
            chamferloss_a [B,N]: x1到x2的最短距離
            idx_a [B,N]
            chamferloss_b [B,M]: x2到x1的最短距離
            idx_b [B,M]
        '''
        chamfer_a, idx_a, chamfer_b, idx_b = compute_batch_chamfer(input_a, input_b)
        return chamfer_a, idx_a, chamfer_b, idx_b

    def compute_area_weighted_chamfer_loss(self, weight_ch=1.0, weight_norm=0.005):
        """
        計算area weighted chamfer loss.

        Inputs:
            權重1: chamfer loss(默認 1.0)
            權重2: normal loss(默認 0.005)
        """
        chamfer_loss, normal_loss = area_weighted_chamfer_loss(
                self.mtds, self.points, self.normals, 
                self.pcd_points, self.pcd_normals, 
                self.prep_weights_scaled,
            )
        return chamfer_loss * weight_ch + normal_loss * weight_norm
    
    def compute_planar_loss(self, weight=2.0):
        """
        Flatness loss
        """
        if weight == 0:
            return 0
        planar_loss = planar_patch_loss(self.st, self.points, self.mtds)
        return planar_loss * weight
    
    def compute_patch_symmetry_loss(self, weight=0.1):
        if weight == 0:
            return 0
        symmetry_loss = patch_symmetry_loss(self.tpl_sym_idx[0],self. tpl_sym_idx[1], self.inputs.cp_coord)
        return symmetry_loss * weight
    
    def compute_beam_gap_loss(self, weight=0.20):
        if weight == 0:
            return 0
        thres = 0.9
        beam_gap_loss = compute_beam_gap_loss(self.points, self.normals, self.pcd_points, self.pcd_points, thres)
        return beam_gap_loss * weight
    
    def compute_curvature_loss(self, weight=0.002):
        if weight == 0:
            return 0
        curve_loss = curvature_loss(self.inputs.curves)
        return curve_loss * weight
    
    def compute_patch_rectangular_loss(self, weight=0.02):
        if weight == 0:
            return 0
        rectangular_loss = patch_rectangular_loss(self.inputs.patches)
        return rectangular_loss * weight
    
    def compute_patch_grad_uniform_loss(self, weight=0.1):
        if weight == 0:
            return 0
        grad_uniform_loss = chamfer_and_grad_uniformity(self.points, self.inputs.pcd_points)
        return grad_uniform_loss * weight
    
    def extract_clip_features_(self, input_tensor:torch.Tensor) -> torch.Tensor:
        features = self.model.encode_image(input_tensor)
        return features
    
    def preprocess_tensor_(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B', C, H, W], float in [0,1]
        返回 [B', C, 224, 224]，并做 Normalize
        """
        x = F.interpolate(x, size=(224, 224), mode='bicubic', align_corners=False)
        x = self.preprocess(x)
        return x
    
    def compute_clip_loss(self, weight=0.5):
        if weight == 0:
            return 0
        wire_imgs = self.renderer.render(rearrange(self.points, 'B P N C->B (P N) C'), self.state.step)
        wire_imgs = rearrange(wire_imgs, 'B V H W C->(B V) C H W')
        pcd_imgs = self.renderer.render(self.pcd_points, self.state.step)
        pcd_imgs = rearrange(pcd_imgs, 'B V H W C->(B V) C H W')

        feature_wire = self.extract_clip_features_(self.preprocess_tensor_(wire_imgs))
        feature_pcd = self.extract_clip_features_(self.preprocess_tensor_(pcd_imgs))
        f1 = feature_wire / feature_wire.norm(dim=-1, keepdim=True)
        f2 = feature_pcd / feature_pcd.norm(dim=-1, keepdim=True)

        cos_sim = torch.nn.functional.cosine_similarity(f1, f2, dim=-1)  # [1]
        loss_cos = (1 - cos_sim).mean()
        # loss_mse = torch.nn.functional.mse_loss(f1, f2)
        return loss_cos * weight
    
    def compute_clip_loss_2(self, clip_conv_loss_weight=1.0, clip_fc_loss_weight=1.0):
        wire_imgs = self.renderer.render(rearrange(self.points, 'B P N C->B (P N) C'), self.state.step)
        wire_imgs = rearrange(wire_imgs, 'B V H W C->(B V) C H W')
        pcd_imgs = self.renderer.render(self.pcd_points, self.state.step)
        pcd_imgs = rearrange(pcd_imgs, 'B V H W C->(B V) C H W')
        self.save_rendered_images(wire_imgs, pcd_imgs, 'test_files')

        sketch_augs, img_augs = [self.normalize_transform(wire_imgs)], [self.normalize_transform(pcd_imgs)]
        # if self.state.step % 100 == 0:
        #     self.save_rendered_images(sketch_augs[0], img_augs[0], 'test_files')

        for n in range(self.num_augs):
            # for i in range(wire_imgs[0].shape[0]):
            augmented_pair = self.augment_trans(torch.cat([wire_imgs, pcd_imgs]))
            img_num_ = wire_imgs.shape[0]
            for i in range(img_num_):
                sketch_augs.append(augmented_pair[i].unsqueeze(0))
                img_augs.append(augmented_pair[1+img_num_].unsqueeze(0))

        xs = torch.cat(sketch_augs, dim=0)
        ys = torch.cat(img_augs, dim=0)
        # print("================================")
        # print(xs.requires_grad, ys.requires_grad)
        # print(xs.shape)
        # print(ys.shape)

        # sketch_utils.plot_batch(xs, ys, f"{self.args.output_dir}/jpg_logs", self.counter, use_wandb=False, title="fc_aug{}_iter{}_{}.jpg".format(1, self.counter, mode))

        xs_fc_features, xs_conv_features = self.visual_encoder(xs)
        ys_fc_features, ys_conv_features = self.visual_encoder(ys)

        conv_loss = self.l2_layers_(
            xs_conv_features, 
            ys_conv_features
            )
        clip_conv_loss = conv_loss[self.clip_conv_layer]

        if clip_fc_loss_weight:
            # fc distance is always cos
            # fc_loss = torch.nn.functional.mse_loss(xs_fc_features, ys_fc_features).mean()
            fc_loss = (1 - torch.cosine_similarity(xs_fc_features, ys_fc_features, dim=1)).mean()

        loss = clip_conv_loss * clip_conv_loss_weight + fc_loss * clip_fc_loss_weight
        return loss
    
    def silhouette_loss(self, weight = 0.2):
        if weight == 0:
            return 0
        wire_imgs = self.renderer.render(rearrange(self.points, 'B P N C->B (P N) C'), self.state.step, random_bool=True)
        wire_imgs = rearrange(wire_imgs, 'B V H W C->(B V) C H W')
        pcd_imgs = self.renderer.render(self.pcd_points, self.state.step)
        pcd_imgs = rearrange(pcd_imgs, 'B V H W C->(B V) C H W')
        # if self.state.step % 100 == 0:
        #     self.save_rendered_images(wire_imgs, pcd_imgs, 'test_files')
        # self.save_rendered_images(wire_imgs, pcd_imgs, 'test_files', invert=True)
        wire_imgs_blurred = self.gaussian_blur(wire_imgs)
        pcd_imgs_blurred = self.gaussian_blur(pcd_imgs.detach())
        diff = wire_imgs_blurred - pcd_imgs_blurred
        return torch.mean(torch.square(diff)) * weight
    
    def l2_layers_(self, xs_conv_features, ys_conv_features):
        return [torch.square(x_conv - y_conv).mean() for x_conv, y_conv in
                zip(xs_conv_features, ys_conv_features)]
    
    def save_rendered_images(self, wire_imgs: torch.Tensor, pcd_imgs: torch.Tensor, output_dir: str, prefix: str = "render", invert: bool = False):
        """
        将渲染的图片张量应用高斯模糊后保存到文件。若 invert 为 True，则在保存前对图像进行反色处理。

        Args:
            wire_imgs (torch.Tensor): 线框图张量，形状为 (B*V, C, H, W)。
            pcd_imgs (torch.Tensor): 点云图张量，形状为 (B*V, C, H, W)。
            output_dir (str): 图片保存的目录。
            prefix (str): 保存文件名的前缀。
                        例如，如果 prefix='render'，文件可能保存为 render_wire_blurred_0000.png, render_pcd_blurred_0000.png 等。
            invert (bool): 是否在保存前进行反色。如果为 True，会将每个像素值替换为 255 - 原值。
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # --- 添加高斯模糊功能 ---
        # 定义高斯模糊 transform，使用 kernel_size=100, sigma=7.0
        # 注意：GaussianBlur 期望输入张量的格式是 (..., C, H, W) 或 (C, H, W)
        # 这里的 wire_imgs 和 pcd_imgs 经过前面的 rearrange 已经是 (B*V, C, H, W) 格式，符合要求
        # 如果你的张量在 GPU 上，这个 transform 也可以在 GPU 上执行

        # 应用高斯模糊到输入的图片张量
        wire_imgs_blurred = self.gaussian_blur(wire_imgs)
        pcd_imgs_blurred = self.gaussian_blur(pcd_imgs)
        # --- 添加高斯模糊功能结束 ---


        num_images = wire_imgs_blurred.size(0)  # B * V 的数量 (使用模糊后的张量，大小相同)

        # 假设通道数 C=3 (RGB) 或 C=4 (RGBA)
        # 确保数据类型是浮点数，范围在 [0, 1] 以便后续转换
        # 注意：GaussianBlur 的输出类型通常会与输入类型相同，所以这里保持 float 类型
        wire_imgs_blurred = wire_imgs_blurred.float().clamp(0, 1)
        pcd_imgs_blurred = pcd_imgs_blurred.float().clamp(0, 1)

        for i in range(num_images):
            # 提取单张模糊后的图片张量，形状为 (C, H, W)
            wire_img_tensor = wire_imgs_blurred[i]
            pcd_img_tensor = pcd_imgs_blurred[i]

            # 将通道维度从第二个位置移到最后一个位置，形状变为 (H, W, C)
            wire_img_hwc = wire_img_tensor.permute(1, 2, 0)
            pcd_img_hwc = pcd_img_tensor.permute(1, 2, 0)

            # 将 PyTorch 张量转换为 NumPy 数组
            # 转换为 uint8 类型，范围从 [0, 1] 变为 [0, 255]
            # .detach().cpu().numpy() 是将张量移动到 CPU 并转换为 NumPy 数组的标准做法
            wire_img_np = (wire_img_hwc.detach().cpu().numpy() * 255).astype(np.uint8)
            pcd_img_np = (pcd_img_hwc.detach().cpu().numpy() * 255).astype(np.uint8)

            # --- 如果启用了反色，则进行反色处理 ---
            if invert:
                # 颜色反转：new_pixel = 255 - old_pixel
                wire_img_np = 255 - wire_img_np
                pcd_img_np = 255 - pcd_img_np
            # --- 反色处理结束 ---

            # 转换为 PIL 图像对象
            # 如果通道数 C=4，指定模式为 'RGBA'，否则为 'RGB'
            mode = 'RGB' if wire_img_np.shape[-1] == 3 else 'RGBA'
            wire_img_pil = Image.fromarray(wire_img_np, mode=mode)
            pcd_img_pil = Image.fromarray(pcd_img_np, mode=mode)

            # 构建文件名 (为了区分，可以在文件名中添加 'blurred'，并且在文件名中标注是否反色)
            suffix = "inverted" if invert else "normal"
            wire_filename = os.path.join(output_dir, f"{prefix}_wire_blurred_{suffix}_{i:04d}.png")
            pcd_filename = os.path.join(output_dir, f"{prefix}_pcd_blurred_{suffix}_{i:04d}.png")

            # 保存图片
            wire_img_pil.save(wire_filename)
            pcd_img_pil.save(pcd_filename)

            # print(f"Saved {wire_filename} and {pcd_filename}") # 可选：打印保存信息

        print(f"Successfully saved {num_images * 2} {'inverted ' if invert else ''}blurred images to {output_dir}")


    def compute_smoothness_loss(self, weight=0.1, K: int = 10):
        """
        计算每对相邻 patch 在共享边上的法向量，并输出 G1 连续性损失。

        Args:
            patches: Tensor[n,12,3]，n 片 Coons patch 的控制点坐标
            adjacency: List of (i1, e1, i2, e2)
            K: 每条边上采样点数

        Returns:
            normals1: Tensor[M, K, 3]，第一片 patch 的法向量
            normals2: Tensor[M, K, 3]，第二片 patch 的法向量（方向已对齐）
            loss: Tensor(,)，所有 adjacency 上的平均 G1 损失
        """
        if weight == 0:
            return 0
        device = self.device
        t = torch.linspace(0.0, 1.0, K, device=device)

        losses = []
        for (i1, e1, i2, e2) in self.tpl_adjacency:
            # --- 1) 构建两片 patch 的边界 control dict ---
            def make_control(cp12: torch.Tensor):
                return {
                    'b0': cp12[0:4],           # bottom
                    'b1': cp12[3:7],           # right
                    'b2': cp12[6:10],          # top
                    'b3': cp12[[9,10,11,0]]    # left
                }

            control1 = make_control(self.inputs.patches[0][i1])
            control2 = make_control(self.inputs.patches[0][i2])

            # --- 2) 生成采样参数 u1,v1 和 u2,v2 ---
            def edge_uv(edge_idx: int):
                if   edge_idx == 0: return t, torch.zeros_like(t)
                elif edge_idx == 1: return torch.ones_like(t), t
                elif edge_idx == 2: return 1 - t, torch.ones_like(t)
                elif edge_idx == 3: return torch.zeros_like(t), 1 - t
                else: raise ValueError("edge must be 0..3")

            u1, v1 = edge_uv(e1)
            u2, v2 = edge_uv(e2)

            # --- 3) 计算两片 patch 的偏导并叉积归一化 ---
            # coons_patch_full(control, u, v) 返回 (S, Su, Sv)
            _, Su1, Sv1 = self.coons_patch_full_(control1, u1, v1)
            _, Su2, Sv2 = self.coons_patch_full_(control2, u2, v2)

            n1 = torch.cross(Su1, Sv1, dim=-1)
            n1 = n1 / (n1.norm(dim=-1, keepdim=True) + 1e-9)

            n2 = torch.cross(Su2, Sv2, dim=-1)
            n2 = n2 / (n2.norm(dim=-1, keepdim=True) + 1e-9)

            # --- 4) 对齐方向：若平均内积为负则翻转 n2 ---
            if (n1 * n2).sum(dim=-1).mean() < 0:
                n2 = -n2

            # --- 5) 计算该对 patch 的 G1 损失 ---
            loss_ij = torch.mean((n1 - n2) ** 2)
            losses.append(loss_ij)

        loss = torch.stack(losses).mean()

        return loss * weight
    
    def coons_patch_full_(self, control: dict, u: torch.Tensor, v: torch.Tensor):
        """
        计算 Coons patch 的位置 S 以及偏导 Su, Sv。
        control: dict with keys 'b0','b1','b2','b3', each is Tensor[4,3]
        u, v: Tensor[K]
        返回:
        S:  Tensor[K,3]
        Su: Tensor[K,3]
        Sv: Tensor[K,3]
        """
        # 1) 四条边和它们的导数
        b0, db0 = self.bezier_point_and_derivative_(control['b0'], u)  # [K,3], [K,3]
        b2, db2 = self.bezier_point_and_derivative_(control['b2'], u)
        b3, db3 = self.bezier_point_and_derivative_(control['b3'], v)
        b1, db1 = self.bezier_point_and_derivative_(control['b1'], v)

        # 2) 取四个角点 (shape [3])
        p00 = control['b0'][0]   # (0,0)
        p30 = control['b0'][-1]  # (1,0)
        p33 = control['b2'][-1]  # (1,1)
        p03 = control['b2'][0]   # (0,1)

        # 3) 双线性插值项 BL，注意广播
        # 标量系数先扩展到 [K,1]，角点先扩展到 [1,3]
        coef00 = ((1-u)*(1-v)).unsqueeze(-1)    # [K,1]
        coef10 = (     u*(1-v)).unsqueeze(-1)
        coef11 = (     u*    v ).unsqueeze(-1)
        coef01 = (((1-u)*    v )).unsqueeze(-1)

        BL = (coef00 * p00.unsqueeze(0)
            + coef10 * p30.unsqueeze(0)
            + coef11 * p33.unsqueeze(0)
            + coef01 * p03.unsqueeze(0))      # [K,3]

        # 4) Coons patch 主公式 S(u,v)
        S = ((1-v).unsqueeze(-1)*b0
            +    v .unsqueeze(-1)*b2
            + (1-u).unsqueeze(-1)*b3
            +    u .unsqueeze(-1)*b1
            - BL)

        # 5) 计算偏导 ∂S/∂u 和 ∂S/∂v
        #    ∂BL/∂u = -(1-v)*p00 + (1-v)*p30 + v*p33 - v*p03
        dBL_du = (-(1-v).unsqueeze(-1)*p00.unsqueeze(0)
                + (1-v).unsqueeze(-1)*p30.unsqueeze(0)
                +    v .unsqueeze(-1)*p33.unsqueeze(0)
                -    v .unsqueeze(-1)*p03.unsqueeze(0))

        #    ∂BL/∂v = -(1-u)*p00 - u*p30 + u*p33 + (1-u)*p03
        dBL_dv = (-(1-u).unsqueeze(-1)*p00.unsqueeze(0)
                -    u .unsqueeze(-1)*p30.unsqueeze(0)
                +    u .unsqueeze(-1)*p33.unsqueeze(0)
                + (1-u).unsqueeze(-1)*p03.unsqueeze(0))

        Su = ( (1-v).unsqueeze(-1)*db0
            +    v .unsqueeze(-1)*db2
            - dBL_du )

        Sv = ( -b0
            + b2
            + (1-u).unsqueeze(-1)*db3
            +    u .unsqueeze(-1)*db1
            - dBL_dv )

        return S, Su, Sv
    
    def bezier_point_and_derivative_(self, cp: torch.Tensor, t: torch.Tensor):
        """
        计算 4 阶（cubic）Bezier 曲线在参数 t 处的点和导数。
        cp: Tensor[4,3]
        t: Tensor[K] 或标量
        返回:
        pts: Tensor[K,3]
        ders: Tensor[K,3]
        """
        # Bernstein basis
        B0 = (1 - t)**3
        B1 = 3 * t * (1 - t)**2
        B2 = 3 * t**2 * (1 - t)
        B3 = t**3
        pts = torch.matmul(torch.stack([B0, B1, B2, B3], -1), cp)  # [K,4] @ [4,3] -> [K,3]

        # 导数的 Bernstein basis
        dB0 = -3 * (1 - t)**2
        dB1 = 3 * (1 - t)**2 - 6 * t * (1 - t)
        dB2 = 6 * t * (1 - t) - 3 * t**2
        dB3 = 3 * t**2
        ders = torch.matmul(torch.stack([dB0, dB1, dB2, dB3], -1), cp)
        return pts, ders