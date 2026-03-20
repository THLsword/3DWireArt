import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange, repeat
from .backbones import APESSegBackbone
from .heads import APESSegHead, MLPHead

class APESSegmentor(nn.Module):
    def __init__(self):
        super(APESSegmentor, self).__init__()
        self.backbone = APESSegBackbone()
        self.head = APESSegHead()

    def forward(self, inputs: Tensor):
        x, x2, x3 = self.backbone(inputs)
        # x = self.head(x)
        return x, x2, x3
    
class Model(nn.Module):
    def __init__(self, template_params: torch.Tensor, apes_model: APESSegmentor):
        super(Model, self).__init__()
        self.apes = apes_model
        self.register_buffer('template_params', template_params) # (B, 122, 3)
        self.head = MLPHead(output_size = template_params.shape[1]*3)

    def forward(self, pcd):
        feature, self.ds_feature1, self.ds_feature2 = self.apes(pcd)
        output = self.head(feature)
        output = rearrange(output, 'B (M N) -> B M N', N = 3)
        output = self.template_params + output
        return output
