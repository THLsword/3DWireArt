from torch import nn
from einops import rearrange, repeat
import torch.nn.functional as F
import torch
import math
import sys
import os
sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')
))
from utils import ops, kmeans
from torch import nn
from einops import rearrange, repeat

class APESClsHead(nn.Module):
    def __init__(self):
        super(APESClsHead, self).__init__()
        self.linear1 = nn.Sequential(nn.Linear(3072, 1024), nn.BatchNorm1d(1024), nn.LeakyReLU(0.2))
        self.linear2 = nn.Sequential(nn.Linear(1024, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.linear3 = nn.Linear(256, 40)
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear1(x)  # (B, 3072) -> (B, 1024)
        x = self.dp1(x)  # (B, 1024) -> (B, 1024)
        x = self.linear2(x)  # (B, 1024) -> (B, 256)
        x = self.dp2(x)  # (B, 256) -> (B, 256)
        x = self.linear3(x)  # (B, 256) -> (B, 40)
        return x

class NewOut(nn.Module):
    def __init__(self):
        super(NewOut, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(6144, 4096), nn.BatchNorm1d(4096), nn.Sigmoid())
        self.fc2 = nn.Sequential(nn.Linear(4096, 2048), nn.BatchNorm1d(2048), nn.Sigmoid())
        self.fc3 = nn.Sequential(nn.Linear(2048, 2048), nn.Sigmoid())
        self.dp1 = nn.Dropout(0.5)
        self.dp2 = nn.Dropout(0.5)

    def forward(self, x):
        # (B, 3, 2048)
        x = rearrange(x,'B C N -> B (C N)')
        x = self.fc1(x)
        x = self.dp1(x)
        x = self.fc2(x)
        x = self.dp2(x)
        x = self.fc3(x)
        return x
    
    def maxmin_normalize(self, tensor):
        min_val = tensor.min(dim=1, keepdim=True).values
        max_val = tensor.max(dim=1, keepdim=True).values
        
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

    def modified_sigmoid(self, x):
        return 1 / (1 + torch.pow(20, -x))

class NewOut2(nn.Module):
    def __init__(self):
        super(NewOut2, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3073, 1024, 1, bias=False), nn.BatchNorm1d(1024), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv1d(1024, 256, 1, bias=False), nn.BatchNorm1d(256), nn.SiLU())
        self.conv3 = nn.Conv1d(256, 1, 1, bias=False)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)  # (B, 3073, 2048) -> (B, 1024, 2048)
        x = self.dp1(x)  # (B, 1024, 2048) -> (B, 1024, 2048)
        x = self.conv2(x)  # (B, 1024, 2048) -> (B, 256, 2048)
        x = self.dp2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv3(x).squeeze(1)  # (B, 256, 2048) -> (B, 1, 2048) -> (B, 2048)
        x = self.modified_sigmoid(x)
        return x
    
    def maxmin_normalize(self, tensor):
        min_val = tensor.min(dim=1, keepdim=True).values
        max_val = tensor.max(dim=1, keepdim=True).values
        
        normalized_tensor = (tensor - min_val) / (max_val - min_val)
        return normalized_tensor

    def modified_sigmoid(self, x):
        return 1 / (1 + torch.pow(20, -x))

class Feature_fusion(nn.Module):
    def __init__(self):
        super(Feature_fusion, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1024, 256, 1, bias=False), nn.BatchNorm1d(256), nn.SiLU())
        self.conv2 = nn.Sequential(nn.Conv1d(256, 64, 1, bias=False), nn.BatchNorm1d(64), nn.SiLU())
        self.conv3 = nn.Conv1d(64, 1, 1)
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

        self.q_conv = nn.Conv1d(1152, 1024, 1, bias=False)
        self.k_conv = nn.Conv1d(1024, 1024, 1, bias=False)
        self.v_conv = nn.Conv1d(1024, 1024, 1, bias=False)
        self.skip_link = nn.Conv1d(1152, 1024, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, pcd_f, img_f):
        q = self.q_conv(pcd_f)  # (B, 1152, N) -> (B, 1024, N)
        k = self.k_conv(img_f)  # (B, 1024, N) -> (B, 1024, N)
        v = self.v_conv(img_f)  # (B, 1024, N) -> (B, 1024, N)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, N) -> (B, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
        x = attention @ rearrange(v, 'B C N -> B N C').contiguous()  # (B, N, N) @ (B, N, C) -> (B, N, C)
        x = rearrange(x, 'B N C -> B C N').contiguous()  # (B, N, C) -> (B, C, N)
        x = self.skip_link(pcd_f) + x  # (B, C, N) + (B, C, N) -> (B, C, N)

        x = self.conv1(x)  # (B, 1024, 2048) -> (B, 256, 2048)
        # x = self.dp1(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv2(x)  # (B, 256, 2048) -> (B, 64, 2048)
        # x = self.dp2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv3(x).squeeze(1)  # (B, 64, 2048) -> (B, 1, 2048)
        x = self.modified_sigmoid(x)
        return x

    def modified_sigmoid(self, x):
        return 1 / (1 + torch.pow(20, -x))

class APESSegHead(nn.Module):
    def __init__(self):
        super(APESSegHead, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(1152, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 256, 1, bias=False), nn.BatchNorm1d(256), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(256, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv1d(128, 1, 1, bias=False), nn.BatchNorm1d(1))
        self.dp1 = nn.Dropout(p=0.5)
        self.dp2 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv1(x)  # (B, 1152, 2048) -> (B, 256, 2048)
        x = self.dp1(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.dp2(x)  # (B, 256, 2048) -> (B, 256, 2048)
        x = self.conv3(x)  # (B, 256, 2048) -> (B, 128, 2048)
        x = self.conv4(x)  # (B, 128, 2048) -> (B, 1, 2048)
        x = self.modified_sigmoid(x.squeeze(1))
        return x

    def modified_sigmoid(self, x):
        return 1 / (1 + torch.pow(20, -x))

class StdHead(nn.Module):
    def __init__(self):
        super(StdHead, self).__init__()
        self.K = 32  # number of neighbors
        self.group_type = 'diff'
        self.conv1 = nn.Sequential(nn.Conv1d(1152, 128, 1, bias=False), nn.BatchNorm1d(128), nn.LeakyReLU(0.2))
        self.q_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.k_conv = nn.Conv2d(128, 128, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.conv1(x) # (B,1152,2048) -> (B,128,2048)
        neighbors = ops.group(x, self.K, self.group_type)  # (B, C, N) -> (B, C, N, K)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = rearrange(q, 'B C N 1 -> B N 1 C').contiguous()  # (B, C, N, 1) -> (B, N, 1, C)
        k = self.k_conv(neighbors)  # (B, C, N, K) -> (B, C, N, K)
        k = rearrange(k, 'B C N K -> B N C K').contiguous()  # (B, C, N, K) -> (B, N, C, K)
        energy = q @ k  # (B, N, 1, C) @ (B, N, C, K) -> (B, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, N, 1, K) -> (B, N, 1, K)
        out = rearrange(torch.std(attention, dim=-1, unbiased=False), 'B N 1 -> B N').contiguous()  # (B, N, 1, K) -> (B, N, 1) -> (B, N)
        # normalize 
        mean = out.mean(dim=1, keepdim=True)
        std = out.std(dim=1, keepdim=True)
        normalized_tensor = (out - mean) / std
        output = self.modified_sigmoid(normalized_tensor)
        
        return output

    def modified_sigmoid(self, x):
        return 1 / (1 + torch.pow(20, -x))

class MLP_Head(nn.Module):
    def __init__(self):
        super(MLP_Head, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(3, 256, 1), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1, 1))
        
    def forward(self, x):
        x = self.conv1(x) # (B, 3, N) -> (B,256,N)
        x = self.conv2(x) # (B, 256, N) -> (B,128,N)
        x = self.conv3(x) # (B, 128, N) -> (B,1,N)
        x = F.sigmoid(x.squeeze(1))
        return x

    def modified_sigmoid(self, x):
        return 1 / (1 + torch.pow(20, -x))