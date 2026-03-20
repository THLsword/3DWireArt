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
from ..utils import ops, kmeans

class MLPHead(nn.Module):
    def __init__(self, output_size):
        super(MLPHead, self).__init__()
        self.output_size = output_size
        self.linear1 = nn.Sequential(nn.Linear(128, 256), nn.LeakyReLU(0.2))
        self.linear2 = nn.Linear(256, self.output_size)
        nn.init.zeros_(self.linear2.bias)
        nn.init.zeros_(self.linear2.weight)
        
    def forward(self, x):
        x, _ = x.max(dim = 2)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

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

