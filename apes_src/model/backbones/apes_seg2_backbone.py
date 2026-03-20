import torch
from torch import nn
from einops import pack
import os
import sys
# from ..utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample
sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')
))
from utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample, DownSample_new, UpSample
from einops import reduce, pack, repeat

class APESSeg2Backbone(nn.Module):
    def __init__(self, which_ds):
        super(APESSeg2Backbone, self).__init__()
        self.embedding = Embedding()
        if which_ds == 'global':
            self.ds1 = GlobalDownSample(1024)  # 2048 pts -> 1024 pts
            self.ds2 = GlobalDownSample(512)  # 1024 pts -> 512 pts
            # self.ds3 = GlobalDownSample(256)  # 512 pts -> 256 pts
        elif which_ds == 'local':
            self.ds1 = LocalDownSample(1024)  # 2048 pts -> 1024 pts
            self.ds2 = LocalDownSample(512)  # 1024 pts -> 512 pts
        else:
            raise NotImplementedError
        self.n2p_attention1 = N2PAttention()
        self.n2p_attention2 = N2PAttention()
        self.n2p_attention3 = N2PAttention()
        self.n2p_attention4 = N2PAttention()
        self.n2p_attention5 = N2PAttention()
        self.ups1 = UpSample()
        self.ups2 = UpSample()
        self.conv1 = nn.Sequential(nn.Conv1d(256, 3, 1, bias=False), nn.BatchNorm1d(3), nn.LeakyReLU(0.2))

    def forward(self, x):
        tmp = self.embedding(x)  # (B, 3, 2048) -> (B, 128, 2048)
        d1 = self.n2p_attention1(tmp)  # (B, 128, 2048) -> (B, 128, 2048)
        tmp = self.ds1(d1)  # (B, 128, 2048) -> (B, 128, 1024)
        d2 = self.n2p_attention2(tmp)  # (B, 128, 1024) -> (B, 128, 1024)
        tmp = self.ds2(d2)  # (B, 128, 1024) -> (B, 128, 512)
        d3 = self.n2p_attention3(tmp)  # (B, 128, 512) -> (B, 128, 512)

        tmp = self.ups2(d2, d3)  # (B, 128, 512) -> (B, 128, 1024)
        up2 = self.n2p_attention4(tmp)  # (B, 128, 1024) -> (B, 128, 1024)
        tmp = self.ups1(d1, up2)  # (B, 128, 1024) -> (B, 128, 2048)
        up1 = self.n2p_attention5(tmp)  # (B, 128, 2048) -> (B, 128, 2048)
        up1 = torch.cat((d1,up1),dim=1) # (B, 128, 2048)*2 -> (B, 256, 2048)
        x = self.conv1(up1)  # (B, 128, 2048) -> (B, 3, 2048)

        # x = reduce(x, 'B C N -> B C', 'max')  # (B, 1024, 2048) -> (B, 1024)
        # x = x.unsqueeze(2).repeat(1,1,2048) # (B, 1024) -> (B, 1024, 2048)
        # x, _ = pack([x, x1], 'B * N')  # (B, 1024, 2048) -> (B, 1152, 2048)

        return x # (B, 3, 2048)