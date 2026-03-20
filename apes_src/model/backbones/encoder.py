import torch
from torch import nn
from einops import pack
import os
import sys
# from ..utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample
sys.path.insert(0, os.path.abspath(
	os.path.join(os.path.dirname(__file__), '..')
))
# from utils.layers import Embedding, N2PAttention, GlobalDownSample, LocalDownSample, DownSample_new, UpSample
from einops import reduce, pack, repeat

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(64), nn.LeakyReLU(0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(128), nn.LeakyReLU(0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv7 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.conv8 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(512), nn.LeakyReLU(0.2))
        self.conv9 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))
        self.conv10 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(0.2))

        self.maxpool = nn.MaxPool2d(2, stride=2)
    def forward(self, x):
        x = self.conv1(x) # (B,1,256,256)  -> (B,64,256,256)
        x = self.conv2(x) # (B,64,256,256) -> (B,64,256,256)
        x = self.maxpool(x) # (B,64,256,256) -> (B,64,128,128)

        x = self.conv3(x) # (B,64,128,128) -> (B,128,128,128)
        x = self.conv4(x) # (B,128,128,128) -> (B,128,128,128)
        x = self.maxpool(x) # (B,256,128,128) -> (B,256,64,64)

        x = self.conv5(x) # (B,128,64,64) -> (B,256,64,64)
        x = self.conv6(x) # (B,256,64,64) -> (B,256,64,64)
        x = self.maxpool(x) # (B,256,64,64) -> (B,256,32,32)

        x = self.conv7(x) # (B,256,32,32) -> (B,512,32,32)
        x = self.conv8(x) # (B,512,32,32) -> (B,512,32,32)
        x = self.maxpool(x) # (B,512,32,32) -> (B,512,16,16)

        x = self.conv9(x) # (B,512,16,16) -> (B,256,16,16)
        x = self.conv10(x) # (B,256,16,16) -> (B,256,16,16)
        B, _, _, _ = x.shape
        x, _ = torch.max(x.view(B, 256, -1), dim=-1) # (B,256,16,16) -> (B,256)
        return x


class Decoder(nn.Module):
    def __init__(self, filename_obj, dim_in=512, centroid_scale=0.1, bias_scale=1.0, centroid_lr=0.1, bias_lr=1.0):
        super(Decoder, self).__init__()
        # load .obj
        self.template_mesh = sr.Mesh.from_obj(filename_obj)
        self.register_buffer('vertices_base', self.template_mesh.vertices.cpu()[0])  # vertices_base)
        self.register_buffer('faces', self.template_mesh.faces.cpu()[0])  # faces)

        self.nv = self.vertices_base.size(0)
        self.nf = self.faces.size(0)
        self.centroid_scale = centroid_scale
        self.bias_scale = bias_scale
        self.obj_scale = 0.5

        dim = 1024
        dim_hidden = [dim, dim*2]
        self.fc1 = nn.Linear(dim_in, dim_hidden[0])
        self.fc2 = nn.Linear(dim_hidden[0], dim_hidden[1])
        self.fc_centroid = nn.Linear(dim_hidden[1], 3)
        self.fc_bias = nn.Linear(dim_hidden[1], self.nv*3)

    def forward(self, x):
        batch_size = x.shape[0]
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)

        # decoder follows NMR
        centroid = self.fc_centroid(x) * self.centroid_scale

        bias = self.fc_bias(x) * self.bias_scale
        bias = bias.view(-1, self.nv, 3)

        base = self.vertices_base * self.obj_scale

        sign = torch.sign(base)
        base = torch.abs(base)
        base = torch.log(base / (1 - base))

        centroid = torch.tanh(centroid[:, None, :])
        scale_pos = 1 - centroid
        scale_neg = centroid + 1

        vertices = torch.sigmoid(base + bias) * sign
        vertices = F.relu(vertices) * scale_pos - F.relu(-vertices) * scale_neg
        vertices = vertices + centroid
        vertices = vertices * 0.5
        faces = self.faces[None, :, :].repeat(batch_size, 1, 1)

        return vertices, faces

if __name__ == '__main__':
    encoder = Encoder()
    input_image = torch.randn(1, 1, 256, 256)
    encoded_image = encoder(input_image)
    print(encoded_image.shape)