import math
import torch
from torch import nn
from torch.autograd import Variable

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import os
from TDEncoder import ICACompression, MultiNoduleEmbedding

class MLP(nn.Module):
    """nodule features come in (b n d) shape"""
    def __init__(self, num_feat, nod_dim, code_dim, feat_dim, num_classes) -> None:
        super().__init__()
        dim = num_feat*feat_dim
        self.img_embedding = MultiNoduleEmbedding(feat_dim=feat_dim, embedding_dim=dim)
        self.code_embedding = nn.Linear(code_dim, feat_dim)

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img, code):
        x_img = rearrange(self.img_embedding(img), 'b n d -> b (n d)')
        x_code = rearrange(self.code_embedding(code), 'b n d -> b (n d)')
        x = torch.cat((x_img, x_code), dim=1)
        return self.mlp(x)
        
class ICAMLP(nn.Module):
    """nodule features come in (b (n d)) shape"""
    def __init__(self, nod_dim, code_dim, feat_dim, num_classes) -> None:
        super().__init__()
        self.img_embedding = MultiNoduleEmbedding(feat_dim=feat_dim, embedding_dim=dim)
        self.code_embedding = ICACompression(in_channel=code_dim, out_channel=feat_dim)

        dim = feat_dim*2
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_classes),
        )

    def forward(self, img, code):
        x_img = self.img_embedding(img) # b d
        x_code = self.code_embedding(code) # b d
        x = torch.cat((x_img, x_code), dim=1)
        return self.mlp(x)
