import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from einops import rearrange
import numbers


##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)



class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)




class Merge_Attention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Merge_Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.q_r = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=bias)
        self.q_d = nn.Conv2d(dim, dim // 2, kernel_size=1, bias=bias)
        self.k_f = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_f = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.conv_cat = nn.Conv2d(dim , dim, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y, z):
        assert x.shape == y.shape, 'The shape of feature maps from image and event branch are not equal!'
        b, c, h, w = x.shape

        q_r = self.q_r(x)  # rgb
        q_d = self.q_d(y)  # depth
        k_f = self.k_f(z)  # fusion_block
        v_f = self.v_f(z)  # fusion_block
        q = self.conv_cat(torch.cat([q_r , q_d],dim=1))
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_f = rearrange(k_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_f = rearrange(v_f, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

    
        q = torch.nn.functional.normalize(q, dim=-1)
        k_f = torch.nn.functional.normalize(k_f, dim=-1)

        attn = (q @ k_f.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = (attn @ v_f)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out = self.project_out(out) + z
        return out


class MaskAttentionTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor=2, bias=False, LayerNorm_type='WithBias'):
        super(MaskAttentionTransformerBlock, self).__init__()

        #self.norm1_x = LayerNorm(dim, LayerNorm_type)
        #self.norm1_y = LayerNorm(dim, LayerNorm_type)
        #self.norm1_z = LayerNorm(dim, LayerNorm_type)
        self.attn = Merge_Attention(dim, num_heads, bias)
        # mlp
        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * ffn_expansion_factor)
        #self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=nn.GELU, drop=0.)
        self.ffn = nn.Linear(dim, mlp_hidden_dim//2)
        
    def forward(self, x, y,z):
        b, c, h, w = x.shape
        #fused =  self.attn(self.norm1_x(x), self.norm1_y(y),self.norm1_z(z))  # b, c, h, w
        fused =  self.attn(x, y, z)  # b, c, h, w

        # mlp
        fused = to_3d(fused)  # b, h*w, c
        fused = fused + self.ffn(self.norm2(fused))
        x = fused.transpose(1,2).view(b, c, h, w)#  ->[B, H*W, C]->[B, C, H*W] 
        #fused = to_4d(fused, h, w)

        return x



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

