import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import warnings
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
import math
from mmcv.cnn.utils.weight_init import (constant_init, normal_init,
                                        trunc_normal_init)
from mmcv.runner import BaseModule
from collections import OrderedDict
from mmcv.cnn import build_norm_layer
from mmrotate.models.builder import ROTATED_BACKBONES


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation Block proposed in SENet (https://arxiv.org/abs/1709.01507)
    We assume the inputs to this layer are (N, C, H, W)
    """
    def __init__(self, input_channels, internal_neurons):
        super(SEBlock, self).__init__()
        self.down = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons,
                              kernel_size=1, stride=1, bias=True)
        self.up = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels,
                            kernel_size=1, stride=1, bias=True)
        self.input_channels = input_channels
        self.nonlinear = nn.ReLU(inplace=True)

    def forward(self, inputs):
        x = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x = self.down(x)
        x = self.nonlinear(x)
        x = self.up(x)
        x = torch.sigmoid(x)
        return inputs * x.view(-1, self.input_channels, 1, 1)




def Evalue_Conv(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("silu", nn.SiLU(inplace=True)),
    ]))

class HHFM(nn.Module):
    def __init__(self, dim, p_i:int, gamma:int):
        super().__init__()
        self.depth = p_i
        self.gamma = gamma
        self.srfconv = nn.Conv2d(dim, dim, kernel_size=3,padding=1)
        self.mrfconv = nn.Conv2d(dim, dim, kernel_size=5, dilation=1,padding=2)
        self.lrfconv = nn.Conv2d(dim, dim, kernel_size=7, dilation=3,padding=9)

        # self.srfconv = nn.Conv2d(dim, dim, kernel_size=3,padding=1,groups=dim)
        # self.mrfconv = nn.Conv2d(dim, dim, kernel_size=5,padding=2,groups=dim)
        # self.lrfconv = nn.Conv2d(dim, dim, kernel_size=7, dilation=3,padding=9,groups=dim)

        self.compress = dim // 2
        self.mid = self.compress * 3
        
        self.bottle = nn.Conv2d(dim, self.compress, 1)
        self.se = SEBlock(self.mid, self.mid // 4)
        self.weight_evauation = Evalue_Conv(self.mid, 3, 1)
        self.last_conv = nn.Conv2d(self.compress, dim, 1)
    
    def forward(self, x):
        #MFEB
        srf = self.srfconv(x)
        mrf = self.mrfconv(x)
        mid = (srf + mrf) / 2
        lrf = self.lrfconv(mid)

        #channel_compress
        srf_e = self.bottle(srf)
        mrf_e = self.bottle(mrf)
        lrf_e = self.bottle(lrf)

        #HFCB
        weight = self.weight_evauation(self.se(torch.cat([srf_e,mrf_e,lrf_e],dim=1)))
        weight = F.softmax(weight, dim=1)

        std = torch.std(weight, dim=1, keepdim=True)
        weight_decay_a = (self.gamma//2 - self.depth)
        weight_decay_b = (self.depth - self.gamma//2)

        weight_levels = weight.clone()
        for i in range(weight.size()[1]):
            weight_bias = weight_decay_a if i == 0 else weight_decay_b
            weight_levels[:, i:i+1, :, :] += (weight_bias*std)

        weight_new = F.leaky_relu(weight_levels).softmax(dim=1)

        w = (srf_e * weight_new[:, 0, :, :].unsqueeze(1) + mrf_e * weight_new[:, 1, :, :].unsqueeze(1) + lrf_e * weight_new[:, 2, :, :].unsqueeze(1))
        w = self.last_conv(w)

        return x * w


class Attention(nn.Module):
    def __init__(self, d_model,p_i = None,gamma = None):
        super().__init__()

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.gating = HSFM(d_model,p_i,gamma)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.gating(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x

    
class Block(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,drop_path=0., act_layer=nn.GELU, p_i = None, gamma = None,norm_cfg=None):
        super().__init__()
        if norm_cfg:
            self.norm1 = build_norm_layer(norm_cfg, dim)[1]
            self.norm2 = build_norm_layer(norm_cfg, dim)[1]
        else:
            self.norm1 = nn.BatchNorm2d(dim)
            self.norm2 = nn.BatchNorm2d(dim)
        self.gamma = gamma
        self.p_i = p_i

        self.attn = Attention(dim,self.p_i,self.gamma)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2            
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x


class Downsampling(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768, norm_cfg=None):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        if norm_cfg:
            self.norm = build_norm_layer(norm_cfg, embed_dim)[1]
        else:
            self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)        
        return x, H, W

@ROTATED_BACKBONES.register_module()
class HHFENet(BaseModule):
    def __init__(self, img_size=640, in_chans=3, embed_dims=[64, 128, 256, 512],out_indices=(0, 1, 2, 3),
                mlp_ratios=[8, 8, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=partial(nn.LayerNorm,eps = 1e-6),
                depths=[3, 4, 6, 3], num_stages=4, 
                pretrained=None,
                init_cfg=None,
                norm_cfg=None):
        super().__init__(init_cfg=init_cfg)

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be set at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is not None:
            raise TypeError('pretrained must be a str or None')
        
        # if flag == False:
        #     self.num_classes = num_classes

        self.depths = depths
        self.num_stages = num_stages
        self.out_indices = out_indices

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            gamma = sum(depths)
            p_i = sum(depths[:i])
            patch_embed = Downsampling(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i],norm_cfg=norm_cfg)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j],gamma=gamma,p_i=p_i + j + 1)
                for j in range(depths[i])])
            norm = norm_layer(embed_dims[i])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"block{i + 1}", block)
            setattr(self, f"norm{i + 1}", norm)

    def init_weights(self):
        print('init cfg', self.init_cfg)
        if self.init_cfg is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, val=1.0, bias=0.)
                elif isinstance(m, nn.Conv2d):
                    fan_out = m.kernel_size[0] * m.kernel_size[
                        1] * m.out_channels
                    fan_out //= m.groups
                    normal_init(
                        m, mean=0, std=math.sqrt(2.0 / fan_out), bias=0)
        else:
            super(HHFENet, self).init_weights()

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs=[]
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            block = getattr(self, f"block{i + 1}")
            norm = getattr(self, f"norm{i + 1}")
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.flatten(2).transpose(1, 2)
            x = norm(x)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            if i in self.out_indices:
                outs.append(x)
        return outs

    def forward(self, x):
        x = self.forward_features(x)
        #x = self.head(x)

        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x):
        x = self.dwconv(x)
        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict

