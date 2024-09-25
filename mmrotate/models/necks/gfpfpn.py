# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
import torch
from mmcv.ops import DeformConv2d

from mmrotate.models.builder import ROTATED_NECKS

class GFRB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GFRB, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dwconv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, groups=self.in_channels)
        self.dwconv2 = nn.Conv2d(self.out_channels, self.out_channels, 1, groups=self.out_channels)
        self.dwconv3 = nn.Conv2d(self.out_channels, self.out_channels, 1, groups=self.out_channels)

        self.asym_conv1 = nn.Conv2d(self.out_channels, self.out_channels, (5, 1), padding=(2, 0))
        self.asym_conv2 = nn.Conv2d(self.out_channels, self.out_channels, (1, 5), padding=(0, 2))
        self.offset = nn.Conv2d(self.out_channels, 18, kernel_size=3,stride=1,padding=1)
        self.deform_conv = DeformConv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        identity = x
        dw_x = self.dwconv1(x)

        offset = self.offset(self.dwconv2(x))
        dfconv_x = self.deform_conv(x, offset)

        asym_x1 = self.asym_conv1(self.dwconv3(x))
        asym_x2 = self.asym_conv2(asym_x1)

        out = dw_x + dfconv_x + asym_x2 + identity

        return out

#downsample for x times.
class Downsample_x(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample_x, self).__init__()

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=scale_factor, kernel_size= scale_factor),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, ):
        x = self.downsample(x)

        return x
    
class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        )

    def forward(self, x, ):
        x = self.upsample(x)
        return x

class PFEB(nn.Module):
    def __init__(self, channels=256):
        super(PFEB, self).__init__()
        self.downsample_2 = Downsample_x(channels, channels, 2)
        self.downsample_4 = Downsample_x(channels, channels, 4)

        self.dwconv1 = nn.Conv2d(channels, channels, 1, groups=channels)

        self.dwconv_query = nn.Conv2d(channels, channels//8, 1)
        self.dwconv_key = nn.Conv2d(channels, channels//8, 1)
        self.dwconv_value = nn.Conv2d(channels, channels//8, 1)

        self.upsample_8 = Upsample(channels, channels, 8)

        self.last_conv = nn.Conv2d(channels//8, channels, 1)

    def forward(self, inputs):
        x1, x2, x3, x4 = inputs
        x2_resized = self.downsample_4(x2)
        x3_resized = self.downsample_2(x3)

        fusion_stage1 = x2_resized + x3_resized + x4
        fusion_stage1 = self.dwconv1(fusion_stage1)

        query = self.dwconv_query(self.upsample_8(fusion_stage1))
        key = self.dwconv_key(x1)
        value = self.dwconv_value(x1)
        

        attention_map = F.softmax(torch.matmul(query.view(query.size(0), -1, query.size(3)*query.size(2)).permute(0, 2, 1), \
                                               key.view(key.size(0), -1, key.size(3)*key.size(2))), dim=2)
        # Note: This is a generic fix and might need adjustment based on the actual dimensions of your tensors
        attention_output = torch.matmul(attention_map, value.view(value.size(0), value.size(2)*value.size(3), -1))
        attention_output = attention_output.permute(0, 2, 1).contiguous().view(value.size())

        attention_output = self.last_conv(attention_output)

        return attention_output + x1

@ROTATED_NECKS.register_module()
class GFPFPN(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(GFPFPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()
        self.fpn = FPN(in_channels, out_channels, num_outs, start_level, end_level, add_extra_convs, relu_before_extra_convs, no_norm_on_lateral, conv_cfg, norm_cfg, act_cfg, upsample_cfg, init_cfg)
        self.GFRB = nn.ModuleList()
        for i in range(self.num_ins):
            self.GFRB.append(GFRB(in_channels[i], in_channels[i]))
        self.ffaa = PFEB(out_channels)
    
    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        chan_num = len(self.in_channels)
        outputs = []
        for i in range(chan_num):
            outputs.append(self.GFRB[i](inputs[i]))
        out = self.fpn(tuple(outputs))
        out[-4] = self.ffaa(out[-4:])
        return tuple(out)

class FPN(BaseModule):
    r"""Feature Pyramid Network.

    This is an implementation of paper `Feature Pyramid Networks for Object
    Detection <https://arxiv.org/abs/1612.03144>`_.

    Args:
        in_channels (list[int]): Number of input channels per scale.
        out_channels (int): Number of output channels (used at each scale).
        num_outs (int): Number of output scales.
        start_level (int): Index of the start input backbone level used to
            build the feature pyramid. Default: 0.
        end_level (int): Index of the end input backbone level (exclusive) to
            build the feature pyramid. Default: -1, which means the last level.
        add_extra_convs (bool | str): If bool, it decides whether to add conv
            layers on top of the original feature maps. Default to False.
            If True, it is equivalent to `add_extra_convs='on_input'`.
            If str, it specifies the source feature map of the extra convs.
            Only the following options are allowed

            - 'on_input': Last feat map of neck inputs (i.e. backbone feature).
            - 'on_lateral': Last feature map after lateral convs.
            - 'on_output': The last output feature map after fpn convs.
        relu_before_extra_convs (bool): Whether to apply relu before the extra
            conv. Default: False.
        no_norm_on_lateral (bool): Whether to apply norm on lateral.
            Default: False.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Config dict for normalization layer. Default: None.
        act_cfg (dict): Config dict for activation layer in ConvModule.
            Default: None.
        upsample_cfg (dict): Config dict for interpolate layer.
            Default: dict(mode='nearest').
        init_cfg (dict or list[dict], optional): Initialization config dict.

    Example:
        >>> import torch
        >>> in_channels = [2, 3, 5, 7]
        >>> scales = [340, 170, 84, 43]
        >>> inputs = [torch.rand(1, c, s, s)
        ...           for c, s in zip(in_channels, scales)]
        >>> self = FPN(in_channels, 11, len(in_channels)).eval()
        >>> outputs = self.forward(inputs)
        >>> for i in range(len(outputs)):
        ...     print(f'outputs[{i}].shape = {outputs[i].shape}')
        outputs[0].shape = torch.Size([1, 11, 340, 340])
        outputs[1].shape = torch.Size([1, 11, 170, 170])
        outputs[2].shape = torch.Size([1, 11, 84, 84])
        outputs[3].shape = torch.Size([1, 11, 43, 43])
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__(init_cfg)
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                in_channels[i],
                out_channels,
                1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=act_cfg,
                inplace=False)
            fpn_conv = ConvModule(
                out_channels,
                out_channels,
                3,
                padding=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg,
                inplace=False)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                if i == 0 and self.add_extra_convs == 'on_input':
                    in_channels = self.in_channels[self.backbone_end_level - 1]
                else:
                    in_channels = out_channels
                extra_fpn_conv = ConvModule(
                    in_channels,
                    out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            if 'scale_factor' in self.upsample_cfg:
                # fix runtime error of "+=" inplace operation in PyTorch 1.10
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], **self.upsample_cfg)
            else:
                prev_shape = laterals[i - 1].shape[2:]
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=prev_shape, **self.upsample_cfg)

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return outs

