import torch
from torch import nn
import torch.nn.functional as F
from functools import partial
import math

from timm.models.layers import trunc_normal_tf_
from timm.models.helpers import named_apply

__all__ = ['MK_UNet', 'MK_UNet_ViewBranch']


def gcd(a, b):
    while b:
        a, b = b, a % b
    return a


def _init_weights(module, name, scheme=''):
    if isinstance(module, nn.Conv2d):
        if scheme == 'normal':
            nn.init.normal_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'trunc_normal':
            trunc_normal_tf_(module.weight, std=.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'xavier_normal':
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif scheme == 'kaiming_normal':
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        else:
            fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
            fan_out //= module.groups
            nn.init.normal_(module.weight, 0, math.sqrt(2.0 / fan_out))
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)
    elif isinstance(module, nn.LayerNorm):
        nn.init.constant_(module.weight, 1)
        nn.init.constant_(module.bias, 0)


def act_layer(act, inplace=False, neg_slope=0.2, n_prelu=1):
    act = act.lower()
    if act == 'relu':
        layer = nn.ReLU(inplace)
    elif act == 'relu6':
        layer = nn.ReLU6(inplace)
    elif act == 'leakyrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    elif act == 'gelu':
        layer = nn.GELU()
    elif act == 'hswish':
        layer = nn.Hardswish(inplace)
    else:
        raise NotImplementedError(f'activation layer [{act}] is not found')
    return layer


def channel_shuffle(x, groups):
    batchsize, num_channels, height, width = x.data.size()
    channels_per_group = num_channels // groups
    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, height, width)
    return x


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes=None, ratio=16, activation='relu'):
        super(ChannelAttention, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes if out_planes is not None else in_planes
        if self.in_planes < ratio:
            ratio = self.in_planes
        self.reduced_channels = max(1, self.in_planes // ratio)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.activation = act_layer(activation, inplace=True)
        self.fc1 = nn.Conv2d(in_planes, self.reduced_channels, 1, bias=False)
        self.fc2 = nn.Conv2d(self.reduced_channels, self.out_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = self.fc2(self.activation(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.activation(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7, 11), 'kernel size must be 3 or 7 or 11'
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


class GroupedAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int, kernel_size=1, groups=1, activation='relu'):
        super(GroupedAttentionGate, self).__init__()
        if kernel_size == 1:
            groups = 1

        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=kernel_size, stride=1,
                      padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.activation = act_layer(activation, inplace=True)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.activation(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MultiKernelDepthwiseConv(nn.Module):
    def __init__(self, in_channels, kernel_sizes, stride, activation='relu6', dw_parallel=True):
        super(MultiKernelDepthwiseConv, self).__init__()
        self.in_channels = in_channels
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    self.in_channels, self.in_channels,
                    kernel_size, stride, kernel_size // 2,
                    groups=self.in_channels, bias=False
                ),
                nn.BatchNorm2d(self.in_channels),
                act_layer(activation, inplace=True)
            )
            for kernel_size in kernel_sizes
        ])
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        outputs = []
        for dwconv in self.dwconvs:
            dw_out = dwconv(x)
            outputs.append(dw_out)
            if not self.dw_parallel:
                x = x + dw_out
        return outputs


class MultiKernelInvertedResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride, expansion_factor=2,
                 dw_parallel=True, add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
        super(MultiKernelInvertedResidualBlock, self).__init__()
        assert stride in [1, 2]

        self.stride = stride
        self.in_c = in_c
        self.out_c = out_c
        self.kernel_sizes = kernel_sizes
        self.add = add
        self.n_scales = len(kernel_sizes)
        self.use_skip_connection = True if self.stride == 1 else False

        self.ex_c = int(self.in_c * expansion_factor)
        self.pconv1 = nn.Sequential(
            nn.Conv2d(self.in_c, self.ex_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.ex_c),
            act_layer(activation, inplace=True)
        )

        self.multi_scale_dwconv = MultiKernelDepthwiseConv(
            self.ex_c, self.kernel_sizes, self.stride, activation, dw_parallel=dw_parallel
        )

        if self.add:
            self.combined_channels = self.ex_c
        else:
            self.combined_channels = self.ex_c * self.n_scales

        self.pconv2 = nn.Sequential(
            nn.Conv2d(self.combined_channels, self.out_c, 1, 1, 0, bias=False),
            nn.BatchNorm2d(self.out_c),
        )

        if self.use_skip_connection and (self.in_c != self.out_c):
            self.conv1x1 = nn.Conv2d(self.in_c, self.out_c, 1, 1, 0, bias=False)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        pout1 = self.pconv1(x)
        dwconv_outs = self.multi_scale_dwconv(pout1)

        if self.add:
            dout = 0
            for dwout in dwconv_outs:
                dout = dout + dwout
        else:
            dout = torch.cat(dwconv_outs, dim=1)

        dout = channel_shuffle(dout, gcd(self.combined_channels, self.out_c))
        out = self.pconv2(dout)

        if self.use_skip_connection:
            if self.in_c != self.out_c:
                x = self.conv1x1(x)
            return x + out
        else:
            return out


def mk_irb_bottleneck(in_c, out_c, n, s, expansion_factor=2,
                      dw_parallel=True, add=True, kernel_sizes=[1, 3, 5], activation='relu6'):
    convs = []
    xx = MultiKernelInvertedResidualBlock(
        in_c, out_c, s,
        expansion_factor=expansion_factor,
        dw_parallel=dw_parallel,
        add=add,
        kernel_sizes=kernel_sizes,
        activation=activation
    )
    convs.append(xx)

    if n > 1:
        for _ in range(1, n):
            xx = MultiKernelInvertedResidualBlock(
                out_c, out_c, 1,
                expansion_factor=expansion_factor,
                dw_parallel=dw_parallel,
                add=add,
                kernel_sizes=kernel_sizes,
                activation=activation
            )
            convs.append(xx)

    conv = nn.Sequential(*convs)
    return conv


class TransformerBottleneck(nn.Module):
    def __init__(self, in_channels, num_heads=4, num_layers=1, mlp_ratio=4.0, dropout=0.1):
        super(TransformerBottleneck, self).__init__()
        assert in_channels % num_heads == 0, "in_channels 必须能被 num_heads 整除"

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_channels,
            nhead=num_heads,
            dim_feedforward=int(in_channels * mlp_ratio),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)
        return x


class SmallLesionEnhance(nn.Module):
    """
    小病灶增强模块：
    - 保持分辨率不变
    - 并行多感受野
    - 引入空洞卷积增强弱小灶响应
    - 残差输出，尽量不破坏原始特征
    """
    def __init__(self, channels, reduction=8):
        super().__init__()
        inter = max(channels // 2, 8)

        self.pre = nn.Sequential(
            nn.Conv2d(channels, inter, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter),
            nn.GELU()
        )

        self.branch1 = nn.Sequential(
            nn.Conv2d(inter, inter, kernel_size=3, padding=1, groups=inter, bias=False),
            nn.BatchNorm2d(inter),
            nn.GELU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(inter, inter, kernel_size=5, padding=2, groups=inter, bias=False),
            nn.BatchNorm2d(inter),
            nn.GELU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(inter, inter, kernel_size=3, padding=2, dilation=2, groups=inter, bias=False),
            nn.BatchNorm2d(inter),
            nn.GELU()
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(inter * 3, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )

        self.ca = ChannelAttention(channels, ratio=max(4, reduction))
        self.sa = SpatialAttention(kernel_size=7)

        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        x0 = self.pre(x)
        b1 = self.branch1(x0)
        b2 = self.branch2(x0)
        b3 = self.branch3(x0)
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.fuse(out)
        out = self.ca(out) * out
        out = self.sa(out) * out
        return x + self.res_scale * out


class DetailFuseBlock(nn.Module):
    """
    用于浅层skip特征与解码特征融合时，进一步保留细节
    """
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU()
        )
        self.ca = ChannelAttention(channels, ratio=8)
        self.sa = SpatialAttention(kernel_size=7)
        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def forward(self, x):
        out = self.conv(x)
        out = self.ca(out) * out
        out = self.sa(out) * out
        return x + out


class MK_UNet(nn.Module):
    def __init__(self, num_classes=1, in_channels=3,
                 channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 gag_kernel=3,
                 **kwargs):
        super().__init__()

        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)

        self.sle1 = SmallLesionEnhance(channels[0])
        self.sle2 = SmallLesionEnhance(channels[1])
        self.detail_fuse_t1 = DetailFuseBlock(channels[0])
        self.detail_fuse_t2 = DetailFuseBlock(channels[1])

        self.trans_bottleneck = TransformerBottleneck(
            in_channels=channels[4],
            num_heads=4,
            num_layers=1,
            mlp_ratio=4.0,
            dropout=0.1
        )

        self.AG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
                                        kernel_size=gag_kernel, groups=max(1, channels[3] // 2))
        self.AG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
                                        kernel_size=gag_kernel, groups=max(1, channels[2] // 2))
        self.AG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
                                        kernel_size=gag_kernel, groups=max(1, channels[1] // 2))
        self.AG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2,
                                        kernel_size=gag_kernel, groups=max(1, channels[0] // 2))

        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)

        self.SA = SpatialAttention()
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        out = self.encoder1(x)
        out = self.sle1(out)
        t1 = F.max_pool2d(out, 2, 2)

        out = self.encoder2(t1)
        out = self.sle2(out)
        t2 = F.max_pool2d(out, 2, 2)

        out = F.max_pool2d(self.encoder3(t2), 2, 2)
        t3 = out

        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out

        out = F.max_pool2d(self.encoder5(out), 2, 2)
        out = self.trans_bottleneck(out)

        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False))
        t4 = self.AG1(g=out, x=t4)
        out = out + t4

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False))
        t3 = self.AG2(g=out, x=t3)
        out = out + t3

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False))
        t2 = self.detail_fuse_t2(self.AG3(g=out, x=t2))
        out = out + t2

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear', align_corners=False))
        t1 = self.detail_fuse_t1(self.AG4(g=out, x=t1))
        out = out + t1

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear', align_corners=False))

        return self.out4(out)


class MK_UNet_ViewBranch(nn.Module):
    """
    共享主干 + Transformer + view embedding + 小病灶增强
    输入方式保持不变：
        forward(x, view)
    view:
        0 -> anterior
        1 -> posterior
    """

    def __init__(self, num_classes=1, in_channels=3,
                 channels=[16, 32, 64, 96, 160],
                 depths=[1, 1, 1, 1, 1],
                 kernel_sizes=[1, 3, 5],
                 expansion_factor=2,
                 gag_kernel=3,
                 view_embed_dim=32,
                 view_scale=0.1,
                 **kwargs):
        super().__init__()

        self.view_scale = view_scale

        self.encoder1 = mk_irb_bottleneck(in_channels, channels[0], depths[0], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.encoder2 = mk_irb_bottleneck(channels[0], channels[1], depths[1], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.encoder3 = mk_irb_bottleneck(channels[1], channels[2], depths[2], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.encoder4 = mk_irb_bottleneck(channels[2], channels[3], depths[3], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.encoder5 = mk_irb_bottleneck(channels[3], channels[4], depths[4], 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)

        # 小病灶增强：浅层和高分辨率skip特征
        self.sle1 = SmallLesionEnhance(channels[0])
        self.sle2 = SmallLesionEnhance(channels[1])
        self.detail_fuse_t1 = DetailFuseBlock(channels[0])
        self.detail_fuse_t2 = DetailFuseBlock(channels[1])

        self.trans_bottleneck = TransformerBottleneck(
            in_channels=channels[4],
            num_heads=4,
            num_layers=1,
            mlp_ratio=4.0,
            dropout=0.1
        )

        self.view_embed = nn.Embedding(2, view_embed_dim)
        self.view_proj = nn.Sequential(
            nn.Linear(view_embed_dim, channels[4]),
            nn.GELU(),
            nn.Linear(channels[4], channels[4])
        )

        self.AG1 = GroupedAttentionGate(F_g=channels[3], F_l=channels[3], F_int=channels[3] // 2,
                                        kernel_size=gag_kernel, groups=max(1, channels[3] // 2))
        self.AG2 = GroupedAttentionGate(F_g=channels[2], F_l=channels[2], F_int=channels[2] // 2,
                                        kernel_size=gag_kernel, groups=max(1, channels[2] // 2))
        self.AG3 = GroupedAttentionGate(F_g=channels[1], F_l=channels[1], F_int=channels[1] // 2,
                                        kernel_size=gag_kernel, groups=max(1, channels[1] // 2))
        self.AG4 = GroupedAttentionGate(F_g=channels[0], F_l=channels[0], F_int=channels[0] // 2,
                                        kernel_size=gag_kernel, groups=max(1, channels[0] // 2))

        self.decoder1 = mk_irb_bottleneck(channels[4], channels[3], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.decoder2 = mk_irb_bottleneck(channels[3], channels[2], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.decoder3 = mk_irb_bottleneck(channels[2], channels[1], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.decoder4 = mk_irb_bottleneck(channels[1], channels[0], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)
        self.decoder5 = mk_irb_bottleneck(channels[0], channels[0], 1, 1,
                                          expansion_factor=expansion_factor, dw_parallel=True,
                                          add=True, kernel_sizes=kernel_sizes)

        self.CA1 = ChannelAttention(channels[4], ratio=16)
        self.CA2 = ChannelAttention(channels[3], ratio=16)
        self.CA3 = ChannelAttention(channels[2], ratio=16)
        self.CA4 = ChannelAttention(channels[1], ratio=8)
        self.CA5 = ChannelAttention(channels[0], ratio=4)

        self.SA = SpatialAttention()
        self.out4 = nn.Conv2d(channels[0], num_classes, kernel_size=1)

        self.init_weights('normal')

    def init_weights(self, scheme=''):
        named_apply(partial(_init_weights, scheme=scheme), self)

    def _normalize_view(self, view, batch_size, device):
        if isinstance(view, int):
            return torch.full((batch_size,), view, device=device, dtype=torch.long)

        if isinstance(view, list):
            view = torch.tensor(view, device=device, dtype=torch.long)

        if torch.is_tensor(view):
            view = view.to(device).long().view(-1)
            if view.numel() == 1 and batch_size > 1:
                view = view.repeat(batch_size)
            if view.numel() != batch_size:
                raise ValueError(f"view 的 batch 维度不匹配: got {view.numel()}, expected {batch_size}")
            return view

        raise TypeError("view 必须是 int、list 或 torch.Tensor")

    def forward(self, x, view):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        b = x.shape[0]
        device = x.device
        view = self._normalize_view(view, b, device)

        # 编码 + 浅层小病灶增强
        out = self.encoder1(x)
        out = self.sle1(out)
        t1 = F.max_pool2d(out, 2, 2)

        out = self.encoder2(t1)
        out = self.sle2(out)
        t2 = F.max_pool2d(out, 2, 2)

        out = F.max_pool2d(self.encoder3(t2), 2, 2)
        t3 = out

        out = F.max_pool2d(self.encoder4(out), 2, 2)
        t4 = out

        out = F.max_pool2d(self.encoder5(out), 2, 2)

        # Transformer bottleneck
        out = self.trans_bottleneck(out)

        # view embedding 注入
        v = self.view_embed(view)
        v = self.view_proj(v)
        v = v.unsqueeze(-1).unsqueeze(-1)
        out = out + self.view_scale * v

        # 解码
        out = self.CA1(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder1(out), scale_factor=2, mode='bilinear', align_corners=False))
        t4 = self.AG1(g=out, x=t4)
        out = out + t4

        out = self.CA2(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder2(out), scale_factor=2, mode='bilinear', align_corners=False))
        t3 = self.AG2(g=out, x=t3)
        out = out + t3

        out = self.CA3(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder3(out), scale_factor=2, mode='bilinear', align_corners=False))
        t2 = self.detail_fuse_t2(self.AG3(g=out, x=t2))
        out = out + t2

        out = self.CA4(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder4(out), scale_factor=2, mode='bilinear', align_corners=False))
        t1 = self.detail_fuse_t1(self.AG4(g=out, x=t1))
        out = out + t1

        out = self.CA5(out) * out
        out = self.SA(out) * out
        out = F.relu(F.interpolate(self.decoder5(out), scale_factor=2, mode='bilinear', align_corners=False))

        p4 = self.out4(out)
        return p4