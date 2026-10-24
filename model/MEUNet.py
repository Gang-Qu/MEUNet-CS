import torch
import torch.nn as nn
import einops
import torch.nn.functional as F
import torch.nn.init as init
from typing import Type, Callable, Tuple, Optional, Set, List, Union
import torch.utils.checkpoint as checkpoint
import numpy as np
from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from utils import *
from timm.models.layers import drop_path, trunc_normal_, Mlp, DropPath, to_2tuple
import torch.fft as fft
from timm.models.layers import trunc_normal_, DropPath
from functools import partial
from math import sqrt



def EuclideanProj(X, Y, H, W, HT, WT, mu):
    b, c, _, _ = X.shape
    Delta_Y = Y - torch.matmul(torch.matmul(H.repeat((b, c, 1, 1)), X), WT.repeat((b, c, 1, 1)))
    Delta_X = torch.matmul(torch.matmul(HT.repeat((b, c, 1, 1)), Delta_Y), W.repeat((b, c, 1, 1)))
    Z = X + torch.div(Delta_X, mu + 1)
    return Z

class AgentAttention(nn.Module):
    """
    这个模块 要求输入 的维度是 [B, N, C]  N=H*W
    所以，对于 [B, C, H, W]的张量，需要先转换 维度 ，再进行处理
    转换维度：
    from einops.einops import rearrange
    [B, C, H, W]->[B, H*W， C] : rearrange(x, 'b c h w -> b (h w) c')
    [B, H*W， C]->[B, C, H, W] : rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
    """
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=64, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        qq = 8
        self.num_patches = num_patches
        window_size = (int(num_patches ** 0.5), int(num_patches ** 0.5))
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num

        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3), padding=1, groups=dim)
        self.an_bias = nn.Parameter(torch.zeros(num_heads, agent_num, qq, qq))
        self.na_bias = nn.Parameter(torch.zeros(num_heads, agent_num, qq, qq))
        self.ah_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, window_size[0] // sr_ratio, 1))
        self.aw_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1, window_size[1] // sr_ratio))
        self.ha_bias = nn.Parameter(torch.zeros(1, num_heads, window_size[0], 1, agent_num))
        self.wa_bias = nn.Parameter(torch.zeros(1, num_heads, 1, window_size[1], agent_num))
        trunc_normal_(self.an_bias, std=.02)
        trunc_normal_(self.na_bias, std=.02)
        trunc_normal_(self.ah_bias, std=.02)
        trunc_normal_(self.aw_bias, std=.02)
        trunc_normal_(self.ha_bias, std=.02)
        trunc_normal_(self.wa_bias, std=.02)
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, y):

        b, n, c = x.shape
        # print('X:', x.shape)
        num_heads = self.num_heads
        head_dim = c // num_heads
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]

        agent_tokens = self.pool(q.reshape(b, H, W, c).permute(0, 3, 1, 2)).reshape(b, c, -1).permute(0, 2, 1)
        # print('aT:', agent_tokens.shape)
        # print('y:', y.shape)
        agent_tokens = y.reshape(b, c, -1).permute(0, 2, 1)
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(b, n // self.sr_ratio ** 2, num_heads, head_dim).permute(0, 2, 1, 3)
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)

        kv_size = (self.window_size[0] // self.sr_ratio, self.window_size[1] // self.sr_ratio)
        position_bias1 = nn.functional.interpolate(self.an_bias, size=kv_size, mode='bilinear')
        position_bias1 = position_bias1.reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias2 = (self.ah_bias + self.aw_bias).reshape(1, num_heads, self.agent_num, -1).repeat(b, 1, 1, 1)
        position_bias = position_bias1 + position_bias2
        agent_attn = self.softmax((agent_tokens * self.scale) @ k.transpose(-2, -1) + position_bias)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v

        agent_bias1 = nn.functional.interpolate(self.na_bias, size=self.window_size, mode='bilinear')
        agent_bias1 = agent_bias1.reshape(1, num_heads, self.agent_num, -1).permute(0, 1, 3, 2).repeat(b, 1, 1, 1)
        agent_bias2 = (self.ha_bias + self.wa_bias).reshape(1, num_heads, -1, self.agent_num).repeat(b, 1, 1, 1)
        agent_bias = agent_bias1 + agent_bias2
        q_attn = self.softmax((q * self.scale) @ agent_tokens.transpose(-2, -1) + agent_bias)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v

        x = x.transpose(1, 2).reshape(b, n, c)
        v = v.transpose(1, 2).reshape(b, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 3, 1, 2)
        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v, size=(H, W), mode='bilinear')
        x = x + self.dwc(v).permute(0, 2, 3, 1).reshape(b, n, c)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ChannelAttention2D(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias=False, qk_scale=None):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, (dim // 2) * 3, bias=qkv_bias)
        # self.proj = nn.Linear(dim // 2, dim)
        self.proj = nn.Conv2d(dim // 2, dim, kernel_size=1, bias=False)
        self.softmax = nn.Softmax(dim=-1)

        self.scale2 = nn.Parameter(1e-5 * torch.ones(1, dim//2, 1, 1))
        self.squeeze = nn.Conv2d(dim//2, 1, kernel_size=1, bias=False)  # mean cnannel

    def forward(self, x, h, w):
        x = einops.rearrange(x, "b c h w->  (h w) b c", h=h, w=w)
        B_, N, C = x.shape
        C = C // 2
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 4, 1)  ## (1, 4) channel weight  (4, 1) channel attention
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        out = rearrange(x, '(h w) b c -> b c h w',  h=h, w=w)
        out = out + self.scale2 * (out - F.gelu(self.squeeze(out)))
        x = self.proj(out)
        return x


class FFN(nn.Module):
    def __init__(self, dim, exp_ratio=1, bias=True):
        super().__init__()
        self.proj_in = nn.Conv2d(dim, int(dim * exp_ratio), kernel_size=1, bias=bias)
        self.act = nn.GELU()
        self.dwconv = nn.Conv2d(int(dim * exp_ratio), int(dim * exp_ratio), kernel_size=3, stride=1, padding=1,
                                groups=int(dim * exp_ratio), bias=bias)
        self.proj_out = nn.Conv2d(int(dim * exp_ratio), dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.act(self.proj_in(x))
        x = self.proj_out(self.dwconv(x))
        return x


class DCCABlock(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.scb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
        self.csab = ChannelAttention2D(dim, num_heads=4)
        self.ffn = FFN(dim)
        self.channel_interaction = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GroupNorm(4, dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )
        self.spatial_interaction = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GroupNorm(4, dim),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=1)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        scb_out = self.scb(x)
        scb_i = self.spatial_interaction(scb_out)

        csab_out = self.csab(x, h, w)

        csab_i = self.channel_interaction(csab_out)
        x1 = scb_out * torch.sigmoid(csab_i)
        x2 = csab_out * torch.sigmoid(scb_i)
        ffn_in = x1 + x2 + x
        ffn_out = self.ffn(ffn_in) + ffn_in
        return ffn_out


class ResDNetBlock(nn.Module):
    def __init__(self, dim, group_num, feature_size):
        super().__init__()
        self.dcca_list = nn.ModuleList()
        self.group_num = group_num
        group_dim = dim // group_num
        self.dense_conv = nn.ModuleList()
        for i in range(group_num):
            self.dcca_list.append(DCCABlock(group_dim))
            if i > 0:
                self.dense_conv.append(
                    nn.Sequential(
                        nn.Conv2d(group_dim * (i + 1), group_dim, 1),
                        nn.LeakyReLU(inplace=True)
                    )
                )
        self.last_conv = nn.Conv2d(1*dim, dim, 1)

        if feature_size[0] == 256:
            dim_hid = 64
            self.attn = AgentAttention(dim, feature_size[0]*feature_size[1])
            self.QK_encoder = Qcoder(1, dim_hid, dim_hid)
        elif feature_size[0] == 128:
            dim_hid = 128
            self.attn = AgentAttention(dim, feature_size[0]*feature_size[1])
            self.QK_encoder = Qcoder(1, dim_hid, dim_hid)
        elif feature_size[0] == 64:
            dim_hid = 256
            self.attn = AgentAttention(dim, feature_size[0]*feature_size[1])
            self.QK_encoder = Qcoder(1, dim_hid, dim_hid)

        # For CBSD68 Dataset
        # if meas_size[0] == 321 or meas_size[0] == 320:
        #     dim_hid = 64
        #     self.attn = AgentAttention(dim_hid, dim_hid // 4)
        #     self.QK_encoder = Qcoder(1, dim_hid, dim_hid)
        # elif meas_size[0] == 160:
        #     dim_hid = 128
        #     self.attn = AgentAttention(dim_hid, dim_hid // 4)
        #     self.QK_encoder = Qcoder(1, dim_hid, dim_hid)
        # elif meas_size[0] == 80:
        #     dim_hid = 256
        #     self.attn = AgentAttention(dim_hid, dim_hid // 4)
        #     self.QK_encoder = Qcoder(1, dim_hid, dim_hid)

    def forward(self, x, y):
        b, c, h, w = x.shape

        input_list = torch.chunk(x, chunks=self.group_num, dim=1)
        cf_in = input_list[0]
        out_list = []
        cf_out = self.dcca_list[0](cf_in)
        out_list.append(cf_out)
        Q = self.QK_encoder(y)
        for i in range(1, self.group_num):
            in_list = out_list.copy()
            in_list.append(input_list[i])
            cf_in = torch.cat(in_list, dim=1)
            cf_in = self.dense_conv[i - 1](cf_in)
            cf_out = self.dcca_list[i](cf_in)
            out_list.append(cf_out)
        out = torch.cat(out_list, dim=1)

        out = rearrange(out, 'b c h w -> b (h w) c')
        out1 = self.attn(out, h, w, Q)
        out = rearrange(out1, 'b (h w) c -> b c h w', h=h, w=w)
        out = self.last_conv(out)
        out = x + out
        return out


class CustomEncoder(nn.Module):
    def __init__(self, dim, group_num, num_blocks, img_size):
        super().__init__()
        self.blocks = nn.ModuleList([
            ResDNetBlock(dim=dim, group_num=group_num, feature_size=img_size) for _ in range(num_blocks)
        ])

    def forward(self, x, y):
        for block in self.blocks:
            x = block(x, y)
        return x


class BasicBlock(nn.Module):
    def __init__(self, color_channels=1, width=64, middle_blk_num=6, enc_blk_nums=[1, 2, 4], dec_blk_nums=[1, 1, 1],
                 meas_size=[256, 256]):
        super(BasicBlock, self).__init__()
        self.embedding = nn.Conv2d(in_channels=color_channels, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.encoders = nn.ModuleList()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.mapping = nn.Conv2d(in_channels=width, out_channels=color_channels, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        group_num = 4
        meas_size = meas_size
        for num in enc_blk_nums:
            self.convs1.append(
                nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
            )
            self.encoders.append(CustomEncoder(chan, group_num, num, meas_size))
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
            meas_size = [meas_size[0] // 2, meas_size[1] // 2]
        self.convs3 = nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
        self.middle_blks = CustomEncoder(chan, group_num, num, meas_size)

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            meas_size = [meas_size[0] * 2, meas_size[1] * 2]
            self.convs2.append(
                nn.Conv2d(chan * 2, chan, 1, 1, bias=False)
            )
            self.decoders.append(CustomEncoder(chan, group_num, num, meas_size))

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp, Y, last_decs=None, first_stage=False):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)
        x = self.embedding(inp)
        encs = []
        decs = []
        if not first_stage:
            for encoder, down, conv1, last_dec in zip(self.encoders, self.downs, self.convs1, last_decs[::-1]):
                x = conv1(torch.cat([x, last_dec], dim=1))
                x = encoder(x, Y)
                encs.append(x)
                x = down(x)
        else:
            for encoder, down in zip(self.encoders, self.downs):
                x = encoder(x, Y)
                encs.append(x)
                x = down(x)
        if not first_stage:
            x = self.convs3(torch.cat([x, last_decs[::-1][2]], dim=1))
            x = self.middle_blks(x, Y)
        else:
            x = self.middle_blks(x, Y)
        decs.append(x)
        for decoder, up, conv2, enc_skip in zip(self.decoders, self.ups, self.convs2, encs[::-1]):
            x = up(x)
            x = conv2(torch.cat([x, enc_skip], dim=1))
            x = decoder(x, Y)
            decs.append(x)
        x = self.mapping(x)
        x = x + inp
        return x[:, :, :H, :W], decs

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


def Guclidean_proj_vec(X, Y, H, Wg, mu=1.0):
    """
    向量化的投影梯度步 (分组共享版本)
    """
    X_flat = X.squeeze(1)
    Y_flat = Y.squeeze(1)
    # 核心: Gather操作 (同上)
    B_expanded = Wg  # 尺寸: (m, n, w)
    # 1. 计算预测值 Y_pred (向量化的前向传播)
    Y_pred_flat = torch.einsum('mh,bhw,mjw->bmj', H, X_flat, B_expanded)
    # 2. 计算残差
    Delta_Y_flat = Y_flat - Y_pred_flat
    # 3. 计算梯度项 (向量化的逆向传播)
    Delta_X_flat = torch.einsum('mh,bmj,mjw->bhw', H, Delta_Y_flat, B_expanded)
    # 更新Z
    Z = X + Delta_X_flat.unsqueeze(1) / (mu + 1)
    return Z


def F_AKCS_vec(X, H, Wg):
    """
    向量化的前向压缩测量 (分组共享版本)
    """
    X_flat = X.squeeze(1)  # (b, h, w)
    B_expanded = Wg  # 尺寸: (m, n, w)
    Y_flat = torch.einsum('mh,bhw,mjw->bmj', H, X_flat, B_expanded)
    return Y_flat.unsqueeze(1)


def I_AKCS_vec(Y, H, Wg):
    """
    向量化的逆向初始化 (分组共享版本)
    """
    Y_flat = Y.squeeze(1)  # (b, m, n)
    B_expanded = Wg  # 尺寸: (m, n, w)
    X_init_flat = torch.einsum('mh,bmj,mjw->bhw', H, Y_flat, B_expanded)
    return X_init_flat.unsqueeze(1)



def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class Qcoder(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(Qcoder, self).__init__()

        self.conv_in = conv_3x3_bn(dim_in, dim_hid, 11)  ## cr=25%: 16
        self.conv_Q = conv_1x1_bn(dim_hid, dim_out)

    def forward(self, x):
        x = self.conv_in(x)
        Q = self.conv_Q(x)
        return Q


class MEUNet(nn.Module):
    def __init__(self, imag_size, meas_size, img_channels=1, channels=64, mid_blocks=6, enc_blocks=[1, 2, 4],
                 dec_blocks=[1, 1, 1], stages=7, matrix_train=True, only_test=False):
        super().__init__()
        self.stages = stages
        self.imag_size = imag_size
        self.meas_size = meas_size
        self.only_test = only_test
        self.H = nn.Parameter(init.xavier_normal_(torch.Tensor(meas_size[0], imag_size[0])), requires_grad=True)
        self.Wg = nn.Parameter(init.xavier_normal_(torch.Tensor(meas_size[0], self.meas_size[1], imag_size[1])), requires_grad=True)
        self.mu = nn.Parameter(torch.Tensor([0.001]).repeat(stages))
        self.shared_denoiser = BasicBlock(img_channels, channels, mid_blocks, enc_blocks, dec_blocks, self.imag_size)

    def forward(self, X):
        """
        :input X: [b,256,256]
        """
        X_list = []
        X = X.unsqueeze(1)
        b, c, h, w = X.shape
        H = self.H
        Wg = self.Wg
        Y = F_AKCS_vec(X, H, Wg)
        X = I_AKCS_vec(Y, H, Wg)

        features = None 
        if b > 1:
            Z_list = []
            for j in range(b):
                X0 = X[j].unsqueeze(0)
                Y0 = Y[j].unsqueeze(0)
                for i in range(self.stages):
                    mu = self.mu[i]
                    Z = Guclidean_proj_vec(X0, Y0, H, Wg, mu)
                    X0, features = self.shared_denoiser(Z, Y0, features, first_stage=(i == 0))
                Z_list.append(X0)
            X = torch.stack(Z_list, dim=0).squeeze(1)
        else:
            for i in range(self.stages):
                mu = self.mu[i]
                Z = Guclidean_proj_vec(X, Y, H, Wg, mu)
                X, features = self.shared_denoiser(Z, Y, features, first_stage=(i == 0))
        X0 = X.squeeze(1)
        X_list.append(X0)
        return X_list, self.H, self.Wg
