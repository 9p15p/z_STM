#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/7/10 下午8:41
# @Author  : Merci
# @mail    : Merci@mail.dlut.edu.cn
# @File    : DepthCorr.py
from torch import nn
import torch.nn.functional as F

class DepthCorr(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size=3):
        super(DepthCorr, self).__init__()
        # adjust layer for asymmetrical features
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.conv_search = nn.Sequential(
            nn.Conv2d(in_channels, hidden, kernel_size=kernel_size, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, out_channels, kernel_size=1)
        )

    def conv2d_dw_group(self, x, kernel):
        batch, channel = kernel.shape[:2]
        x = x.view(1, batch * channel, x.size(2), x.size(3))  # 1 * (b*c) * k * k
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))  # (b*c) * 1 * H * W
        H, W = kernel.size(2), kernel.size(3)
        out = F.conv2d(x, kernel, groups=batch * channel, padding=(H // 2, W // 2))
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def forward_corr(self, kernel, input):
        kernel = self.conv_kernel(kernel)
        input = self.conv_search(input)
        feature = self.conv2d_dw_group(input, kernel)
        return feature

    def forward(self, kernel, search):
        feature = self.forward_corr(kernel, search)
        feature = F.interpolate(feature, size=(search.shape[2], search.shape[3]))
        out = self.head(feature)
        return out