import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.nn.functional as F
import torchvision


# 定义卷积block：1、卷积层；2、BN层；3、激活层
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self, inp):
        x = self.conv2d(inp)
        x = self.bn(x)
        out = F.relu(x)
        return out

# 定义Inception-block网络结构
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, pool_features):
        super(InceptionBlock, self).__init__()
        self.conv_1x1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.conv_3x3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.conv_3x3_2 = BasicConv2d(64, 128, kernel_size=3, padding=1)
        self.conv_5x5_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.conv_5x5_2 = BasicConv2d(64, 128, kernel_size=5, padding=2)
        self.pooling_2 = BasicConv2d(in_channels, pool_features, kernel_size=1)
    def forward(self, inp):
        out_branch1 = self.conv_1x1(inp)
        tmp_branch2 = self.conv_3x3_1(inp)
        out_branch2 = self.conv_3x3_2(tmp_branch2)
        tmp_branch3 = self.conv_5x5_1(inp)
        out_branch3 = self.conv_5x5_2(tmp_branch3)
        tmp_branch4 = F.max_pool2d(inp, kernel_size=3, stride=1, padding=1)
        out_branch4 = self.pooling_2(tmp_branch4)
        outputs = [out_branch1, out_branch2, out_branch3, out_branch4]

        return torch.cat(outputs, dim=1)

# 初始化
my_inception_block = InceptionBlock(32, 64)
print(my_inception_block)