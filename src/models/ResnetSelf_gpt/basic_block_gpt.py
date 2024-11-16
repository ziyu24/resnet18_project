"""
Created on 11 15, 2024
@author: <Cui>
@bref: 通过 torch.nn，手动实现残差块类
"""

import torch.nn as nn
import torch.nn.functional as F


# 定义残差块 (Residual Block)
class BasicBlock_gpt(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_gpt, self).__init__()

        # 第一层卷积，步幅为stride
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二层卷积
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            # 如果输入和输出的尺寸不匹配，则需要通过卷积来调整输入
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 先进行卷积、批归一化、ReLU激活，再通过第二个卷积、批归一化
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 将输入通过shortcut进行跳跃连接，最后加到输出上
        out += self.shortcut(x)

        # 激活函数ReLU
        out = F.relu(out)
        return out

if __name__ == '__main__':
    model = BasicBlock_gpt(3, 3)
    print(model)
