"""
Created on 11 15, 2024
@author: <Cui>
@bref: 通过 torch.nn，手动构建ResNet18 类
"""

import torch
from torch import nn
import torch.nn.functional as F

from project.src.models.ResnetSelf.basic_block import BasicBlock


# 定义 ResNet-18 网络
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18, self).__init__()

        # 初始卷积层 + 批归一化 + 最大池化
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # (3x224x224) -> (64x112x112)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # (64x112x112) -> (64x56x56)

        # 四个阶段，每个阶段包含多个残差块
        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # (512x7x7) -> (512x1x1)

        # 全连接层，用于分类
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        # 用于构建每一层的残差块
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))  # 第一块需要改变输入通道数或者步幅
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels))  # 后续的块只需要改变通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始的卷积和池化层
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        # 通过各个残差块层
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 全局平均池化
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)  # 展平成一维向量

        # 全连接层
        x = self.fc(x)
        return x


if __name__ == '__main__':
    # 创建 ResNet-18 模型
    model = ResNet18(num_classes=1000)

    # 打印模型架构
    print(model)
