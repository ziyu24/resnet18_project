"""
Created on 11 15, 2024
@author: <Cui>
@bref: 通过 torch.nn，手动构建ResNet18 类
"""

import torch
from torch import nn
import torch.nn.functional as F

from project.src.models.resnetSelf_gpt.basic_block_gpt import BasicBlock_gpt


# 定义 ResNet-18 网络
class ResNet18_gpt(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet18_gpt, self).__init__()

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
        layers.append(BasicBlock_gpt(in_channels, out_channels, stride))  # 第一块需要改变输入通道数或者步幅
        for _ in range(1, num_blocks):
            layers.append(BasicBlock_gpt(out_channels, out_channels))  # 后续的块只需要改变通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        # print("x============input: {}".format(x.shape))
        # 初始的卷积和池化层
        x = F.relu(self.bn1(self.conv1(x)))
        # print("x============conv1{}".format(x.shape))
        x = self.maxpool(x)
        # print("x============maxpool{}".format(x.shape))

        # 通过各个残差块层
        x = self.layer1(x)
        # print("x============layer1{}".format(x.shape))
        x = self.layer2(x)
        # print("x============layer2{}".format(x.shape))
        x = self.layer3(x)
        # print("x============layer3{}".format(x.shape))
        x = self.layer4(x)

        # print("x============layer4:{}".format(x.shape))
        # print("x============layer4:{}".format(x))
        # 全局平均池化
        x = self.avg_pool(x)
        # print("x============ave_pool:{}".format(x.shape))
        # print("x============ave_pool:{}".format(x))
        x = torch.flatten(x, 1)  # 展平成一维向量
        # print("x============flatten:{}".format(x.shape))
        # print("x============flatten:{}".format(x))

        # 全连接层
        x = self.fc(x)
        # print("x============fc:{}".format(x.shape))
        # print("x============fc:{}".format(x))
        return x


if __name__ == '__main__':
    # 创建 ResNet-18 模型
    model = ResNet18_gpt(num_classes=1000)

    # 打印模型架构
    print(model)
