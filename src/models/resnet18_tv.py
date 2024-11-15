"""
Created on 11 14, 2024
@author: <Cui>
@bref: 通过 torchvision 实现 ResNet18 类
"""

import torch
from torch import nn
from torchvision import models

from project.src.common.config import config_yaml


def get_resnet18_tv():
    """
    创建 resnet18 模型
    :return: 返回 resnet18 模型
    """

    model = models.resnet18(weights=None)
    # 修改最后一层（全连接层）输出为 23 类
    model.fc = nn.Linear(model.fc.in_features, config_yaml['data']['num_classes'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model
