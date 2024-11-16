"""
Created on 11 15, 2024
@author: <Cui>
@bref: 获取手动构建的 resnet18 类
"""
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights

from project.src.common.config import config_yaml
from project.src.models.ResnetSelf_gpt.resnet18_constructor_gpt import ResNet18_gpt
from project.src.models.ResnetSelf_self.resnet18_constructor_self import ResNet18_self


def get_resnet18_self(num_classes=1000):
    return ResNet18_self(num_classes)


def get_pretrained_model(num_classes=1000):
    model_self = get_resnet18_self(num_classes=num_classes)
    return _load_pretrained_model(model_self, num_classes=num_classes)


def _load_pretrained_model(model_self, pretrained=True, num_classes=1000, freeze_backbone=True):
    """
    加载 torchvision 提供的预训练模型权重到自定义模型中，并根据需求进行调整。

    Parameters:
    - model (nn.Module): 自定义的模型（例如 ResNet18）
    - pretrained (bool): 是否加载预训练模型（默认为 True）
    - num_classes (int): 目标分类任务的类别数（默认为 1000，用于 ImageNet）
    - freeze_backbone (bool): 是否冻结特征提取部分（ResNet的卷积层等），默认为 True

    Returns:
    - model (nn.Module): 加载了预训练权重的模型
    """
    # 如果需要加载预训练模型
    if pretrained:
        # 加载 torchvision 提供的 ResNet18 模型
        pretrained_model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

        # 加载预训练模型的权重到自定义模型
        pretrained_dict = pretrained_model.state_dict()
        model_dict = model_self.state_dict()

        # 只加载与自定义模型结构匹配的层
        pretrained_dict = {k: v for k, v in pretrained_dict.items()
                           if k in model_dict and k != 'fc.weight' and k != 'fc.bias'}
        model_dict.update(pretrained_dict)
        model_self.load_state_dict(model_dict)
        print("Pretrained weights loaded into the model.")

        # 根据需要冻结卷积层（特征提取部分），只训练全连接层
        if freeze_backbone:
            for param in model_self.parameters():
                param.requires_grad = config_yaml['train']['not_freeze_conv']  # 是否冻结卷积层参数
            for param in model_self.fc.parameters():
                param.requires_grad = config_yaml['train']['not_freeze_fn']  # 是否冻结全连接层

    else:
        print("Training from scratch.")

    # 修改最后一层全连接层以适应新的类别数
    model_self.fc = nn.Linear(512, num_classes)

    return model_self


if __name__ == '__main__':
    model = get_pretrained_model()

    print(model)
