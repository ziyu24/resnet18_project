"""
Created on 11 17, 2024
@author: <Cui>
@bref: 获取手动构建的 resnet18_gpt 类
"""

from project.src.models.ResnetSelf_self.resnet18_constructor_self import ResNet18_self
from project.src.models.pretrained.pretrain_model import load_pretrained_model


def get_resnet18_gpt(num_classes=1000):
    return ResNet18_self(num_classes)


def get_pretrained_model_gpt(num_classes=1000):
    model_self = get_resnet18_gpt(num_classes=num_classes)
    return load_pretrained_model(model_self, num_classes=num_classes)


if __name__ == '__main__':
    model = get_pretrained_model_gpt()

    print(model)