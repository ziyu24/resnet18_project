"""
Created on 11 16, 2024
@author: <Cui>
@bref: 手动构建 resnet18 类
"""

from torch import nn, flatten
import torch.nn.functional as F

from project.src.models.ResnetSelf_self.basic_block_self import BasicBlock_self


class ResNet18_self(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64)
        self.layer2 = self._make_layer(64, 128, 2)
        self.layer3 = self._make_layer(128, 256, 2)
        self.layer4 = self._make_layer(256, 512, 2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride=1, num_blocks=2):
        layer_list = []

        layer_list.append(BasicBlock_self(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layer_list.append(BasicBlock_self(out_channels, out_channels))

        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        x = self.max_pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)

        x = flatten(x, 1)

        x = self.fc(x)

        return x


if __name__ == '__main__':
    model = ResNet18_self()
    print(model)
