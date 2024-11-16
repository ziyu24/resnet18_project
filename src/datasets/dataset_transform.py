"""
Created on 11 14, 2024
@author: <Cui>
@brief: 创建 transform
"""

from torchvision.transforms import transforms

from project.src.common.config import config_yaml

norm_mean = [0.485, 0.456, 0.406]
norm_std = [0.229, 0.224, 0.225]

transform_train = transforms.Compose([
    transforms.Resize(config_yaml['data']['height']),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomCrop(config_yaml['data']['height'], padding=4),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

transform_val = transforms.Compose([
    transforms.Resize((config_yaml['data']['height'], config_yaml['data']['width'])),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
