#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2024 user <user@adam>
#
# Distributed under terms of the MIT license.

"""
train simple
"""
import torch, torchvision
import torch.nn.functional as F
from models.torchvision import TorchVisionModel
from project.src.common.config_resnet18 import data_set_train_dir
from val import val
import os.path as osp

batch_size_train = 64
batch_size_val = 1000

from dataset import CustomImageFolder 

# dataset_root = '/home/user/data/Butterfly'
dataset_name_or_path = data_set_train_dir

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
train_loader = torch.utils.data.DataLoader(
    CustomImageFolder(osp.join(dataset_name_or_path, 'train'), transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
      ])),
    batch_size=batch_size_train, shuffle=True)

val_loader = torch.utils.data.DataLoader(
    CustomImageFolder(osp.join(dataset_name_or_path, 'valid'), transform=torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=mean, std=std)
    ])),
    batch_size=batch_size_val, shuffle=True)


# model
model = TorchVisionModel(base_encoder='resnet18', num_classes=100)
# model = vit_base_patch16_224(num_classes=100) 
# pretrain_path = 'weights/vit_base_patch16_224.pth'
# state_dict = torch.load(pretrain_path, map_location='cpu')
# # 删除分类头的权重
# if 'head.weight' in state_dict:
#     del state_dict['head.weight']
# if 'head.bias' in state_dict:
#     del state_dict['head.bias']
# model.load_state_dict(state_dict, strict=False)


device = torch.device("cpu")
model = model.to(device)

criterion = torch.nn.CrossEntropyLoss()

# optimizer schedule
optimizer = torch.optim.AdamW([
     dict(params=model.parameters(), lr=0.001),
 ])
#  scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 8, 15], gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

epochs = 16

def train(model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    for step, (imgs, targets) in enumerate(train_loader):
        imgs, targets = imgs.to(device), targets.to(device)

        optimizer.zero_grad()
        y = model(imgs)
        loss = criterion(y, targets)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, step * len(imgs), len(train_loader.dataset),
                100. * step / len(train_loader), loss.item()))



for epoch in range(epochs):
    print(f'Epoch: {epoch}')
    train(model, device, train_loader, optimizer, epoch, criterion)
    val(model, device, val_loader, epoch, criterion)
    scheduler.step()





