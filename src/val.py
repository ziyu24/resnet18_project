"""
Created on 11 15, 2024
@author: <Cui>
@bref: 评估训练的结果
"""

import torch
import matplotlib.pyplot as plt
from project.src.common.config import config_yaml


def evaluate(model, data_loader, loss_fn, device):
    model.eval()  # 设置模型为评估模式
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0

    with torch.no_grad():  # 评估时不需要计算梯度
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = running_loss / total_samples
    accuracy = correct_predictions / total_samples

    return avg_loss, accuracy
