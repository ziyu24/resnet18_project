"""
Created on 11 15, 2024
@author: <Cui>
@bref: 评估训练的结果
"""

import torch
from torch import nn

from project.src.common.config import config_yaml
from project.src.datasets.data_loader import get_data_loader_val
from project.src.models.resnet18_tv import get_resnet18_tv


def evaluate():
    model = get_resnet18_tv()
    model_path = config_yaml['train_model_save_path']
    test_loader = get_data_loader_val()
    loss_fn = nn.CrossEntropyLoss()

    evaluate_model(model, model_path, test_loader, loss_fn)


def evaluate_model(model, model_path, test_loader, loss_fn):
    # 加载 state_dict 并与模型结构结合
    state_dict = torch.load(model_path)
    incompatible_keys = model.load_state_dict(state_dict, strict=True)  # 加载参数

    # 打印任何不匹配的键
    if incompatible_keys.missing_keys or incompatible_keys.unexpected_keys:
        print("Missing keys:", incompatible_keys.missing_keys)
        print("Unexpected keys:", incompatible_keys.unexpected_keys)

    # 继续设置模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)  # 将模型移动到设备
    evaluate_one_epoch(model, test_loader, loss_fn, device, 1)


def evaluate_one_epoch(model, data_loader, loss_fn, device, epoch):
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

    # 打印当前的验证损失和精度
    print(f"evaluate model, Epoch [{epoch}], "
          f"Validation Loss: {avg_loss:.4f}, Validation Accuracy: {accuracy * 100:.4f}%")

    return avg_loss, accuracy
