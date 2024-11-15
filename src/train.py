"""
Created on 11 13, 2024
@author: <Cui>
@bref: 模型训练操作，推理，可视化等操作
"""

import torch
from torch import nn

from project.src.common.config import config_yaml
from project.src.val import evaluate


def train(model, train_loader, val_loader,
          use_save_model=config_yaml['train']['use_save_model'],
          num_epochs=config_yaml['train']['num_epochs'],
          lr=config_yaml['optimizer']['lr'],
          val_model_save_path=config_yaml['val_model_save_path'],
          val_check_point_save_path=config_yaml['val_check_point_save_path']):
    """
    训练函数: 训练一个模型
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs, use_save_model,
                val_model_save_path, val_check_point_save_path)


def train_model(model, train_loader, val_loader, optimizer, loss_fn, device, num_epochs, use_save_model,
                val_model_save_path, val_check_point_save_path):
    # 如果检查点存在，加载它
    if use_save_model:
        print("Load save model and train, check point path is: {}".format(config_yaml['train_check_point_save_path']))
        try:
            model, optimizer, start_epoch, best_loss = load_checkpoint(model, optimizer,
                                                                       config_yaml['train_check_point_save_path'])
        except FileNotFoundError:
            print("No checkpoint found, path is: {}, "
                  "starting fresh.".format(config_yaml['train_check_point_save_path']))
            start_epoch = 0
            best_loss = float('inf')
    else:
        print("No use save model, begin new model train, "
              "check point path is: {}".format(config_yaml['train_check_point_save_path']))
        start_epoch = 0
        best_loss = float('inf')

    for epoch in range(start_epoch, num_epochs):
        # 训练一个 epoch
        train_one_epoch(model, optimizer, train_loader, loss_fn, device, epoch)

        # 计算验证集损失
        val_loss, val_accuracy = evaluate(model, val_loader, loss_fn, device)

        # 达到需求精度时，保存模型和检查点，退出训练
        if val_accuracy > config_yaml['train']['preference_accuracy']:
            print(f"model evaluate Accuracy: {val_accuracy}, "
                  f"preference Accuracy: {config_yaml['train']['preference_accuracy']}, stop train")
            save_model(model, config_yaml['val_model_save_path'], "preference_model")
            save_checkpoint(model, optimizer, epoch, val_loss, config_yaml['val_check_point_save_path'],
                            "preference_checkpoint")
            return

        # 打印当前的验证损失和精度
        print(f"evaluate model, Epoch [{epoch}], "
              f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.4f}%")

        # 如果验证损失是最佳的，保存模型
        print(f"evaluate model, Epoch [{epoch}], val_loss[{val_loss}], best_loss[{best_loss}]")
        if val_loss < best_loss:
            best_loss = val_loss
            print(f"New best validation loss: {val_loss:.4f}, saved checkpoint and model")
            save_model(model, config_yaml['val_model_save_path'], "val_model")
            save_checkpoint(model, optimizer, epoch, val_loss, config_yaml['val_check_point_save_path'],
                            "val_checkpoint")

        # 每个 epoch 保存训练的模型
        if config_yaml['train']['save_model'] and (((epoch + 1) % config_yaml['train']['save_model_freq']) == 0):
            save_model(model, config_yaml['train_model_save_path'], "train_model")

        # 每个 epoch 保存训练的检查点
        if (config_yaml['train']['save_check_point'] and
                (((epoch + 1) % config_yaml['train']['save_check_point_freq']) == 0)):
            save_checkpoint(model, optimizer, epoch, val_loss,
                            config_yaml['train_check_point_save_path'], "train_checkpoint")


def train_one_epoch(model, optimizer, data_loader, loss_fn, device, epoch):
    """
    训练一个 epoch 的函数
    :param model: PyTorch 模型
    :param optimizer: 优化器（如 SGD 或 Adam）
    :param data_loader: DataLoader，训练数据集
    :param loss_fn: 损失函数（如 CrossEntropyLoss）
    :param device: 设备（例如 "cuda" 或 "cpu"）
    :param epoch: 当前训练轮次
    :return: 返回当前 epoch 的平均训练损失和精度
    """
    model.train()  # 设置模型为训练模式
    running_loss = 0.0  # 用于统计总损失
    correct_predictions = 0  # 用于统计正确预测的数量
    total_samples = 0  # 用于统计总样本数
    total_epochs = config_yaml['train']['num_epochs']

    for i, (inputs, labels) in enumerate(data_loader):
        inputs, labels = inputs.to(device), labels.to(device)  # 将数据移到指定设备上

        # 清除上一步的梯度
        optimizer.zero_grad()
        # 前向传播：通过模型获取输出
        outputs = model(inputs)
        # 计算损失
        loss = loss_fn(outputs, labels)
        # 反向传播：计算梯度
        loss.backward()
        # 优化器更新模型参数
        optimizer.step()

        # 统计损失
        running_loss += loss.item() * inputs.size(0)  # 对每个批次的损失加权平均

        # 计算正确预测的数量
        _, predicted = torch.max(outputs, 1)  # 获取最大概率的类别索引
        correct_predictions += (predicted == labels).sum().item()  # 计算正确预测的数量
        total_samples += labels.size(0)  # 更新总样本数

        # 每 10 个批次打印一次当前的训练状态（损失和精度）
        if i % 10 == 0:
            batch_accuracy = (correct_predictions / total_samples) * 100  # 当前批次精度
            print(f"one epoch training, Epoch [{epoch}/{total_epochs}], "
                  f"Batch [{i}/{len(data_loader)}], Loss: {loss.item():.4f}, Accuracy: {batch_accuracy:.2f}%")

    # 计算平均损失和总训练精度
    avg_loss = running_loss / total_samples  # 平均损失
    accuracy = correct_predictions / total_samples  # 总训练精度

    print(f"one epoch end, Epoch [{epoch}/{total_epochs}], "
          f"training average Loss: {avg_loss:.4f}, average Accuracy: {accuracy * 100:.4f}%")

    return avg_loss, accuracy


def save_model(model, model_save_path, log_info):
    print(f"{log_info} saved to {model_save_path}")
    torch.save(model.state_dict(), model_save_path)


def save_checkpoint(model, optimizer, epoch, loss, save_checkpoint_path, log_info):
    """
    保存模型和优化器的状态，包含当前训练的 epoch 和 loss，用于恢复训练。

    :param model: PyTorch 模型，保存模型的参数（state_dict）。
    :param optimizer: PyTorch 优化器，保存优化器的状态（state_dict）。
    :param epoch: 当前训练的 epoch 数，方便恢复训练时从哪里开始。
    :param loss: 当前训练的损失值，用于恢复训练时的损失。
    :param save_checkpoint_path: 保存检查点的文件路径，通常是一个 `.pth` 或 `.pt` 后缀的文件。
    :param log_info: 打印的日志信息
    """
    # 创建一部字典来存储检查点信息，包括模型的参数、优化器的参数、epoch 和损失
    checkpoint = {
        'epoch': epoch,  # 当前训练的 epoch 数
        'model_state_dict': model.state_dict(),  # 保存模型的参数（模型的权重）
        'optimizer_state_dict': optimizer.state_dict(),  # 保存优化器的参数（如学习率、动量等）
        'loss': loss  # 当前的损失值，可以用来恢复训练状态
    }

    # 使用 torch.save 保存检查点到指定的文件
    torch.save(checkpoint, save_checkpoint_path)

    # 打印保存成功的提示信息
    print(f"{log_info} saved to {save_checkpoint_path}")


def load_checkpoint(model, optimizer, save_model_path):
    """
    从保存的检查点文件中加载模型和优化器的状态，以恢复训练。

    :param model: PyTorch 模型，加载保存的模型参数（state_dict）。
    :param optimizer: PyTorch 优化器，加载保存的优化器参数（state_dict）。
    :param save_model_path: 存储检查点的文件路径，通常是一个 `.pth` 或 `.pt` 后缀的文件。
    :return: 返回加载后的模型、优化器、epoch 和损失值。
    """
    # 从文件中加载检查点（字典形式），包含模型、优化器状态及其他信息
    checkpoint = torch.load(save_model_path)

    # 加载模型的参数（state_dict）
    model.load_state_dict(checkpoint['model_state_dict'])

    # 加载优化器的参数（state_dict）
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # 获取保存的 epoch 和损失值
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']

    # 打印加载成功的提示信息，告诉用户从哪个 epoch 恢复训练
    print(f"Checkpoint loaded from {save_model_path}. Starting from epoch {epoch}.")

    # 返回加载后的模型、优化器、epoch 和损失值
    return model, optimizer, epoch, loss
