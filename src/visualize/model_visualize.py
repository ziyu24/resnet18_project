"""
Created on 11 16, 2024
@author: <Cui>
@bref: 模型训练或者推理的时候的可视化
"""

from PIL import Image
from project.src.infer import infer

import os
import torch
import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix

from project.src.common.config import config_yaml


def visualize_predictions(model, image_dir, grid_size=None, use_save_model=config_yaml['test']['use_save_model']):
    """
    可视化image_dir下的预测结果，并将图像排列成正方形网格

    参数:
    - image_dir: 图像目录
    - grid_size: 如果传入此参数，将会使用指定的网格尺寸
    """
    # 获取目录中的所有图像文件
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if
                   f.endswith(('.png', '.jpg', '.jpeg'))]

    image_path_predictions = []
    if use_save_model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(config_yaml['model_save_path'], map_location=device))
    model.eval()  # 切换到评估模式

    for image_path in image_files:
        # 获取每个图像的预测结果
        image_path_predictions.append((image_path, infer(model, image_path)))

    num_images = len(image_path_predictions)

    if num_images == 1:
        # 只有一张图像时，不使用子图，直接绘制
        image_path, pred = image_path_predictions[0]
        image = Image.open(image_path).convert("RGB")
        plt.imshow(image)
        plt.title(f'Pred: {pred}')
        plt.axis('off')
        plt.show()
        return

    # 如果没有提供grid_size，自动计算网格大小
    if not grid_size:
        grid_size = _get_grid_size(num_images)

    # 创建子图
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size * 2, grid_size * 2))
    axes = axes.flatten()  # 将二维的axes展平，以便按顺序填充每个子图

    # 遍历所有图像并绘制
    for i, (image_path, pred) in enumerate(image_path_predictions):
        image = Image.open(image_path).convert("RGB")
        axes[i].imshow(image)
        axes[i].set_title(f'Pred: {pred}')
        axes[i].axis('off')  # 关闭坐标轴显示

    # 如果图像数量少于网格大小，隐藏多余的子图
    for i in range(num_images, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


def visualize_confusion_matrix(model, test_data, num_classes,
                               save_model_path=config_yaml['train_model_save_path'],
                               title='train'):
    """
    公共方法：可视化混淆矩阵
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if config_yaml['test']['use_save_model']:
        model.load_state_dict(torch.load(save_model_path, map_location=device))
    model.eval()  # 切换到评估模式

    y_true, y_pred = _get_predictions(model, test_data, device)  # 获取真实标签和预测标签
    _plot_confusion_matrix(y_true, y_pred, num_classes, title)  # 绘制混淆矩阵


def _get_predictions(model, test_data, device):
    """
    私有方法：获取所有预测结果和真实标签

    :return: 返回预测标签和真实标签
    """
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 禁用梯度计算
        for images, labels in test_data:
            # 将图像和标签移动到设备
            images, labels = images.to(device), labels.to(device)

            # 模型推理
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.numpy())  # 真实标签
            all_preds.extend(predictions.numpy())  # 预测标签

    return all_labels, all_preds


def _plot_confusion_matrix(y_true, y_pred, num_classes, title='train'):
    """
    私有方法：绘制混淆矩阵的热力图

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param num_classes: 类别个数
    :param title: 混合矩阵头
    """
    cm = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=num_classes, yticklabels=num_classes)
    plt.title(title + ' Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def display_inference_results(model, test_data, show_number,
                              saved_model_path=config_yaml['train_model_save_path'],
                              dynamic_resize=True, device=None):
    """
    显示模型推理结果，可选择动态调整窗口大小。

    参数:
    - model: 待测试的模型 (torch.nn.Module)
    - test_data: 测试数据集 (DataLoader)
    - show_number: 显示图片的个数 (int)
    - saved_model_path: 保存的模型参数路径 (str)
    - dynamic_resize: 是否根据图像大小动态调整窗口 (bool)
    - device: 运行设备，默认自动选择 (torch.device)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    if config_yaml['test']['use_save_model']:
        model.load_state_dict(torch.load(saved_model_path, map_location=device))
    model.eval()

    displayed_count = 0
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    with torch.no_grad():
        for images, labels in test_data:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            for i in range(images.size(0)):
                if displayed_count >= show_number:
                    return

                # 获取当前图像及其预测结果
                image = images[i].cpu()
                predicted_label = predictions[i].item()
                true_label = labels[i].item()

                # **逆归一化图像**
                image = _denormalize_image(image, mean, std)

                # 根据参数动态调整窗口大小
                if dynamic_resize:
                    dpi = 100
                    height, width = image.shape[:2]
                    plt.figure(figsize=(width / dpi, height / dpi))
                else:
                    plt.figure(figsize=(4, 4))  # 默认固定窗口大小

                # 显示图像
                plt.imshow(image)
                plt.title(f"Prediction: {predicted_label}, True: {true_label}")
                plt.axis('off')
                plt.show()

                displayed_count += 1


def _denormalize_image(image, mean, std):
    """
    逆归一化单张图像。

    :param image: 归一化的图像 Tensor (C, H, W)
    :param mean: 每个通道的均值
    :param std: 每个通道的标准差
    :return: 逆归一化后的 NumPy 图像
    """
    image = image.clone()
    for t, m, s in zip(image, mean, std):
        t.mul_(s).add_(m)
    image = torch.clamp(image, 0, 1)  # 限制像素值范围在 [0, 1]
    return image.permute(1, 2, 0).numpy()  # 转换为 NumPy 格式 (H, W, C)




def _get_grid_size(num_images):
    """根据图像数量自动计算合适的网格大小"""
    grid_size = int(num_images ** 0.5)  # 取平方根作为网格大小的初步估算
    if grid_size * grid_size < num_images:
        grid_size += 1  # 如果网格的总大小不足以放下所有图像，增加网格大小
    return grid_size
