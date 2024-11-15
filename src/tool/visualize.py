import torch
from matplotlib import pyplot as plt

# from project.src.net.net_result import NetResult
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from project.src.infer import infer

import matplotlib.pyplot as plt
import math
import os

import seaborn as sns
from sklearn.metrics import confusion_matrix


from project.src.common.config import config_yaml


def visualize_predictions(model, image_dir, grid_size=None, use_save_model=config_yaml['val']['use_save_model']):
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


def visualize_confusion_matrix(model, test_data, classes, use_save_model=config_yaml['val']['use_save_model']):
    """
    公共方法：可视化混淆矩阵
    """
    if use_save_model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(config_yaml['model_save_path'], map_location=device))
    model.eval()  # 切换到评估模式

    y_true, y_pred = _get_predictions(model, test_data)  # 获取真实标签和预测标签
    _plot_confusion_matrix(y_true, y_pred, classes)  # 绘制混淆矩阵


def _get_predictions(model, test_data):
    """
    私有方法：获取所有预测结果和真实标签

    :return: 返回预测标签和真实标签
    """
    all_labels = []
    all_preds = []

    with torch.no_grad():  # 禁用梯度计算
        for (x, y) in test_data:
            output = model(x.view(-1, config_yaml['net']['input_size']))  # 输入数据经过模型
            _, predicted = torch.max(output, 1)  # 获取最大概率的标签

            all_labels.extend(y.numpy())  # 真实标签
            all_preds.extend(predicted.numpy())  # 预测标签

    return all_labels, all_preds


def _plot_confusion_matrix(y_true, y_pred, classes):
    """
    私有方法：绘制混淆矩阵的热力图

    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param classes: 类别个数
    """
    cm = confusion_matrix(y_true, y_pred)  # 计算混淆矩阵

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


def show_test_result(model, test_data, show_number, use_save_model=config_yaml['val']['use_save_model']):
    """
    显示预测结果: 从 test_data 数据集中查看 show_number 个图像预测结果

    参数:
    - model: 待评估的模型
    - test_data: 评估的数据集
    - show_number: 预测图片的个数
    """

    if use_save_model:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(config_yaml['model_save_path'], map_location=device))
    model.eval()  # 切换到评估模式

    for (n, (x, _)) in enumerate(test_data):
        if n > (show_number - 1):
            break

        predict = torch.argmax(model.forward(x[0].view(-1, config_yaml['net']['input_size'])))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))

        plt.show()


def _get_grid_size(num_images):
    """根据图像数量自动计算合适的网格大小"""
    grid_size = int(num_images ** 0.5)  # 取平方根作为网格大小的初步估算
    if grid_size * grid_size < num_images:
        grid_size += 1  # 如果网格的总大小不足以放下所有图像，增加网格大小
    return grid_size
