"""
Created on 11 13, 2024
@author: <Cui>
@bref: 模型训练操作，推理，可视化等操作
"""

import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from project.src.common.config import config_yaml


def infer(model, image_path):
    """
    推理函数: 给定一个模型和图片路径，输出预测的类别标签

    参数:
    - model: 推理模型
    - image_path: 推理图像路径
    """

    # 设置模型为评估模式
    model.eval()

    # 读取并预处理图片
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转为灰度图（如果需要）
        transforms.Resize((config_yaml['net']['height'], config_yaml['net']['width'])),  # 调整为28x28的大小
        transforms.ToTensor(),  # 转为Tensor
        # transforms.Normalize(mean=[config_yaml['train']['mean']], std=[config_yaml['train']['std']])  # 使用MNIST的归一化值
    ])

    # 打开图片
    image = Image.open(image_path)
    # 对图片进行转换
    image_tensor = transform(image)
    # # 保存图像的路径
    # image_path = config_yaml['test_data_dir'] + '/../temp/out_put_1.jpg'
    # save_image(image_tensor, image_path)
    # 展平图像为1x784（batch_size, 784）张量
    image_tensor = image_tensor.view(1, -1)  # 添加batch维度，并将28x28图像展平为一维

    # 禁用梯度计算
    with torch.no_grad():
        # 进行推理
        output = model(image_tensor)  # 直接传入模型，保持 batch 维度
        # print(output)

        # 使用 softmax 转换为概率分布
        output_probs = F.softmax(output, dim=1)
        # print(output_probs)

        # 获取最大值索引，这就是预测类别
        predicted_class = output_probs.argmax(dim=1).item()

    return predicted_class
