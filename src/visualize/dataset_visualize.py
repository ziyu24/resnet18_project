"""
Created on 11 20, 2024
@author: <Cui>
@bref: 数据集的可视化
"""
import os

import matplotlib.pyplot as plt
from collections import Counter

import numpy as np
from PIL import Image

from project.src.common.config_dataset_dota_1_0 import dataset_dota_to_voc_dir, dataset_train_image_dir, \
    dataset_dota_to_yolo_dir


# from project.src.common.config_dataset_fair1m_2_0 import dataset_train_image_dir, dataset_train_txt_label_dir


def visualize_dota_txt(image_path, txt_path):
    """
    可视化 DOTA1.0 数据集的旋转框标注。

    参数:
    - image_path: 图片路径
    - txt_path: 对应的 DOTA1.0 标注路径
    - class_names: 类别名称列表
    """
    # 读取图片
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    # 读取 TXT 标注
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 9:  # 至少需要 9 个字段 (x1, y1, ..., x4, y4, class_name, difficulty)
                continue

            # 提取坐标点
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
            class_name = parts[8]

            # 绘制多边形
            points = [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]
            for i in range(len(points)):
                cv2.line(image, points[i], points[(i + 1) % 4], (0, 255, 0), 2)

            # 绘制类别标签
            label_position = (int(x1), int(y1) - 10)
            cv2.putText(image, class_name, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 展示结果
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()


import matplotlib.pyplot as plt
from collections import Counter


def plot_category_histogram(dota_txt_folder):
    """
    绘制 DOTA1.0 数据集的类别直方图。

    参数:
    - dota_txt_folder: DOTA 数据集标签文件的目录路径
    """
    category_counts = Counter()

    # 遍历所有标签文件
    for txt_file in os.listdir(dota_txt_folder):
        if txt_file.endswith('.txt'):
            with open(os.path.join(dota_txt_folder, txt_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:  # 至少包含类别字段
                        category = parts[8]
                        category_counts[category] += 1

    # 绘制直方图
    categories = list(category_counts.keys())
    counts = list(category_counts.values())
    plt.figure(figsize=(12, 6))
    plt.bar(categories, counts, color='skyblue')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Category Distribution in DOTA Dataset')
    plt.tight_layout()
    plt.show()


def plot_image_size_distribution(dota_image_folder):
    """
    绘制 DOTA1.0 数据集图像宽高分布图。

    参数:
    - dota_image_folder: DOTA 数据集图像文件的目录路径
    """
    widths, heights = [], []

    # 遍历图像文件
    for image_file in os.listdir(dota_image_folder):
        if image_file.endswith(('.png', '.jpg', '.tif')):
            image = cv2.imread(os.path.join(dota_image_folder, image_file))
            if image is not None:
                h, w, _ = image.shape
                widths.append(w)
                heights.append(h)

    # 绘制宽高分布图
    plt.figure(figsize=(10, 6))
    plt.scatter(widths, heights, alpha=0.5, color='blue', label='Images')
    plt.xlabel('Width')
    plt.ylabel('Height')
    plt.title('Image Size Distribution in DOTA Dataset')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def visualize_category_grid(image_folder, txt_folder, category, grid_size=3, suffix='.png'):
    """
    可视化特定类别的所有目标，生成 9 宫格。

    参数:
    - image_folder: DOTA 数据集图像文件的目录路径
    - txt_folder: DOTA 数据集标签文件的目录路径
    - category: 指定的类别名称
    - grid_size: 每行/列的图片数量
    - suffix: 数据集图片后缀
    """
    import random

    # 找到所有包含指定类别的样本
    samples = []
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            with open(os.path.join(txt_folder, txt_file), 'r') as f:
                has_category = False
                coords_list = []
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9 and parts[8] == category:
                        has_category = True
                        coords_list.append(parts[:8])  # 提取坐标
                if has_category:
                    samples.append((txt_file.replace('.txt', suffix), coords_list))

    # 随机选择 grid_size**2 个样本
    selected_samples = random.sample(samples, min(grid_size**2, len(samples)))

    # 绘制 9 宫格
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12))
    axes = axes.flatten()

    for ax, (image_file, coords_list) in zip(axes, selected_samples):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        if image is None:
            ax.axis('off')
            continue

        # 绘制所有旋转框
        for coords in coords_list:
            x1, y1, x2, y2, x3, y3, x4, y4 = map(float, coords)
            points = [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]
            for i in range(4):
                cv2.line(image, points[i], points[(i + 1) % 4], (0, 255, 0), 2)

        # 显示图像
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(f"Category: {category}", fontsize=10)
        ax.axis('off')

    # 如果子图不足，则隐藏剩余的子图
    for ax in axes[len(selected_samples):]:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


import os
import cv2
import matplotlib.pyplot as plt

def visualize_dense_sparse(image_folder, txt_folder, num_dense=5, num_sparse=5, suffix='.png'):
    """
    可视化密集和稀疏目标的特例，并在图片上绘制目标框和类别。

    参数:
    - image_folder: 图像文件夹路径
    - txt_folder: 标签文件夹路径
    - num_dense: 展示密集目标图片的数量
    - num_sparse: 展示稀疏目标图片的数量
    - suffix: 数据集图片后缀
    """
    object_counts = {}

    # 统计每张图片的目标数量
    for txt_file in os.listdir(txt_folder):
        if txt_file.endswith('.txt'):
            count = 0
            with open(os.path.join(txt_folder, txt_file), 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 9:
                        count += 1
            object_counts[txt_file.replace('.txt', suffix)] = count

    # 找到密集和稀疏目标图片
    sorted_counts = sorted(object_counts.items(), key=lambda x: x[1])
    sparse_images = sorted_counts[:num_sparse]
    dense_images = sorted_counts[-num_dense:]

    # 可视化单张图片及绘制目标框
    def visualize_image_with_boxes(image_path, txt_path, ax, title):
        """
        绘制单张图片的目标框。
        """
        image = cv2.imread(image_path)
        if image is None:
            ax.set_title(f"Failed to load {os.path.basename(image_path)}")
            ax.axis('off')
            return

        # 绘制目标框
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 9:
                    continue
                x1, y1, x2, y2, x3, y3, x4, y4 = map(float, parts[:8])
                class_name = parts[8]

                # 绘制旋转框
                points = [(int(x1), int(y1)), (int(x2), int(y2)), (int(x3), int(y3)), (int(x4), int(y4))]
                for i in range(4):
                    cv2.line(image, points[i], points[(i + 1) % 4], (0, 255, 0), 2)

                # 绘制类别标签
                label_position = (int(x1), int(y1) - 10)
                cv2.putText(image, class_name, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # 显示结果
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis('off')

    # 可视化稀疏和密集图片
    def plot_images_with_boxes(images, title):
        fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
        for ax, (image_file, count) in zip(axes, images):
            image_path = os.path.join(image_folder, image_file)
            txt_path = os.path.join(txt_folder, image_file.replace(suffix, '.txt'))
            visualize_image_with_boxes(image_path, txt_path, ax, f'{image_file}\nCount: {count}')
        plt.suptitle(title)
        plt.tight_layout()
        plt.show()

    plot_images_with_boxes(sparse_images, "Sparse Examples")
    plot_images_with_boxes(dense_images, "Dense Examples")


import cv2
import xml.etree.ElementTree as ET

def visualize_voc_annotation(image_path, xml_path, output_path=None):
    """
    可视化基于 VOC XML 格式的标注。

    参数:
    - image_path (str): 输入图像路径。
    - xml_path (str): 对应的 VOC XML 文件路径。
    - output_path (str): 可选，保存标注可视化图像的路径。若为 None，则仅展示。

    返回:
    - None: 展示或保存带标注的图像。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    # 解析 XML 文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    for obj in root.findall('object'):
        label = obj.find('name').text
        bndbox = obj.find('bndbox')
        x_min = int(bndbox.find('xmin').text)
        y_min = int(bndbox.find('ymin').text)
        x_max = int(bndbox.find('xmax').text)
        y_max = int(bndbox.find('ymax').text)

        # 绘制矩形框和标签
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if output_path:
        cv2.imwrite(output_path, image)
    else:
        cv2.imshow("VOC Annotation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def visualize_yolo_annotation(image_path, txt_path, label_map, output_path=None):
    """
    可视化基于 YOLO TXT 格式的标注。

    参数:
    - image_path (str): 输入图像路径。
    - txt_path (str): 对应的 YOLO TXT 文件路径。
    - label_map (dict): YOLO 标签 ID 到类别名的映射。
    - output_path (str): 可选，保存标注可视化图像的路径。若为 None，则仅展示。

    返回:
    - None: 展示或保存带标注的图像。
    """
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    height, width, _ = image.shape

    # 读取 TXT 文件
    with open(txt_path, 'r') as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        label_id = int(parts[0])
        x_center, y_center, box_width, box_height = map(float, parts[1:])

        # 反算框的坐标
        x_min = int((x_center - box_width / 2) * width)
        y_min = int((y_center - box_height / 2) * height)
        x_max = int((x_center + box_width / 2) * width)
        y_max = int((y_center + box_height / 2) * height)

        # 获取类别名
        label = label_map.get(label_id, f"ID {label_id}")

        # 绘制矩形框和标签
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    if output_path:
        cv2.imwrite(output_path, image)
    else:
        cv2.imshow("YOLO Annotation", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




if __name__ == '__main__':
    # image_path = dataset_train_image_dir + '/2.tif'
    # txt_path = dataset_train_txt_label_dir + '/2.txt'
    # visualize_dota_txt(image_path, txt_path)

    # plot_category_histogram(dataset_train_txt_label_dir)
    # plot_image_size_distribution(dataset_train_image_dir)
    # visualize_category_grid(dataset_train_image_dir, dataset_train_txt_label_dir, 'Small', suffix='.tif')
    # visualize_dense_sparse(dataset_train_image_dir, dataset_train_txt_label_dir, suffix='.tif')
    image_path = dataset_train_image_dir + '/P0000.png'
    # label_voc_path = dataset_dota_to_voc_dir + '/P0000.xml'
    # visualize_voc_annotation(image_path, label_voc_path, './temp.jpg')
    label_yolo_path = dataset_dota_to_yolo_dir + '/P0000.txt'

    # 原始字典
    original_mapping = {
        "baseball-diamond": 0,
        "basketball-court": 1,
        "bridge": 2,
        "container-crane": 3,
        "ground-track-field": 4,
        "harbor": 5,
        "helicopter": 6,
        "large-vehicle": 7,
        "plane": 8,
        "roundabout": 9,
        "ship": 10,
        "small-vehicle": 11,
        "soccer-ball-field": 12,
        "storage-tank": 13,
        "swimming-pool": 14,
        "tennis-court": 15
    }
    reversed_mapping = {v: k for k, v in original_mapping.items()}
    visualize_yolo_annotation(image_path, label_yolo_path,  reversed_mapping, './temp_yolo.jpg')

