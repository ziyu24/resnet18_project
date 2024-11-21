"""
Created on 11 14, 2024
@author: <Cui>
@brief: 创建定制的 Dataset
"""

import xml.etree.ElementTree as ET

from project.src.common.config_dataset_dota_1_0 import dataset_train_image_dir, dataset_dota_to_yolo_dir, \
    dataset_train_txt_label_dir, dataset_dota_to_voc_dir
# from project.src.common.config_dataset_fair1m_2_0 import dataset_train_xml_label_dir, dataset_train_txt_label_dir

import os
import xml.etree.ElementTree as ET


def convert_fair1m_to_dota(fair1m_folder, dota_folder, gsd=1):
    """
    将 FAIR1M 标签文件夹中的所有 XML 文件批量转换为 DOTA1.0 标签文件格式。

    Args:
        fair1m_folder (str): FAIR1M 标签文件夹路径。
        dota_folder (str): 输出的 DOTA1.0 标签文件夹路径。
        gsd (float): 图像分辨率（可选）。
    """
    # 如果输出文件夹不存在，则创建
    os.makedirs(dota_folder, exist_ok=True)

    # 遍历 FAIR1M 文件夹中的所有 XML 文件
    for filename in os.listdir(fair1m_folder):
        if not filename.endswith(".xml"):
            continue

        fair1m_xml_path = os.path.join(fair1m_folder, filename)
        dota_txt_path = os.path.join(dota_folder, filename.replace(".xml", ".txt"))

        # 转换单个 FAIR1M 标签文件
        convert_fair1m_xml_to_dota_txt(fair1m_xml_path, dota_txt_path, gsd)

    print(f"All files converted! DOTA labels saved to: {dota_folder}")


def convert_fair1m_xml_to_dota_txt(fair1m_xml_path, dota_output_path, gsd=1):
    """
    将单个 FAIR1M 标签文件转换为 DOTA1.0 标签格式。

    Args:
        fair1m_xml_path (str): FAIR1M 标签文件路径。
        dota_output_path (str): 输出的 DOTA1.0 标签文件路径。
        gsd (float): 图像分辨率（可选）。
    """
    # 解析 XML 文件
    tree = ET.parse(fair1m_xml_path)
    root = tree.getroot()

    # 打开输出文件
    with open(dota_output_path, 'w') as dota_file:
        # 写入头部信息
        image_source = root.find("./source/origin").text
        dota_file.write(f"imagesource:{image_source}\n")
        dota_file.write(f"gsd:{gsd}\n")

        # 遍历每个对象
        for obj in root.findall("./objects/object"):
            # 获取类别名称
            category = obj.find("./possibleresult/name").text

            # 获取点的坐标
            points = obj.find("./points")
            coords = []
            for point in points.findall("point"):
                x, y = map(float, point.text.split(","))
                coords.append((x, y))

            # 去掉最后一个重复点
            coords = coords[:-1]

            # 确保有 4 个点
            if len(coords) != 4:
                print(f"Warning: Skipping object with invalid coordinates in file {fair1m_xml_path}: {coords}")
                continue

            # 将坐标展开为顺时针格式
            flattened_coords = " ".join(f"{x:.1f} {y:.1f}" for x, y in coords)

            # 写入 DOTA1.0 格式行
            dota_file.write(f"{flattened_coords} {category} 1\n")



import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def parse_dota_annotations(dota_file):
    """
    解析 DOTA 标注文件，将旋转框读取为列表。

    参数:
    - dota_file (str): DOTA 数据集中的标注文件路径，通常为 `.txt` 格式。

    返回:
    - List[dict]: 包含每个目标的标注信息，每个字典包含以下字段：
        - "points" (list of float): 目标的四个顶点坐标，格式为 [x1, y1, x2, y2, x3, y3, x4, y4]。
        - "label" (str): 目标的类别标签。
        - "difficult" (int): 难度标记，0 表示不困难，1 表示困难。
    """
    annotations = []
    with open(dota_file, 'r') as f:
        for line in f.readlines():
            parts = line.strip().split()
            if len(parts) >= 10:
                # 获取多边形的四个顶点 (x1, y1, ..., x4, y4)
                points = list(map(float, parts[:8]))
                label = parts[8]
                difficult = int(parts[9])  # 第10个元素表示 difficult，转为整数
                annotations.append({"points": points, "label": label, "difficult": difficult})
    return annotations


def rotated_to_horizontal(points):
    """
    将旋转框转换为最小外接水平框。

    参数:
    - points (list of float): 目标旋转框的四个顶点坐标，格式为 [x1, y1, x2, y2, x3, y3, x4, y4]。

    返回:
    - (x_min, y_min, x_max, y_max) (tuple): 最小外接水平框的四个坐标，格式为 (x_min, y_min, x_max, y_max)。
    """
    points = np.array(points).reshape(4, 2)
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    return x_min, y_min, x_max, y_max


import xml.dom.minidom as minidom

def save_as_voc_xml(output_path, image_name, image_size, annotations):
    """
    保存为格式化的 VOC XML 格式。

    参数:
    - output_path (str): 输出的 VOC XML 文件路径。
    - image_name (str): 图像文件名（例如 'image_001.jpg'）。
    - image_size (tuple): 图像的大小，格式为 (height, width, channels)。
    - annotations (list of dict): 标注信息列表，每个字典包含以下字段：
        - "label" (str): 目标的类别标签。
        - "bbox" (tuple): 最小外接框的坐标 (x_min, y_min, x_max, y_max)。
        - "difficult" (int): 难度标记，0 或 1。

    返回:
    - None: 函数直接将结果写入输出的 XML 文件中。
    """
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'VOC'
    ET.SubElement(annotation, 'filename').text = image_name
    ET.SubElement(annotation, 'path').text = output_path

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(image_size[1])
    ET.SubElement(size, 'height').text = str(image_size[0])
    ET.SubElement(size, 'depth').text = str(image_size[2])

    for ann in annotations:
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = ann['label']
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = str(ann['difficult'])  # 添加 difficult 标记

        bbox = ET.SubElement(obj, 'bndbox')
        x_min, y_min, x_max, y_max = ann['bbox']
        ET.SubElement(bbox, 'xmin').text = str(int(x_min))
        ET.SubElement(bbox, 'ymin').text = str(int(y_min))
        ET.SubElement(bbox, 'xmax').text = str(int(x_max))
        ET.SubElement(bbox, 'ymax').text = str(int(y_max))

    # 格式化输出
    xml_string = ET.tostring(annotation, 'utf-8')
    dom = minidom.parseString(xml_string)
    pretty_xml_as_string = dom.toprettyxml(indent="  ")

    with open(output_path, 'w') as f:
        f.write(pretty_xml_as_string)


def save_as_yolo_txt(output_path, image_size, annotations, label_map):
    """
    保存为 YOLO TXT 格式。

    参数:
    - output_path (str): 输出的 YOLO TXT 文件路径。
    - image_size (tuple): 图像的大小，格式为 (height, width, channels)。
    - annotations (list of dict): 标注信息列表，每个字典包含以下字段：
        - "label" (str): 目标类别。
        - "bbox" (tuple): 最小外接框的坐标 (x_min, y_min, x_max, y_max)。
    - label_map (dict): 目标类别到 YOLO label_id 的映射表。

    返回:
    - None: 函数直接将结果写入输出的 TXT 文件中。
    """
    with open(output_path, 'w') as f:
        for ann in annotations:
            label = ann['label']
            if label not in label_map:
                print(f"Warning: Label '{label}' not in label_map, skipping.")
                continue

            label_id = label_map[label]
            x_min, y_min, x_max, y_max = ann['bbox']
            x_center = (x_min + x_max) / 2.0 / image_size[1]
            y_center = (y_min + y_max) / 2.0 / image_size[0]
            width = (x_max - x_min) / image_size[1]
            height = (y_max - y_min) / image_size[0]

            f.write(f"{label_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


import os
import xml.etree.ElementTree as ET
import cv2
import numpy as np


def process_dota_to_voc(dota_dir, image_dir, output_dir, suffix='.png'):
    """
    处理 DOTA 数据集的标注文件并保存为 VOC 格式。

    参数:
    - dota_dir (str): DOTA 数据集的标注文件目录，包含多个 `.txt` 格式的标注文件。
    - image_dir (str): 图像文件目录，包含与标注文件对应的图像文件，通常为 `.jpg` 格式。
    - output_dir (str): 输出目录，用于保存转换后的 VOC 格式的标注文件。
    - suffix (str): 图片的后缀，通常为 '.png'。

    返回:
    - None: 函数遍历所有标注文件并处理，生成 VOC 格式的标注文件，并保存在 `output_dir` 中。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 遍历 DOTA 数据集中的标注文件
    for dota_file in os.listdir(dota_dir):
        if not dota_file.endswith('.txt'):
            continue

        # 获取对应的图像文件路径
        image_file = os.path.join(image_dir, os.path.splitext(dota_file)[0] + suffix)

        if not os.path.exists(image_file):
            print(f"Image file not found for {dota_file}, skipping.")
            continue

        # 读取图像大小
        image = cv2.imread(image_file)
        image_size = image.shape  # (height, width, channels)

        # 解析 DOTA 标注
        annotations = parse_dota_annotations(os.path.join(dota_dir, dota_file))
        voc_annotations = []

        for ann in annotations:
            x_min, y_min, x_max, y_max = rotated_to_horizontal(ann['points'])
            voc_annotations.append(
                {'label': ann['label'], 'bbox': (x_min, y_min, x_max, y_max), 'difficult': ann['difficult']})

        # 保存 VOC 格式
        image_name = os.path.basename(image_file)
        voc_output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.xml")
        save_as_voc_xml(voc_output_path, image_name, image_size, voc_annotations)

        print(f"Processed {image_name}: VOC annotations saved to {output_dir}")


def process_dota_to_yolo(dota_dir, image_dir, output_dir, label_map, suffix='.png'):
    """
    处理 DOTA 数据集的标注文件并保存为 YOLO 格式。

    参数:
    - dota_dir (str): DOTA 数据集的标注文件目录，包含多个 `.txt` 格式的标注文件。
    - image_dir (str): 图像文件目录，包含与标注文件对应的图像文件，通常为 `.jpg` 格式。
    - output_dir (str): 输出目录，用于保存转换后的 YOLO 格式的标注文件。
    - label_map (dict): 目标类别到 YOLO label_id 的映射表。
    - suffix (str): 图片的后缀，通常为 '.png'

    返回:
    - None: 函数遍历所有标注文件并处理，生成 YOLO 格式的标注文件，并保存在 `output_dir` 中。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for dota_file in os.listdir(dota_dir):
        if not dota_file.endswith('.txt'):
            continue

        image_file = os.path.join(image_dir, os.path.splitext(dota_file)[0] + suffix)

        if not os.path.exists(image_file):
            print(f"Image file not found for {dota_file}, skipping.")
            continue

        image = cv2.imread(image_file)
        image_size = image.shape  # (height, width, channels)

        annotations = parse_dota_annotations(os.path.join(dota_dir, dota_file))
        yolo_annotations = []

        for ann in annotations:
            x_min, y_min, x_max, y_max = rotated_to_horizontal(ann['points'])
            yolo_annotations.append({'label': ann['label'], 'bbox': (x_min, y_min, x_max, y_max)})

        image_name = os.path.basename(image_file)
        yolo_output_path = os.path.join(output_dir, f"{os.path.splitext(image_name)[0]}.txt")
        save_as_yolo_txt(yolo_output_path, image_size, yolo_annotations, label_map)

        print(f"Processed {image_name}: YOLO annotations saved to {output_dir}")


import os
import json

def generate_label_mapping(dota_label_dir, output_file=None):
    """
    生成 DOTA 数据集中目标名称到 label_id 的映射，并将映射写入文件。

    参数:
    - dota_label_dir (str): DOTA 数据集的标签目录，包含多个 `.txt` 格式的标注文件。
    - output_file (str): 输出的映射文件路径，支持 `.json` 格式。

    返回:
    - dict: 生成的目标名称到 label_id 的映射。
    """
    label_set = set()

    # 遍历所有 DOTA 标注文件
    for label_file in os.listdir(dota_label_dir):
        if not label_file.endswith('.txt'):
            continue

        label_file_path = os.path.join(dota_label_dir, label_file)

        # 读取标注文件，提取目标名称
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 9:
                    label_set.add(parts[8])  # 第9列是目标名称

    # 按字典序生成映射
    label_list = sorted(label_set)
    label_mapping = {label: idx for idx, label in enumerate(label_list)}

    # 将映射写入文件
    if output_file is not None:
        with open(output_file, 'w') as f:
            json.dump(label_mapping, f, indent=4, ensure_ascii=False)

    print(f"Label mapping saved to {output_file}")
    return label_mapping


if __name__ == '__main__':
    #DOTA转换为VOC的标签内容，是一个很长的行，请弄成方便人工阅读的展示的情况；同时，针对yolo格式文件，请给出函数参数，来容纳后续建立的从DOTA目标类到label_id的映射。

    # convert_fair1m_to_dota(dataset_train_xml_label_dir, dataset_train_txt_label_dir)
    # process_dota_to_voc(dataset_train_txt_label_dir, dataset_train_image_dir, dataset_dota_to_voc_dir)
    # 示例映射表
    label_map = generate_label_mapping(dataset_train_txt_label_dir, "./test.txt")
    # process_dota_to_yolo(dataset_train_txt_label_dir, dataset_train_image_dir, dataset_dota_to_yolo_dir, label_map=label_map)
