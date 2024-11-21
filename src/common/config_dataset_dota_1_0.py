"""
Created on 11 17, 2024
@author: <Cui>
@bref: 配置不能在 yaml 中配置的配置项
"""

import os

current_abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# project 目录的路径
project_dir = current_abs_dir + '/../..'

# 数据集路径
dataset_dir = '/DOTA1.0'
dataset_train_txt_label_dir = project_dir + '/data/dataset' + dataset_dir + '/train/labelTxt-v1.5/DOTA-v1.5_train'
dataset_train_image_dir = project_dir + '/data/dataset' + dataset_dir + '/train/images'
dataset_dota_to_voc_dir = project_dir + '/data/dataset/DOTA1.0_to_VOC/train_label'
dataset_dota_to_yolo_dir = project_dir + '/data/dataset/DOTA1.0_to_YOLO/train_label'


dataset_val_label_dir = project_dir + '/data/dataset' + dataset_dir + '/validation'

