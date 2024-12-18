"""
Created on 11 17, 2024
@author: <Cui>
@bref: 配置不能在 yaml 中配置的配置项
"""

import os

current_abs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__)))

# yaml 文件绝对路径
config_yaml_path = current_abs_dir + '/config.yaml'

# project 目录的路径
project_dir = current_abs_dir + '/../..'

# 数据集路径
dataset_dir = '/FGSC-23'
data_set_train_dir = project_dir + '/data/dataset' + dataset_dir
data_set_test_dir = project_dir + '/data/dataset' + dataset_dir + 'test'
data_set_val_dir = project_dir + '/data/dataset' + dataset_dir + 'valid'

# 训练模型的路径和模型后缀
train_model_dir = project_dir + '/data/model/model_self/'
train_model_name_suffix = '_self.pth'

# onnx 路径
onnx_dir = project_dir + '/data/model/onnx/onnx_resnet18_self.onnx'

# 项目临时测试数据路径
temp_test_dir = project_dir + '/data/temp/test'

