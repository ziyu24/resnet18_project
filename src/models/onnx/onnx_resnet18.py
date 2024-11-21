"""
Created on 11 16, 2024
@author: <Cui>
@bref: 构建适配于 resnet18 的 onnx
"""

import torch
import onnxruntime
import numpy as np
from PIL import Image
from torchvision import transforms

from project.src.common.config import config_yaml
from project.src.models.onnx.onnx_common import export_pth_to_onnx_common, infer_with_onnx_common
from project.src.models.resnet18_self import get_pretrained_model_self


def export_resnet18_pth_to_onnx(input_size=(1, 3, 224, 224)):
    model = get_pretrained_model_self(config_yaml['data']['num_classes'])
    export_pth_to_onnx_common(model, config_yaml['train_model_save_path'], config_yaml['onnx_save_path'], input_size)


def infer_with_onnx_resnet18(image_path):
    return infer_with_onnx_common(config_yaml['onnx_save_path'], image_path)


if __name__ == '__main__':
    export_resnet18_pth_to_onnx()
    # infer_with_onnx_resnet18(config_yaml['dataset_test_dir'] + '/3/3_5_106_12608.jpg')
