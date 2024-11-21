"""
Created on 11 13, 2024
@author: <Cui>
@bref: 测试全部模块
"""
import torch

from project.src.models.resnet18_self import get_resnet18_self, get_pretrained_model_self
from project.src.models.resnet18_tv import get_resnet18_tv
from project.src.common.config import config_yaml
from project.src.visualize.model_visualize import visualize_confusion_matrix, visualize_predictions, \
    display_inference_results
from project.src.datasets.data_loader import get_data_loader_train, get_data_loader_val
from project.src.val import evaluate

# from val import evaluate_train, evaluate_acc
from train import train
from infer import infer


def main():
    # model = get_resnet18_self(config_yaml['data']['num_classes'])
    model = get_pretrained_model_self(config_yaml['data']['num_classes'])

    # model = get_resnet18_tv()
    train(model, get_data_loader_train(), get_data_loader_val())


def test_inference_result():
    model = get_resnet18_tv()
    # show_inference_result(model, get_data_loader_train(), 2)
    display_inference_results(model, get_data_loader_train(), 2, config_yaml['val_model_save_path'], False)
    # visualize_confusion_matrix(model, get_data_loader_val(), 23, title='val')


def val_test():
    evaluate()


def test():
    import torch.nn as nn

    # 使用 nn.Sequential 将卷积、批归一化和ReLU激活函数按顺序排列
    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU()
    )

    # 输入张量通过这些层
    input_tensor = torch.randn(2, 3, 4, 3)  # 假设输入大小为 [batch_size, channels, height, width]
    print(input_tensor.shape)
    print(input_tensor)
    output_tensor = model(input_tensor)  # 自动按顺序传递
    print(output_tensor.shape)
    print(output_tensor)


if __name__ == "__main__":
    print("torch version: {}".format(torch.__version__))
    # main()
    # val_test()
    # test_inference_result()