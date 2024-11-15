"""
Created on 11 13, 2024
@author: <Cui>
@bref: 测试全部模块
"""
from project.src.models.resnet18_self import get_resnet18_self, get_pretrained_model
from project.src.models.resnet18_tv import get_resnet18_tv
from project.src.common.config import config_yaml
from project.src.tool.visualize import show_test_result, visualize_confusion_matrix, visualize_predictions
from project.src.datasets.data_loader import get_data_loader_train, get_data_loader_val


# from val import evaluate_train, evaluate_acc
from train import train
from infer import infer


def main():
    # model = get_resnet18_self(config_yaml['data']['num_classes'])
    model = get_pretrained_model(config_yaml['data']['num_classes'])

    # model = get_resnet18_tv()
    train(model, get_data_loader_train(), get_data_loader_val())


if __name__ == "__main__":
    main()
