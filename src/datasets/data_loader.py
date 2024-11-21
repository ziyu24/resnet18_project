"""
Created on 11 14, 2024
@author: <Cui>
@brief: 创建训练集和验证集的 DataLoader
"""

from torch.utils.data import DataLoader

from project.src.common.config import config_yaml
from project.src.datasets.dataset import DatasetSelf
from project.src.datasets.dataset_transform import transform_train, transform_val


def get_data_loader_train():
    dataset_train = DatasetSelf(config_yaml['dataset_train_dir'], transform_train)

    data_loader_train = DataLoader(dataset=dataset_train,
                                   batch_size=config_yaml['train']['batch_size'],
                                   shuffle=True,
                                   num_workers=config_yaml['train']['num_workers']
                                   )
    return data_loader_train


def get_data_loader_val():
    dataset_val = DatasetSelf(config_yaml['dataset_val_dir'], transform_val)

    data_loader_val = DataLoader(dataset=dataset_val,
                                 batch_size=config_yaml['val']['batch_size'],
                                 num_workers=config_yaml['val']['num_workers']
                                 )

    return data_loader_val
