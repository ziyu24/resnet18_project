"""
Created on 11 14, 2024
@author: <Cui>
@bref: 加载配置文件
"""
import yaml

from project.src.common.config_resnet18 import *


class Config:
    def __init__(self, config_file=config_yaml_path, encoding='utf-8'):
        self._config_local = None
        self.config_file = config_file
        self.encoding = encoding

    def load_config(self):
        """加载 YAML 配置文件，并指定编码格式"""

        if self._validate_yaml():
            with open(self.config_file, 'r', encoding=self.encoding) as file:
                self._config_local = yaml.safe_load(file)

            self._config_project_path()
            return self._config_local
        else:
            return None

    def _config_project_path(self):
        """设置项目的路径，这些路径在 yaml 中配置不了"""

        self._config_local['project_dir'] = project_dir

        self._config_local['dataset_train_dir'] = data_set_train_dir
        self._config_local['dataset_test_dir'] = data_set_test_dir
        self._config_local['dataset_val_dir'] = data_set_val_dir
        self._config_local['temp_test_data_dir'] = temp_test_dir

        self._config_local['train_check_point_save_path'] = (train_model_dir + 'train_checkpoint' +
                                                             train_model_name_suffix)
        self._config_local['train_model_save_path'] = train_model_dir + 'train_model' + train_model_name_suffix
        self._config_local['val_check_point_save_path'] = train_model_dir + 'val_checkpoint' + train_model_name_suffix
        self._config_local['val_model_save_path'] = train_model_dir + 'val_model' + train_model_name_suffix

        self._config_local['onnx_save_path'] = onnx_dir


    def _validate_yaml(self):
        """验证 YAML 文件格式"""
        try:
            with open(self.config_file, 'r', encoding=self.encoding) as file:
                yaml.safe_load(file)
            return True
        except yaml.YAMLError as e:
            print("YAML 格式错误:", e)
            return False


config_yaml = Config().load_config()

if __name__ == '__main__':
    # 使用配置
    config_class = Config()
    config = config_class.load_config()

    # 访问配置中的参数
    input_size = config['net']['input_size']

    # 打印配置值
    print(f"Input size: {input_size}")
