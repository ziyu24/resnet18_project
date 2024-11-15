"""
Created on 11 14, 2024
@author: <Cui>
@brief: 创建定制的 Dataset
"""

import os
from PIL import Image
from torch.utils.data import Dataset


class DatasetSelf(Dataset):
    def __init__(self, data_dir, transform=None):
        super().__init__()
        assert (os.path.exists(data_dir)), "data_dir:{} 不存在！".format(data_dir)

        self.data_dir = data_dir
        self._get_img_info()
        print("file: {}, dataset len:{}".format(data_dir, len(self.img_info)))
        self.transform = transform

    def __getitem__(self, index):
        fn, label = self.img_info[index]
        img = Image.open(fn).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        if len(self.img_info) == 0:
            raise Exception("未获取任何图片路径，请检查 dataset 及文件路径！")
        return len(self.img_info)

    def _get_img_info(self):
        """
        获取数据集信息： 函数遍历指定的数据目录（self.data_dir），
        假设数据集的结构是每个子目录代表一个类别，每个子目录内存储对应类别的图像文件（.jpg）.
        对于每个子目录，获取该目录下所有 .png 格式的图像文件路径，
        并将每个图像文件的路径和该类别的标签（由子目录名称推导）
        以元组的形式存储在 self.img_info 中

        :return: 每个图像文件的路径和该类别的标签（由子目录名称推导）以元组的形式存储在 self.img_info 中
        """
        sub_dir_ = [name for name in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, name))]
        sub_dir = [os.path.join(self.data_dir, c) for c in sub_dir_]

        self.img_info = []
        for c_dir in sub_dir:
            path_img = [(os.path.join(c_dir, i), int(os.path.basename(c_dir))) for i in os.listdir(c_dir) if
                        i.endswith("jpg")]
            # print("iamge and index {}".format(path_img))
            self.img_info.extend(path_img)
