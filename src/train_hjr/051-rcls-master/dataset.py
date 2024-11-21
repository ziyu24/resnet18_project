import torch.utils.data as data
from torchvision.datasets import ImageFolder
from typing import List, Optional, Callable
import torch

class CustomImageFolder(ImageFolder):
    """自定义 ImageFolder 数据集类,支持指定类别名称列表来映射标签
    
    Args:
        root (str): 数据集根目录路径
        class_names (List[str], optional): 类别名称列表,用于指定类别到标签的映射。
                                         如果不指定,则使用目录名称按字母顺序排序映射。
        transform (callable, optional): 图像转换函数
        target_transform (callable, optional): 标签转换函数 一般不指定
    """
    def __init__(
        self,
        root: str,
        class_names: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        # 首先调用父类的初始化
        super().__init__(root, transform=transform, target_transform=target_transform)
        
        if class_names is not None:
            # 验证传入的类别名称是否都存在
            for class_name in class_names:
                if class_name not in self.class_to_idx:
                    raise ValueError(f"类别 '{class_name}' 在数据集目录中不存在")
            
            # 创建新的类别到索引的映射
            new_class_to_idx = {name: idx for idx, name in enumerate(class_names)}
            
            # 更新类别列表和映射
            self.classes = class_names
            old_class_to_idx = self.class_to_idx
            self.class_to_idx = new_class_to_idx
            
            # 更新样本列表中的标签
            self.samples = [(path, new_class_to_idx[self.classes[old_target]]) 
                          for path, old_target in self.samples]
            self.targets = [s[1] for s in self.samples]
            
        # 打印类别信息
        print(f"数据集加载完成,共 {len(self.classes)} 个类别:")
        # for idx, class_name in enumerate(self.classes):
        #     print(f"  类别 {idx}: {class_name}")

    def __getitem__(self, index: int):
        """获取一个样本
        
        Args:
            index (int): 索引
            
        Returns:
            tuple: (sample, target) 其中target是类别索引
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target
