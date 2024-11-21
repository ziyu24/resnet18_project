import torch.nn as nn
import pretrainedmodels

class PretrainedModel(nn.Module):
    def __init__(self, base_encoder: str = 'resnet50', num_classes: int = 1000):
        super(PretrainedModel, self).__init__()
        print(f"Using pretrainedmodels {base_encoder} model!")
        
        # 获取基础模型
        try:
            self.encoder = pretrainedmodels.__dict__[base_encoder](num_classes=1000, pretrained='imagenet')
            
            # 获取最后一层的输入特征维度
            last_dim = self.encoder.last_linear.in_features
            
            # 移除原始分类层
            self.encoder.last_linear = nn.Identity()
            
            # 添加新的分类层
            self.fc = nn.Linear(last_dim, num_classes)
            
        except KeyError:
            raise ValueError(f"Model {base_encoder} not found in pretrainedmodels")

    def forward(self, x):
        features = self.encoder(x)
        out = self.fc(features)
        return out