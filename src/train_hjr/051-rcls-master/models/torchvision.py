import torch.nn as nn
import torchvision.models as models

class TorchVisionModel(nn.Module):
    def __init__(self, base_encoder: str = 'resnet18', num_classes: int = 1000):
        super(TorchVisionModel, self).__init__()
        print(f"Using torchvision {base_encoder} model!")
        
        # 获取基础模型
        if hasattr(models, base_encoder):
            self.encoder = getattr(models, base_encoder)(pretrained=True)
            
            # 获取最后一层的输入特征维度
            if base_encoder.startswith('resnet'):
                last_dim = self.encoder.fc.in_features
                self.encoder.fc = nn.Identity()  # 移除原始分类层
            elif base_encoder.startswith('densenet'):
                last_dim = self.encoder.classifier.in_features
                self.encoder.classifier = nn.Identity()
            else:
                raise ValueError(f"Model {base_encoder} not supported yet")
                
            # 添加新的分类层
            self.fc = nn.Linear(last_dim, num_classes)
        else:
            raise ValueError(f"Model {base_encoder} not found in torchvision.models")

    def forward(self, x):
        features = self.encoder(x)
        out = self.fc(features)
        return out 