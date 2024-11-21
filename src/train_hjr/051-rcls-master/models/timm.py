import timm
import torch.nn as nn

class TimmModel(nn.Module):
    def __init__(
        self, 
        base_encoder: str = 'resnet18', 
        num_classes: int = 1000,
        in_chans: int = 3,
        pretrained: bool = True
    ):
        super(TimmModel, self).__init__()
        print(f"Using timm {base_encoder} model!")

        # 使用timm创建基础模型
        self.encoder = timm.create_model(
            base_encoder,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头
            in_chans=in_chans  # 支持自定义输入通道数
        )
        
        # 获取特征维度
        if hasattr(self.encoder, 'num_features'):
            last_dim = self.encoder.num_features
        else:
            # 对于某些模型,可能需要手动指定特征维度
            last_dim = self.encoder.get_classifier().in_features
            
        # 添加新的分类层
        self.fc = nn.Linear(last_dim, num_classes)

    def forward(self, x):
        features = self.encoder(x)
        out = self.fc(features)
        return out

    @staticmethod
    def list_available_models():
        """列出所有可用的预训练模型"""
        return timm.list_models(pretrained=True)