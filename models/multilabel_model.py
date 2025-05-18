import torch
import torch.nn as nn
import torchvision.models as models

class MultiLabelResNet(nn.Module):
    """
        Multi-label classification model using ResNet
    """

    def __init__(self, output_dim):
        super(MultiLabelResNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
            Forward pass through the model
        """
        return self.backbone(x)