"""CIFAR100ResNet with Squeeze-and-Excitation attention."""

import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from config import SE_REDUCTION, DROPOUT, HIDDEN_DIM, BACKBONE_OUT, NUM_CLASSES


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention (Hu et al., 2018)."""

    def __init__(self, channel, reduction=SE_REDUCTION):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)          # squeeze
        y = self.fc(y).view(b, c, 1, 1)          # excitation
        return x * y.expand_as(x)                 # scale


class CIFAR100ResNet(nn.Module):
    """ResNet-18 backbone + SE attention + custom classifier head."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Truncate to get feature extractor (everything before fc)
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.se = SEBlock(BACKBONE_OUT)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(BACKBONE_OUT, HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM, num_classes),
        )

    def forward(self, x):
        x = self.features(x)          # (B, 512, H', W')
        x = self.se(x)                # (B, 512, H', W')  — recalibrated
        x = x.mean([2, 3])            # global average pool → (B, 512)
        x = self.classifier(x)        # (B, 100)
        return x
