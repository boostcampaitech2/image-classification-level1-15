import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models
import timm


class MnasNet1(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()

        self.pretrained_model = models.mnasnet1_0(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        output = self.pretrained_model(x)
        output = self.fc(output)
        return output


class MnasNet2(BaseModel):
    def __init__(self, num_classes=3):
        super().__init__()

        self.pretrained_model = models.mnasnet1_0(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        output = self.pretrained_model(x)
        output = self.fc(output)
        return output


class MnasNet3(BaseModel):
    def __init__(self, num_classes=3):
        super().__init__()

        self.pretrained_model = models.mnasnet1_0(pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        output = self.pretrained_model(x)
        output = self.fc(output)
        return output


class EfficientNet1(BaseModel):
    def __init__(self, num_classes=2):
        super().__init__()

        self.pretrained_model = timm.create_model(
            'efficientnet_b3', pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        output = self.pretrained_model(x)
        output = self.fc(output)
        return output


class EfficientNet2(BaseModel):
    def __init__(self, num_classes=3):
        super().__init__()

        self.pretrained_model = timm.create_model(
            'efficientnet_b3', pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        output = self.pretrained_model(x)
        output = self.fc(output)
        return output


class EfficientNet3(BaseModel):
    def __init__(self, num_classes=3):
        super().__init__()

        self.pretrained_model = timm.create_model(
            'efficientnet_b3', pretrained=True)
        self.fc = nn.Linear(1000, num_classes)

    def forward(self, x):
        output = self.pretrained_model(x)
        output = self.fc(output)
        return output
