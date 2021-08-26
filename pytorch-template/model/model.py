import torch
import torch.nn as nn
import torch.nn.functional as F
from base import BaseModel
from torchvision import models
import timm


# 3 * 512 * 384
class Model(BaseModel):
    def __init__(self, num_classes=2, label_name="gender", pretrained_model='efficientnet_b1'):
        self.check_args(num_classes, label_name, pretrained_model)

        super().__init__()
        self.pretrained_model = timm.create_model(
            pretrained_model, pretrained=True)
        self.fc = nn.Linear(
            self.pretrained_model.classifier.out_features, num_classes)

    def forward(self, x):
        output = self.pretrained_model(x)
        output = self.fc(output)
        return output

    def check_args(self, num_classes, label_name, pretrained_model):
        if label_name not in ["gender", "age", "mask", "total"]:
            assert "Label Name is incorrect.\n\tLabel name is one of [gender, age, mask, total]."
        if not timm.is_model(pretrained_model):
            assert "Model does not create from timm.\nPlease check the model name."
        if num_classes == 2 and label_name != "gender":
            assert "Num Classes and Label Name are not matched."
        elif num_classes == 3 and label_name not in ["age", "mask"]:
            assert "Num Classes and Label Name are not matched."
        elif num_classes == 18 and label_name != "total":
            assert "Num Classes and Label Name are not matched."


class FastModel(BaseModel):
    def __init__(self, num_classes=2, label_name="gender", pretrained_model='efficientnet_b1', pretrained_out_feature=1000):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 30, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(30),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.fc = nn.Linear(12 * 12 * 30, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out
