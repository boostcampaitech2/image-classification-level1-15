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
        self.fc = nn.Linear(1000, num_classes)

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
