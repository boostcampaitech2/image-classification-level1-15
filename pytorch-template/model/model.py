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


class TotalModel(BaseModel):
    def __init__(self, num_classes=18, label_name="total", pretrained_model="efficientnet_b1"):
        super().__init__()

        gender_model, age_model, mask_model = self.init_model()

        self.gender_model = gender_model
        self.age_model = age_model
        self.mask_model = mask_model

        self.gender_model.fc = nn.Identity()
        self.age_model.fc = nn.Identity()
        self.mask_model.fc = nn.Identity()

        self.fc1 = nn.Linear(1000, 2)
        self.fc2 = nn.Linear(3000, 3)
        self.fc3 = nn.Linear(1000, 3)

    def forward(self, x):
        output_gender = self.gender_model(x.clone())
        output_gender = output_gender.view(output_gender.size(0), -1)
        output_gender = self.fc1(output_gender)

        output_age = self.age_model(x.clone())
        output_age = output_age.view(output_age.size(0), -1)
        output_age = self.fc2(output_age)

        output_mask = self.mask_model(x)
        output_mask = output_mask.view(output_mask.size(0), -1)
        output_mask = self.fc3(output_mask)

        output = torch.cat((output_gender, output_age, output_mask), dim=1)

        return output

    def init_model(self):
        gender_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_gender/0830_105923/checkpoint-epoch8.pth"
        age_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/age_ensemble/model_best.pth"
        mask_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_mask/0830_121746/checkpoint-epoch11.pth"

        gender_model_checkpoint = torch.load(gender_model_path)
        age_model_checkpoint = torch.load(
            age_model_path)
        mask_model_checkpoint = torch.load(mask_model_path)

        gender_model_state_dict = gender_model_checkpoint['state_dict']
        age_model_state_dict = age_model_checkpoint[
            'state_dict']
        mask_model_state_dict = mask_model_checkpoint['state_dict']

        gender_model = Model(
            num_classes=2, label_name='gender', pretrained_model='regnety_006')
        age_model = AgeModelEnsemble(
            num_classes=3, label_name='age', pretrained_model='efficientnet_b1')
        mask_model = Model(
            num_classes=3, label_name='mask', pretrained_model='regnety_006')

        gender_model.load_state_dict(gender_model_state_dict)
        age_model.load_state_dict(
            age_model_state_dict)
        mask_model.load_state_dict(mask_model_state_dict)

        for param in gender_model.parameters():
            param.requires_grad_(False)

        for param in age_model.parameters():
            param.requires_grad_(False)

        for param in mask_model.parameters():
            param.requires_grad_(False)

        return gender_model, age_model, mask_model


class AgeModelEnsemble(BaseModel):
    def __init__(self, num_classes=3, label_name="age", pretrained_model="efficientnet_b1"):
        super().__init__()

        none_mask_age_model, incorrect_mask_age_model, mask_age_model = self.init_model()

        self.none_mask_age_model = none_mask_age_model
        self.incorrect_mask_age_model = incorrect_mask_age_model
        self.mask_age_model = mask_age_model

        self.none_mask_age_model.fc = nn.Identity()
        self.incorrect_mask_age_model.fc = nn.Identity()
        self.mask_age_model.fc = nn.Identity()

        self.fc = nn.Linear(3000, num_classes)

    def forward(self, x):
        x1 = self.none_mask_age_model(x.clone())
        x1 = x1.view(x1.size(0), -1)
        x2 = self.incorrect_mask_age_model(x.clone())
        x2 = x2.view(x2.size(0), -1)
        x3 = self.mask_age_model(x)
        x3 = x3.view(x3.size(0), -1)
        x = torch.cat((x1, x2, x3), dim=1)

        x = self.fc(F.relu(x))
        return x

    def init_model(self):
        none_mask_age_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/none_mask_age/checkpoint-epoch36.pth"
        incorrect_mask_age_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/incorrect_mask_age/checkpoint-epoch33.pth"
        mask_age_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/mask_age/checkpoint-epoch43.pth"

        none_mask_age_model_checkpoint = torch.load(none_mask_age_model_path)
        incorrect_mask_age_model_checkpoint = torch.load(
            incorrect_mask_age_model_path)
        mask_age_model_checkpoint = torch.load(mask_age_model_path)

        none_mask_age_model_state_dict = none_mask_age_model_checkpoint['state_dict']
        incorrect_mask_age_model_state_dict = incorrect_mask_age_model_checkpoint[
            'state_dict']
        mask_age_model_state_dict = mask_age_model_checkpoint['state_dict']

        none_mask_age_model = Model(
            num_classes=3, label_name='age', pretrained_model='efficientnet_b1')
        incorrect_mask_age_model = Model(
            num_classes=3, label_name='age', pretrained_model='efficientnet_b1')
        mask_age_model = Model(
            num_classes=3, label_name='age', pretrained_model='efficientnet_b1')

        none_mask_age_model.load_state_dict(none_mask_age_model_state_dict)
        incorrect_mask_age_model.load_state_dict(
            incorrect_mask_age_model_state_dict)
        mask_age_model.load_state_dict(mask_age_model_state_dict)

        for param in none_mask_age_model.parameters():
            param.requires_grad_(False)

        for param in incorrect_mask_age_model.parameters():
            param.requires_grad_(False)

        for param in mask_age_model.parameters():
            param.requires_grad_(False)

        return none_mask_age_model, incorrect_mask_age_model, mask_age_model
