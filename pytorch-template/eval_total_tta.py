import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
import pandas as pd
from torchvision import models
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
import numpy as np
from model.model import Model, AgeModelEnsemble, TotalModel


def get_saved_model_state_dict():
    latest_saved_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/total_ensemble/0831_232118/checkpoint-epoch2.pth"
    checkpoint = torch.load(latest_saved_model_path)
    state_dict = checkpoint['state_dict']
    return state_dict


def test_time_augmentation(model, images):
    images = torch.split(images, 3, dim=1)
    for i in range(len(images)):
        if i == 0:
            preds_gender = model(images[i])[:, :2]
            preds_age = model(images[i])[:, 2:5]
            preds_mask = model(images[i])[:, 5:]
        else:
            pred_gender = model(images[i])[:, :2]
            pred_age = model(images[i])[:, 2:5]
            pred_mask = model(images[i])[:, 5:]

            preds_mask = torch.stack((preds_mask, pred_mask), dim=1)
            preds_gender = torch.stack((preds_gender, pred_gender), dim=1)
            preds_age = torch.stack((preds_age, pred_age), dim=1)
    return torch.mean(preds_gender, dim=1), torch.mean(preds_age, dim=1), torch.mean(preds_mask, dim=1)


class EvalDataset(Dataset):
    def __init__(self, img_paths, augs, transform):
        self.img_paths = img_paths
        self.augs = augs
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        for i in range(len(self.augs) + 1):
            if i == 0:
                images = self.transform(image=np.array(image))['image']
            else:
                image = self.augs[i - 1](image=np.array(image))['image']
                images = torch.cat(
                    (images, self.transform(image=image)['image']), dim=0)
        return images

    def __len__(self):
        return len(self.img_paths)

    # transform = albumentations.Compose([
    #     albumentations.Resize(224, 224),
    #     albumentations.Normalize(
    #         mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.245)),
    #     albumentations.pytorch.transforms.ToTensorV2()
    # ])
    # augs = [
    #     albumentations.HorizontalFlip()
    #     # albumentations.ColorJitter(brightness=(0.2, 2), contrast=(
    #     #     0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3))
    # ]


def main():

    test_dir = 'data/input/data/eval'
    image_dir = os.path.join(test_dir, 'mtcnn_only_crop_images')
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))

    image_paths = [os.path.join(image_dir, img_id)
                   for img_id in submission.ImageID]
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=(0.560, 0.524, 0.501), std=(0.233, 0.243, 0.245)),
        A.pytorch.transforms.ToTensorV2()
    ])
    augs = [
        A.CoarseDropout(always_apply=False, p=1.0, max_holes=34, max_height=14,
                        max_width=14, min_holes=20, min_height=1, min_width=1)
        # A.OneOf(transform_list, p=0.5)
        # albumentations.ColorJitter(brightness=(0.2, 2), contrast=(
        #     0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3))
    ]

    testset = EvalDataset(image_paths, augs, transform)
    data_loader = DataLoader(testset, batch_size=64, shuffle=False)

    print()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TotalModel(num_classes=18, label_name="total")
    state_dict = get_saved_model_state_dict()

    model.load_state_dict(state_dict)
    # prepare model for testing
    model = model.to(device)

    model.eval()

    submission = pd.read_csv('data/input/data/eval/info.csv')

    gender_preds = []
    age_preds = []
    mask_preds = []

    with torch.no_grad():
        for i, images in enumerate(tqdm(data_loader)):
            images = images.to(device)
            pred_gender, pred_age, pred_mask = test_time_augmentation(
                model, images)

            pred1 = pred_gender.argmax(dim=-1)
            pred2 = pred_age.argmax(dim=-1)
            pred3 = pred_mask.argmax(dim=-1)

            gender_preds.extend(pred1.cpu().numpy())
            age_preds.extend(pred2.cpu().numpy())
            mask_preds.extend(pred3.cpu().numpy())

    CLASS_DICT = {
        '000': 0, '001': 1, '002': 2, '010': 3, '011': 4, '012': 5,
        '100': 6, '101': 7, '102': 8, '110': 9, '111': 10, '112': 11,
        '200': 12, '201': 13, '202': 14, '210': 15, '211': 16, '212': 17
    }

    preds = zip(gender_preds, age_preds, mask_preds)
    labels = [CLASS_DICT[''.join(map(str, [mask, gender, age]))]
              for gender, age, mask in preds]

    submission['ans'] = labels
    submission.to_csv('age_ensemble.csv', index=False)


if __name__ == '__main__':
    main()
