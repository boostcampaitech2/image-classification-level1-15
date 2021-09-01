import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import pandas as pd
from torchvision import models
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations.pytorch
import numpy as np
from sklearn.metrics import f1_score

lista=[5 ,5 ,5 ,5 ,11 ,17 ,5 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,2 ,2 ,2 ,2 ,8 ,14 ,2 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,5 ,5 ,5 ,5 ,11 ,17 ,5 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,4 ,4 ,4 ,4 ,10 ,16 ,4 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,1 ,1 ,1 ,1 ,7 ,13 ,1 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,3 ,3 ,3 ,3 ,9 ,15 ,3 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0 ,0 ,0 ,0 ,0 ,6 ,12 ,0]
def init_models(config):
    models = [
        config.eval_init_obj('arch', module_arch, 1),
        config.eval_init_obj('arch', module_arch, 2),
        config.eval_init_obj('arch', module_arch, 3)
    ]
    return models


def get_latest_saved_model_paths(config):
    checkpoint_path = "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/"
    save_paths = [
        checkpoint_path + config['save_directory_name']['gender'],
        checkpoint_path + config['save_directory_name']['age'],
        checkpoint_path + config['save_directory_name']['mask']
    ]

    latest_saved_directory = [
        sorted(os.listdir(save_paths[0]))[-1],
        sorted(os.listdir(save_paths[1]))[-1],
        sorted(os.listdir(save_paths[2]))[-1]
    ]

    latest_saved_model_paths = [
        save_paths[0] + "/" + latest_saved_directory[0] + "/model_best.pth",
        save_paths[1] + "/" + latest_saved_directory[1] + "/model_best.pth",
        save_paths[2] + "/" + latest_saved_directory[2] + "/model_best.pth"
    ]
    return latest_saved_model_paths


def get_saved_model_state_dict(latest_saved_model_paths):
    checkpoint1 = torch.load(latest_saved_model_paths[0])
    state_dict1 = checkpoint1['state_dict']
    checkpoint2 = torch.load(latest_saved_model_paths[1])
    state_dict2 = checkpoint2['state_dict']
    checkpoint3 = torch.load(latest_saved_model_paths[2])
    state_dict3 = checkpoint3['state_dict']

    return [state_dict1, state_dict2, state_dict3]


def test_time_augmentation(model1, model2, model3, images):
    images = torch.split(images, 3, dim=1)
    list_gender = []
    list_age = []
    list_mask = []
    for i in range(len(images)):
        if i == 0:
            pred_gender = model1(images[i])
            pred_age = model2(images[i])
            pred_mask = model3(images[i])
            list_gender.append(pred_gender)
            list_age.append(pred_age)
            list_mask.append(pred_mask)
        else:
            pred_gender = model1(images[i])
            # print('test',pred_gender.shape)
            pred_age = model2(images[i])
            pred_mask = model3(images[i])
            list_gender.append(pred_gender)
            list_age.append(pred_age)
            list_mask.append(pred_mask)

    preds_gender = torch.stack(list_gender, dim=2)
    preds_age = torch.stack(list_age, dim=2)
    preds_mask = torch.stack(list_mask, dim=2)
    # print('-',preds_gender.shape)
    preds_gender = torch.mean(preds_gender, dim=2)
    preds_age = torch.mean(preds_age, dim=2)
    preds_mask = torch.mean(preds_mask, dim=2)
    # print(preds_gender.shape)
    # print(preds_gender)
    return preds_gender, preds_age, preds_mask


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


def f1(output, target):
    # output = output.argmax(dim=1)
    return f1_score(target, output, average='macro')


def main(config):
    # test_dir = './data/input/data/eval'
    # image_dir = os.path.join(test_dir, 'crop_images')
    # submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))

    # image_paths = [os.path.join(image_dir, img_id)
    #                for img_id in submission.ImageID]

    val_dir = './data/input/data'
    image_dir = os.path.join(val_dir, 'train/crop_images')
    val_split = pd.read_csv(os.path.join(
        val_dir, 'f1.csv'))

    number = ['mask3', 'mask2', 'mask5', 'mask4', 'incorrect_mask', 'normal', 'mask1']
    image_paths = [os.path.join(image_dir, img_id + '/' + number[i % 7] + '.jpg')
                   for i, img_id in enumerate(val_split.folder)]
    # answer = []
    print(image_paths)
    # exit(0)
    transform = albumentations.Compose([
        albumentations.Resize(224, 224),
        albumentations.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        albumentations.pytorch.transforms.ToTensorV2()
    ])

    augs = [
        albumentations.OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.5199999809265137, 0.5199999809265137),
                                         shift_limit=(-0.08999999612569809, -0.03999999910593033), interpolation=2, border_mode=1, value=(0, 0, 0), mask_value=None),
        albumentations.CLAHE(always_apply=False, p=1.0,
                             clip_limit=(1, 2), tile_grid_size=(3, 3))
    ]
    # CoarseDropout(always_apply=False, p=1.0, max_holes=34, max_height=14, max_width=14, min_holes=20, min_height=1, min_width=1)
    # OpticalDistortion(always_apply=False, p=1.0, distort_limit=(-0.5199999809265137, 0.5199999809265137), shift_limit=(-0.08999999612569809, -0.03999999910593033), interpolation=2, border_mode=1, value=(0, 0, 0), mask_value=None)
    # Equalize(always_apply=False, p=1., mode='cv', by_channels=False)
    # CLAHE(always_apply=False, p=1.0, clip_limit=(1, 2), tile_grid_size=(3, 3))
    # RandomBrightnessContrast(always_apply=False, p=1.0, brightness_limit=(-0.12999999523162842, 0.10999999940395355), contrast_limit=(-0.11999999731779099, 0.10999999940395355), brightness_by_max=True)
    # HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-11, 8), sat_shift_limit=(-23, 11), val_shift_limit=(-7, 17))
    # aug_class = ['testset1', 'testset2', 'testset3']
    testset = EvalDataset(image_paths, augs, transform)

    data_loader = DataLoader(testset, batch_size=64, shuffle=False)

    # print()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model1, model2, model3 = init_models(config)
    latest_saved_model_paths = get_latest_saved_model_paths(config)
    state_dict1, state_dict2, state_dict3 = get_saved_model_state_dict(
        latest_saved_model_paths)

    model1.load_state_dict(state_dict1)
    model2.load_state_dict(state_dict2)
    model3.load_state_dict(state_dict3)

    # prepare model for testing
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)

    model1.eval()
    model2.eval()
    model3.eval()

    # submission = pd.read_csv(
    #     config['data_loader']['args']['data_dir'] + 'eval/info.csv')

    gender_preds = []
    age_preds = []
    mask_preds = []

    with torch.no_grad():
        for i, images in enumerate(tqdm(data_loader)):

            images = images.to(device)
            # print(f'image {images.shape}')
            pred_gender, pred_age, pred_mask = test_time_augmentation(
                model1, model2, model3, images)
            # print(f'shape !!!!!!! {pred_gender.shape}')
            pred1 = pred_gender.argmax(dim=-1)
            pred2 = pred_age.argmax(dim=-1)
            pred3 = pred_mask.argmax(dim=-1)

            gender_preds.extend(pred1.cpu().numpy())
            age_preds.extend(pred2.cpu().numpy())
            mask_preds.extend(pred3.cpu().numpy())
            # if i == 2:
            #     break
    CLASS_DICT = {
        '000': 0, '001': 1, '002': 2, '010': 3, '011': 4, '012': 5,
        '100': 6, '101': 7, '102': 8, '110': 9, '111': 10, '112': 11,
        '200': 12, '201': 13, '202': 14, '210': 15, '211': 16, '212': 17
    }
    # print(gender_preds)
    # print(f1(labels, answer))

    preds = zip(gender_preds, age_preds, mask_preds)
    labels = [CLASS_DICT[''.join(map(str, [mask, gender, age]))]
              for gender, age, mask in preds]
    print(f1(labels, lista))
    # submission['ans'] = labels
    # submission.to_csv('./infinite_test/8_1.csv', index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
