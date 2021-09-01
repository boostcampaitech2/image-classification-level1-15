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
import albumentations as A
import numpy as np
from model.model import Model, AgeModelEnsemble


def init_models(config):
    models = [
        config.eval_init_obj('arch', module_arch, 1),
        config.eval_init_obj('arch', module_arch, 2),
        config.eval_init_obj('arch', module_arch, 3)
    ]
    return models


def get_latest_saved_model_paths(config):
    checkpoint_path = "/opt/ml/level1-15/pytorch-template/saved/models/"
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


        # NOTE: BEST SCORE -> Evaluation using Crop Image F1: 0.7334
        "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_gender/0830_105923/checkpoint-epoch8.pth",
        # "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_age/0830_120150/checkpoint-epoch9.pth",
        "/opt/ml/level1-15/pytorch-template/saved/models/age_ensemble/model_best.pth",
        '/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_mask/0830_121746/checkpoint-epoch11.pth'
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
    for i in range(len(images)):
        if i == 0:
            preds_gender = model1(images[i])
            preds_age = model2(images[i])
            preds_mask = model3(images[i])
        else:
            pred_gender = model1(images[i])
            pred_age = model2(images[i])
            pred_mask = model3(images[i])
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


def main(config):
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
        A.HorizontalFlip(),
        # A.OneOf(transform_list, p=0.5)
        # albumentations.ColorJitter(brightness=(0.2, 2), contrast=(
        #     0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3))
    ]

    testset = EvalDataset(image_paths, augs, transform)
    data_loader = DataLoader(testset, batch_size=64, shuffle=False)

    print()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model1, model2, model3 = init_models(config)
    model1 = Model(num_classes=2, label_name="gender",
                   pretrained_model="regnety_006")
    model2 = AgeModelEnsemble(num_classes=3, label_name="age")
    model3 = Model(num_classes=3, label_name="mask",
                   pretrained_model="regnety_006")

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

    submission = pd.read_csv(
        config['data_loader']['args']['data_dir'] + 'eval/info.csv')

    gender_preds = []
    age_preds = []
    mask_preds = []

    with torch.no_grad():
        for i, images in enumerate(tqdm(data_loader)):
            images = images.to(device)

            pred_gender, pred_age, pred_mask = test_time_augmentation(
                model1, model2, model3, images)

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
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
