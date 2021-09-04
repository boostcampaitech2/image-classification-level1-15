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
from albumentations.pytorch import ToTensorV2
import albumentations as A
import numpy as np

transform_list = [
    A.Compose([
        A.CenterCrop(height=358, width=268, p=1),
    ]),
    A.Compose([
        A.CenterCrop(height=256, width=192, p=1),
    ]),
    A.Compose([
        A.CenterCrop(height=196, width=160, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
    ]),
    A.Compose([
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.CenterCrop(height=358, width=268, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.CenterCrop(height=358, width=268, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.CenterCrop(height=256, width=192, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.CenterCrop(height=256, width=192, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.CenterCrop(height=196, width=160, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.CenterCrop(height=196, width=160, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.CenterCrop(height=358, width=268, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.CenterCrop(height=358, width=268, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.CenterCrop(height=256, width=192, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.CenterCrop(height=256, width=192, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.RandomCrop(height=196, width=160, p=1)
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.RandomCrop(height=196, width=160, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=358, width=268, p=1),
    ]),
    A.Compose([
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=358, width=268, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=256, width=192, p=1),
    ]),
    A.Compose([
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=256, width=192, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.RandomScale(scale_limit=0.3, p=1),
        A.RandomCrop(height=196, width=160, p=1)
    ]),
    A.Compose([
        A.RandomScale(scale_limit=0.3, p=1),
        A.RandomCrop(height=196, width=160, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=358, width=268, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=358, width=268, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=256, width=192, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=256, width=192, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.RandomCrop(height=196, width=160, p=1)
    ]),
    A.Compose([
        A.Rotate(limit=20, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.RandomCrop(height=196, width=160, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=358, width=268, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=358, width=268, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=256, width=192, p=1),
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.CenterCrop(height=256, width=192, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.RandomCrop(height=196, width=160, p=1)
    ]),
    A.Compose([
        A.Rotate(limit=10, p=1),
        A.RandomScale(scale_limit=0.3, p=1),
        A.RandomCrop(height=196, width=160, p=1),
        A.GridDistortion(always_apply=False, p=1.0, num_steps=5, distort_limit=(
            -0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None)
    ])
]


def init_models(config):
    models = [
        config.eval_init_obj('arch', module_arch, 1),
        config.eval_init_obj('arch', module_arch, 2),
        config.eval_init_obj('arch', module_arch, 3),
        config.eval_init_obj('arch', module_arch, 4),
        config.eval_init_obj('arch', module_arch, 5)
    ]
    return models


def get_latest_saved_model_paths(config):
    checkpoint_path = "/opt/ml/level1-15/pytorch-template/saved/models/"
    save_paths = [
        checkpoint_path + config['save_directory_name']['gender'],
        checkpoint_path + config['save_directory_name']['none_mask_age'],
        checkpoint_path + config['save_directory_name']['incorrect_mask_age'],
        checkpoint_path + config['save_directory_name']['mask_age'],
        checkpoint_path + config['save_directory_name']['mask']
    ]

    latest_saved_directory = [
        sorted(os.listdir(save_paths[0]))[-1],
        sorted(os.listdir(save_paths[1]))[-1],
        sorted(os.listdir(save_paths[2]))[-1],
        sorted(os.listdir(save_paths[3]))[-1],
        sorted(os.listdir(save_paths[4]))[-1]
    ]

    latest_saved_model_paths = [
        "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_gender/0830_105923/checkpoint-epoch8.pth",
        "/opt/ml/level1-15/pytorch-template/saved/models/none_mask_age/checkpoint-epoch36.pth",
        "/opt/ml/level1-15/pytorch-template/saved/models/incorrect_mask_age/checkpoint-epoch33.pth",
        "/opt/ml/level1-15/pytorch-template/saved/models/mask_age/0902_012534/checkpoint-epoch10.pth",
        "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_mask/0830_121746/checkpoint-epoch11.pth"
    ]
    return latest_saved_model_paths


def get_saved_model_state_dict(latest_saved_model_paths):
    checkpoint1 = torch.load(latest_saved_model_paths[0])
    state_dict1 = checkpoint1['state_dict']
    checkpoint2 = torch.load(latest_saved_model_paths[1])
    state_dict2 = checkpoint2['state_dict']
    checkpoint3 = torch.load(latest_saved_model_paths[2])
    state_dict3 = checkpoint3['state_dict']
    checkpoint4 = torch.load(latest_saved_model_paths[3])
    state_dict4 = checkpoint4['state_dict']
    checkpoint5 = torch.load(latest_saved_model_paths[4])
    state_dict5 = checkpoint5['state_dict']

    return [state_dict1, state_dict2, state_dict3, state_dict4, state_dict5]


def test_time_augmentation(model1, model2, model3, model4, model5, images):
    # print(f'here >>>>>>> {images.shape} {len(images)}')
    # images = [batch_size, (channel RGB 3 * total aug_num), W, H]
    images = torch.split(images, 3, dim=1)

    # after split images = [64, 3, 224, 224] * aug_num
    # print(f'here ****************** {images[0].shape} {len(images)}')
    for i in range(len(images)):
        if i == 0:
            preds_gender = model1(images[i])
            preds_age1 = model2(images[i])
            preds_age2 = model3(images[i])
            preds_age3 = model4(images[i])
            preds_mask = model5(images[i])
        else:
            pred_gender = model1(images[i])
            pred_age1 = model2(images[i])
            pred_age2 = model3(images[i])
            pred_age3 = model4(images[i])
            pred_mask = model5(images[i])
            preds_mask = torch.stack((preds_mask, pred_mask), dim=1)
            preds_gender = torch.stack((preds_gender, pred_gender), dim=1)
            preds_age1 = torch.stack((preds_age1, pred_age1), dim=1)
            preds_age2 = torch.stack((preds_age2, pred_age2), dim=1)
            preds_age3 = torch.stack((preds_age3, pred_age3), dim=1)
            # print(preds_mask.shape)
            # if batch == 1 preds [2, 3] [aug_num, class num]
            # if batch == 64 preds [128, 3] [aug_num*batch, class num]
    return torch.mean(preds_gender, dim=1), torch.mean(preds_age1, dim=1), torch.mean(preds_age2, dim=1), torch.mean(preds_age3, dim=1), torch.mean(preds_mask, dim=1)


class EvalDataset(Dataset):
    def __init__(self, img_paths, augs, transform):
        self.img_paths = img_paths
        self.augs = augs
        self.transform = transform
        # self.crop_img_dir = 'data/input/data/eval/crop_images'
        # s = pd.read_csv('data/input/data/eval/info.csv')
        # self.crop_imag_paths = [os.path.join(
        #     self.crop_img_dir, img_id) for img_id in s.ImageID]

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        for i in range(len(self.augs) + 1):
            if i == 0:
                images = self.transform(image=np.array(image))['image']
            else:
                image = self.augs[i - 1](image=np.array(image))['image']
                images = torch.cat(
                    (images, self.transform(image=image)['image']), dim=0)

        # crop_image = Image.open(self.crop_imag_paths[index])
        # images = torch.cat((images, self.transform(
        #     image=np.array(crop_image))['image']))
        return images

    def __len__(self):
        return len(self.img_paths)


def main(config):
    test_dir = 'data/input/data/eval'
    image_dir = os.path.join(test_dir, 'crop_images')
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

    model1, model2, model3, model4, model5 = init_models(config)
    latest_saved_model_paths = get_latest_saved_model_paths(config)
    state_dict1, state_dict2, state_dict3, state_dict4, state_dict5 = get_saved_model_state_dict(
        latest_saved_model_paths)

    model1.load_state_dict(state_dict1)
    model2.load_state_dict(state_dict2)
    model3.load_state_dict(state_dict3)
    model4.load_state_dict(state_dict4)
    model5.load_state_dict(state_dict5)

    # prepare model for testing
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model4 = model4.to(device)
    model5 = model5.to(device)

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    # submission = pd.read_csv(
    #     config['data_loader']['args']['data_dir'] + 'eval/info.csv')

    gender_preds = []
    age_preds1 = []
    age_preds2 = []
    age_preds3 = []
    mask_preds = []

    with torch.no_grad():
        for i, images in enumerate(tqdm(data_loader)):
            images = images.to(device)
            # print(f'image {images.shape}')
            pred_gender, pred_age1, pred_age2, pred_age3, pred_mask = test_time_augmentation(
                model1, model2, model3, model4, model5, images)
            # print(f'shape !!!!!!! {pred_gender.shape}')
            pred1 = pred_gender.argmax(dim=-1)
            pred2 = pred_age1.argmax(dim=-1)
            pred3 = pred_age2.argmax(dim=-1)
            pred4 = pred_age3.argmax(dim=-1)
            pred5 = pred_mask.argmax(dim=-1)

            gender_preds.extend(pred1.cpu().numpy())
            age_preds1.extend(pred2.cpu().numpy())
            age_preds2.extend(pred3.cpu().numpy())
            age_preds3.extend(pred4.cpu().numpy())
            mask_preds.extend(pred5.cpu().numpy())

            # if i == 2:
            #     break
    CLASS_DICT = {
        '000': 0, '001': 1, '002': 2, '010': 3, '011': 4, '012': 5,
        '100': 6, '101': 7, '102': 8, '110': 9, '111': 10, '112': 11,
        '200': 12, '201': 13, '202': 14, '210': 15, '211': 16, '212': 17
    }

    age_preds = []
    for i in range(len(mask_preds)):
        if mask_preds[i] == 0:
            age_preds.append(age_preds3[i])
        elif mask_preds[i] == 1:
            age_preds.append(age_preds2[i])
        elif mask_preds[i] == 2:
            age_preds.append(age_preds1[i])

    preds = zip(gender_preds, age_preds, mask_preds)
    labels = [CLASS_DICT[''.join(map(str, [mask, gender, age]))]
              for gender, age, mask in preds]

    submission['ans'] = labels
    submission.to_csv('split_age_tta_crop.csv', index=False)


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
