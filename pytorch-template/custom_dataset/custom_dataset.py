from albumentations.augmentations.transforms import HorizontalFlip
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch


# https://github.com/utkuozbulak/pytorch-custom-dataset-examples#incorporating-pandas
class CustomDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transform, train=True):
        self.train = train

        self.data_dir = data_dir + \
            'train/images/' if self.train else data_dir + 'eval/eval_crop/'
        self.csv_path = csv_path if self.train else data_dir + 'eval/info.csv'

        # Transforms
        self.transform = transform
        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path)

        if self.train:  # Train Dataset
            self.crop_dataset_start_index = len(self.data_info)
            self.crop_dir = data_dir + 'train/train_total_crop/'
            self.crop_data_info = pd.read_csv(csv_path)
            self.horizon_dir = data_dir + 'train/train_horizon/'
            self.horizon_data_info = pd.read_csv(csv_path)

            self.image_arr = np.append(np.asarray(
                self.data_dir + self.data_info['folder'] + '/' + self.data_info['path']),
                np.asarray(
                self.horizon_dir + self.horizon_data_info['folder'] + '_' + self.horizon_data_info['path']))
            self.image_arr = np.append(self.image_arr,
                                       np.asarray(self.crop_dir +
                                                  self.crop_data_info['folder'] + '_' + self.crop_data_info['path']))

            self.label_arr = np.append(
                np.asarray(self.data_info['label']),
                np.asarray(self.crop_data_info['label']))
            self.label_arr = np.append(
                self.label_arr, np.asarray(self.horizon_data_info['label']))

            self.gender_label_arr = np.append(
                np.asarray(self.data_info['gender_label']),
                np.asarray(self.crop_data_info['gender_label']))
            self.gender_label_arr = np.append(self.gender_label_arr,
                                              np.asarray(self.horizon_data_info['gender_label']))

            self.age_label_arr = np.append(
                np.asarray(self.data_info['age_label']),
                np.asarray(self.crop_data_info['age_label']))
            self.age_label_arr = np.append(self.age_label_arr,
                                           np.asarray(self.horizon_data_info['age_label']))

            self.mask_label_arr = np.append(
                np.asarray(self.data_info['mask_label']),
                np.asarray(self.crop_data_info['mask_label']))
            self.mask_label_arr = np.append(self.mask_label_arr,
                                            np.asarray(self.horizon_data_info['mask_label']))

        else:
            eval_crop_data_info = pd.read_csv(self.csv_path)
            self.eval_crop_image_arr = np.asarray(
                data_dir + 'eval/crop_images/' + eval_crop_data_info['ImageID'])

            self.image_arr = np.asarray(
                self.data_dir + self.data_info['ImageID'])

        if train:
            # Calculate len
            self.data_len = len(self.data_info.index) +\
                len(self.crop_data_info.index) + \
                len(self.horizon_data_info.index)
        else:
            self.data_len = len(self.data_info.index)

        self.base_transform = A.Compose([
            A.Resize(240, 240),
            A.Normalize(mean=(0.445, 0.47, 0.52), std=(0.248, 0.24, 0.235)),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        if self.train:
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = np.array(Image.open(single_image_name))
            # Transform image to tensor
            if index < self.crop_dataset_start_index:
                transform_image = self.transform(image=img_as_img)
            else:
                transform_image = self.base_transform(image=img_as_img)
            # Get label(class) of the image based on the cropped pandas column
            single_image_label = self.get_total_label(index)

            single_gender_label = self.gender_label_arr[index]
            single_age_label = self.age_label_arr[index]
            single_mask_label = self.mask_label_arr[index]

            return (transform_image['image'], single_image_label, single_gender_label, single_age_label, single_mask_label)

        else:
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = np.array(Image.open(single_image_name))
            # Transform image to tensor
            # if self.transform is not None:
            # transform_image = self.transform(image=img_as_img)
            transform_image = self.base_transform(image=img_as_img)
            return transform_image['image']

    def __len__(self):
        return self.data_len

    def get_total_label(self, index):
        if self.label_arr[index] == 0:
            single_image_label = torch.tensor(
                [1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 1:
            single_image_label = torch.tensor(
                [1, 0, 0, 1, 0, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 2:
            single_image_label = torch.tensor(
                [1, 0, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 3:
            single_image_label = torch.tensor(
                [0, 1, 1, 0, 0, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 4:
            single_image_label = torch.tensor(
                [0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 5:
            single_image_label = torch.tensor(
                [0, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 6:
            single_image_label = torch.tensor(
                [1, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 7:
            single_image_label = torch.tensor(
                [1, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 8:
            single_image_label = torch.tensor(
                [1, 0, 0, 0, 1, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 9:
            single_image_label = torch.tensor(
                [0, 1, 1, 0, 0, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 10:
            single_image_label = torch.tensor(
                [0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 11:
            single_image_label = torch.tensor(
                [0, 1, 0, 0, 1, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 12:
            single_image_label = torch.tensor(
                [1, 0, 1, 0, 0, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 13:
            single_image_label = torch.tensor(
                [1, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 14:
            single_image_label = torch.tensor(
                [1, 0, 0, 0, 1, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 15:
            single_image_label = torch.tensor(
                [0, 1, 1, 0, 0, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 16:
            single_image_label = torch.tensor(
                [0, 1, 0, 1, 0, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 17:
            single_image_label = torch.tensor(
                [0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.float32)
        return single_image_label


class CustomValidDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transform):
        self.data_dir = data_dir + 'train/train_total_crop/'
        self.csv_path = csv_path

        self.transform = transform
        self.data_info = pd.read_csv(self.csv_path)

        self.image_arr = np.asarray(
            self.data_dir + self.data_info['folder'] + '_' + self.data_info['path'])

        self.label_arr = np.asarray(self.data_info['label'])
        self.gender_label_arr = np.asarray(
            self.data_info['gender_label'])
        self.age_label_arr = np.asarray(
            self.data_info['age_label'])
        self.mask_label_arr = np.asarray(
            self.data_info['mask_label'])

        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        img_as_img = np.array(Image.open(single_image_name))

        transform_image = self.transform(image=img_as_img)

        single_image_label = self.get_total_label(index)
        single_gender_label = self.gender_label_arr[index]
        single_age_label = self.age_label_arr[index]
        single_mask_label = self.mask_label_arr[index]
        return (transform_image['image'], single_image_label,
                single_gender_label, single_age_label, single_mask_label)

    def __len__(self):
        return self.data_len

    def get_total_label(self, index):
        if self.label_arr[index] == 0:
            single_image_label = torch.tensor(
                [1, 0, 1, 0, 0, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 1:
            single_image_label = torch.tensor(
                [1, 0, 0, 1, 0, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 2:
            single_image_label = torch.tensor(
                [1, 0, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 3:
            single_image_label = torch.tensor(
                [0, 1, 1, 0, 0, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 4:
            single_image_label = torch.tensor(
                [0, 1, 0, 1, 0, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 5:
            single_image_label = torch.tensor(
                [0, 1, 0, 0, 1, 1, 0, 0], dtype=torch.float32)
        elif self.label_arr[index] == 6:
            single_image_label = torch.tensor(
                [1, 0, 1, 0, 0, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 7:
            single_image_label = torch.tensor(
                [1, 0, 0, 1, 0, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 8:
            single_image_label = torch.tensor(
                [1, 0, 0, 0, 1, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 9:
            single_image_label = torch.tensor(
                [0, 1, 1, 0, 0, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 10:
            single_image_label = torch.tensor(
                [0, 1, 0, 1, 0, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 11:
            single_image_label = torch.tensor(
                [0, 1, 0, 0, 1, 0, 1, 0], dtype=torch.float32)
        elif self.label_arr[index] == 12:
            single_image_label = torch.tensor(
                [1, 0, 1, 0, 0, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 13:
            single_image_label = torch.tensor(
                [1, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 14:
            single_image_label = torch.tensor(
                [1, 0, 0, 0, 1, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 15:
            single_image_label = torch.tensor(
                [0, 1, 1, 0, 0, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 16:
            single_image_label = torch.tensor(
                [0, 1, 0, 1, 0, 0, 0, 1], dtype=torch.float32)
        elif self.label_arr[index] == 17:
            single_image_label = torch.tensor(
                [0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.float32)
        return single_image_label
