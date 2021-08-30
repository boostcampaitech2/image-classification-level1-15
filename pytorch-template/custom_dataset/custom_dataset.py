import pandas as pd
from torch.utils import data
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from torchvision import datasets, transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from imblearn.over_sampling import ADASYN

# https://github.com/utkuozbulak/pytorch-custom-dataset-examples#incorporating-pandas


class CustomDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transform, train=True):
        self.train = train

        self.data_dir = data_dir + \
            'train/images/' if self.train else data_dir + 'eval/images/'
        # 'train/images/' if self.train else data_dir + 'eval/crop_images/'
        self.csv_path = csv_path if self.train else data_dir + 'eval/info.csv'

        # Transforms
        self.transform = transform
        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path)

        if self.train:  # Train Dataset
            self.crop_dataset_start_index = len(self.data_info)
            self.crop_dir = data_dir + 'train/crop_images/'
            self.crop_data_info = pd.read_csv(self.csv_path)
            # self.data_info = pd.concat(
            #     [self.data_info, self.data_info], ignore_index=True)
            self.image_arr = np.append(np.asarray(
                self.data_dir + self.data_info['folder'] + '/' + self.data_info['path']),
                np.asarray(self.crop_dir + self.crop_data_info['folder'] + '/' + self.crop_data_info['path']))
            self.label_arr = np.append(np.asarray(
                self.data_info['label']), np.asarray(self.crop_data_info['label']))
            self.gender_label_arr = np.append(np.asarray(
                self.data_info['gender_label']), np.asarray(self.crop_data_info['gender_label']))  # gender_label
            self.age_label_arr = np.append(np.asarray(
                self.data_info['age_label']), np.asarray(self.crop_data_info['age_label']))  # age_label
            self.mask_label_arr = np.append(np.asarray(
                self.data_info['mask_label']), np.asarray(self.crop_data_info['mask_label']))  # mask_label

        else:
            eval_crop_data_info = pd.read_csv(self.csv_path)
            self.eval_crop_image_arr = np.asarray(
                data_dir + 'eval/crop_images/' + eval_crop_data_info['ImageID'])

            self.image_arr = np.asarray(
                self.data_dir + self.data_info['ImageID'])

        if train:
            # Calculate len
            self.data_len = len(self.data_info.index) + \
                len(self.crop_data_info.index)
        else:
            self.data_len = len(self.data_info.index)

        self.base_transform = A.Compose([
            A.Resize(300, 300),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ToTensorV2()
        ])
        self.crop_transform = A.Compose([
            A.Resize(300, 300),
            A.HorizontalFlip(),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ToTensorV2()
        ])

    def __getitem__(self, index):
        if self.train:
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = np.array(Image.open(single_image_name))
            
            # img_tensor = transforms.ToTensor()(Image.open(single_image_name))
            # print(img_tensor.shape)
            # Transform image to tensor
            # if self.transform is not None:
            if index < self.crop_dataset_start_index:
                transform_image = self.transform(image=img_as_img)
            else:
                transform_image = self.crop_transform(image=img_as_img)
            # Get label(class) of the image based on the cropped pandas column
            single_image_label = self.label_arr[index]
            single_gender_label = self.gender_label_arr[index]
            single_age_label = self.age_label_arr[index]
            single_mask_label = self.mask_label_arr[index]

            return (np.asarray(transform_image['image']), single_image_label,
                    single_gender_label, single_age_label, single_mask_label)
        else:
            # # Get image name from the pandas df
            # single_image_name = self.image_arr[index]
            # # Open image
            # img_as_img = np.array(Image.open(single_image_name))
            # # Transform image to tensor
            # # if self.transform is not None:
            # # transform_image = self.transform(image=img_as_img)
            # transform_image = self.base_transform(image=img_as_img)
            # return transform_image['image']

            ## NOTE: TTA
            single_image_name = self.image_arr[index]
            single_crop_image_name = self.eval_crop_image_arr[index]

            img_as_img = np.array(Image.open(single_image_name))
            crop_img_as_img = np.array(Image.open(single_crop_image_name))

            transform_image1 = self.crop_transform(image=img_as_img)['image']
            transform_image2 = self.transform(image=img_as_img)['image']
            transform_image3 = self.base_transform(
                image=crop_img_as_img)['image']

            images = torch.cat((transform_image1, transform_image2), dim=0)
            images = torch.cat((images, transform_image3), dim=0)
            return images

    def __len__(self):
        return self.data_len


class CustomValidDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transform):
        self.data_dir = data_dir + 'train/crop_images/'
        self.csv_path = csv_path

        self.transform = transform
        self.data_info = pd.read_csv(self.csv_path)

        self.image_arr = np.asarray(
            self.data_dir + self.data_info['folder'] + '/' + self.data_info['path'])

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

        single_image_label = self.label_arr[index]
        single_gender_label = self.gender_label_arr[index]
        single_age_label = self.age_label_arr[index]
        single_mask_label = self.mask_label_arr[index]
        return (transform_image['image'], single_image_label,
                single_gender_label, single_age_label, single_mask_label)

    def __len__(self):
        return self.data_len

class DataAdasyn(Dataset):
    def __init__(self, dataset):
        self.image_list = dataset[:,0]
        self.image_list = np.array([img.tolist() for img in self.image_list])
        self.ylabel = dataset[:,1].astype('int')  # image_label
        
        self._oversample()
        self.data_len = len(self.ylabel)
    
    def _oversample(self):
        oversampler = ADASYN(random_state=123)
        self.image_list, self.ylabel = oversampler.fit_resample(self.image_list.reshape(self.image_list.shape[0], -1), self.ylabel)
        self.image_list = self.image_list.reshape(self.image_list.shape[0],300,300,3)

    def __getitem__(self, index):
        single_image_name = self.image_list[index]
        single_image_label = self.ylabel[index]

        if single_image_label // 6 == 0:  # Wear
            single_mask_label = 0
            if single_image_label // 3 ==0 : # Male
                single_gender_label = 0
                if single_image_label % 3 == 0: # <30
                    single_age_label = 0
                elif single_image_label % 3 == 1: # 30~60
                    single_age_label = 1
                else: #>60
                    single_age_label = 2
            else: #Female
                single_gender_label = 1
                if single_image_label % 3 == 0: # <30
                    single_age_label = 0
                elif single_image_label % 3 == 1: # 30~60
                    single_age_label = 1
                else: #>60
                    single_age_label = 2

        elif single_image_label // 6 == 1:  # Incorrect
            single_mask_label = 1
            if single_image_label // 3 ==0 : # Male
                single_gender_label = 0
                if single_image_label % 3 == 0: # <30
                    single_age_label = 0
                elif single_image_label % 3 == 1: # 30~60
                    single_age_label = 1
                else: #>60
                    single_age_label = 2
            else: #Female
                single_gender_label = 1
                if single_image_label % 3 == 0: # <30
                    single_age_label = 0
                elif single_image_label % 3 == 1: # 30~60
                    single_age_label = 1
                else: #>60
                    single_age_label = 2

        else: # Not wear
            single_mask_label = 2
            if single_image_label // 3 ==0 : # Male
                single_gender_label = 0
                if single_image_label % 3 == 0: # <30
                    single_age_label = 0
                elif single_image_label % 3 == 1: # 30~60
                    single_age_label = 1
                else: #>60
                    single_age_label = 2
            else: #Female
                single_gender_label = 1
                if single_image_label % 3 == 0: # <30
                    single_age_label = 0
                elif single_image_label % 3 == 1: # 30~60
                    single_age_label = 1
                else: #>60
                    single_age_label = 2

        return (torch.Tensor(single_image_name), single_image_label,
                single_gender_label, single_age_label, single_mask_label)

    def __len__(self):
        return self.data_len

