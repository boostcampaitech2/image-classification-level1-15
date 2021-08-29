from numpy.core.numeric import NaN
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from torchvision import datasets, transforms
import albumentations.pytorch


def change_str_to_list(str_list):
    str_list=str_list.replace('\'','').replace('[','').replace(']','').replace(',','')
    str_list=str_list.split()
    # print(str_list)
    # print(type(str_list))
    int_num_list = []
    for str_num in str_list:
        int_num_list.append(int(str_num))

    return int_num_list

# https://github.com/utkuozbulak/pytorch-custom-dataset-examples#incorporating-pandas


class CustomDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transform, train=True):
        self.train = train

        self.data_dir = data_dir + 'train/images/' if self.train else data_dir + 'eval/images/'
        self.csv_path = csv_path if self.train else data_dir + 'eval/info.csv'
        # transforms
        self.transform = transform
        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path)

        if self.train:  # Train Dataset
            # Image paths
            self.image_arr = np.asarray(
                self.data_dir + self.data_info['folder'] + '/' + self.data_info['path'])

            # Labels
            self.label_arr = np.asarray(self.data_info['label'])
            self.gender_label_arr = np.asarray(
                self.data_info['gender_label'])  # gender_label
            self.age_label_arr = np.asarray(
                self.data_info['age_label'])  # age_label
            self.mask_label_arr = np.asarray(
                self.data_info['mask_label'])  # mask_label

            
        else:
            self.image_arr = np.asarray(
                self.data_dir + self.data_info['ImageID'])

        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        if self.train:
            face_path = './data/input/data/train/train_face.csv' if self.train else './data/input/data/eval/eval_face.csv'
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = np.array(Image.open(single_image_name))
            # Bring face coordinate
            mask_num = single_image_name.split('.')[0].split('/')[-1]
            image_path = single_image_name.split('.')[0].split('/')[-2]
            face_info = pd.read_csv(face_path,index_col=0)
            cut_coord=None
            try:
                cut_coord = change_str_to_list(face_info[mask_num][image_path])
            except:
                cut_coord = None

            # Cut face
            if cut_coord:
                ymin = cut_coord[1]
                ymax = cut_coord[1] + cut_coord[3]
                xmin = cut_coord[0]
                xmax = cut_coord[0] + cut_coord[2]
                ymin = ymin - 40 if ymin - 40 > 0 else 0  # include hair
                xmin = xmin - 30 if xmin - 30 > 0 else 0
                ymax = ymax + 30 if ymax + 30 < img_as_img.shape[0] else img_as_img.shape[0] - 1
                xmax = xmax + 30 if xmax + 30 < img_as_img.shape[1] else img_as_img.shape[1] - 1
                img_as_img = img_as_img[ymin:ymax, xmin:xmax,:]

            # Center crop -> 330x330
            else : 
                img_as_img = img_as_img[91:421,27:357,:]
                
            # Transform image to tensor
            if self.transform is not None:
                transform_image = self.transform(image=img_as_img)
            # Get label(class) of the image based on the cropped pandas column
            single_image_label = self.label_arr[index]
            single_gender_label = self.gender_label_arr[index]
            single_age_label = self.age_label_arr[index]
            single_mask_label = self.mask_label_arr[index]
            return (transform_image['image'], single_image_label,
                    single_gender_label, single_age_label, single_mask_label)
        else:
            face_path = './data/input/data/train/train_face.csv' if self.train else './data/input/data/eval/eval_face.csv'
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = np.array(Image.open(single_image_name))

            # Bring face coordinate
            image_path = single_image_name.split('.')[0].split('/')[-1]
            face_info = pd.read_csv(face_path,index_col=0)
            cut_coord=None
            try:
                cut_coord = change_str_to_list(face_info['face'][image_path])
            except:
                cut_coord = None
            # Cut face
            if cut_coord:
                ymin = cut_coord[1]
                ymax = cut_coord[1] + cut_coord[3]
                xmin = cut_coord[0]
                xmax = cut_coord[0] + cut_coord[2]
                ymin = ymin - 40 if ymin - 40 > 0 else 0  # include hair
                xmin = xmin - 30 if xmin - 30 > 0 else 0
                ymax = ymax + 30 if ymax + 30 < img_as_img.shape[0] else img_as_img.shape[0] - 1
                xmax = xmax + 30 if xmax + 30 < img_as_img.shape[1] else img_as_img.shape[1] - 1
                img_as_img = img_as_img[ymin:ymax, xmin:xmax,:]
            # Center crop
            else : 
                img_as_img = img_as_img[106:406,42:342,:]
                
            # Transform image to tensor
            if self.transform is not None:
                transform_image = self.transform(image=img_as_img)
            return transform_image['image']

    def __len__(self):
        return self.data_len


class CustomValidDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transform):
        self.data_dir = data_dir + 'train/images/'
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