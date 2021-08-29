from cv2 import transform
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import os
from torchvision import datasets, transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch
# import augmentations


# https://github.com/utkuozbulak/pytorch-custom-dataset-examples#incorporating-pandas
class CustomDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transform, train=True):
        self.train = train

        self.data_dir = data_dir + 'train/images/' if self.train else data_dir + 'eval/images/'
        self.csv_path = csv_path if self.train else data_dir + 'eval/info.csv'

        # Transforms
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

        self.augmentations = [
            self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
            self.translate_x, self.translate_y
        ]

        self.augmentations_all = [
            self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
            self.translate_x, self.translate_y, self.color, self.contrast, self.brightness, self.sharpness
        ]

        self.IMAGE_SIZE = 224
        self.mixture_width = 3
        self.mixture_depth = -1
        self.aug_severity = 3
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.preprocess = A.Compose([
            A.RandomResizedCrop(224, 224),
            A.HorizontalFlip(),
            A.Normalize(mean, std),
            ToTensorV2(),
        ])

    def __getitem__(self, index):
        if self.train:
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = Image.open(single_image_name)
            img_as_np = np.array(img_as_img)
            # Transform image to tensor
            if self.transform is not None:
                # transform_image = self.transform(image=img_as_np)
                im_tuple = (self.preprocess(image=img_as_np)['image'], self.aug(
                    img_as_img, self.preprocess), self.aug(img_as_img, self.preprocess, all_ops=True))

            # Get label(class) of the image based on the cropped pandas column
            single_image_label = self.label_arr[index]
            single_gender_label = self.gender_label_arr[index]
            single_age_label = self.age_label_arr[index]
            single_mask_label = self.mask_label_arr[index]
            return (im_tuple, single_image_label,
                    single_gender_label, single_age_label, single_mask_label)
        else:
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = Image.open(single_image_name)
            img_as_np = np.array(img_as_img)
            # Transform image to tensor
            if self.transform is not None:
                im_tuple = (self.preprocess(image=img_as_np)['image'], self.aug(
                    img_as_img, self.preprocess), self.aug(img_as_img, self.preprocess, all_ops=True))
            return im_tuple

    def __len__(self):
        return self.data_len

    def aug(self, image, preprocess, all_ops=False):
        """Perform AugMix augmentations and compute mixture.
        Args:
        image: PIL.Image input image
        preprocess: Preprocessing function which should return a torch tensor.
        Returns:
        mixed: Augmented and mixed image.
        """
        aug_list = self.augmentations
        if all_ops:
            aug_list = self.augmentations_all

        ws = np.float32(np.random.dirichlet([1] * self.mixture_width))
        m = np.float32(np.random.beta(1, 1))

        mix = torch.zeros_like(preprocess(image=np.array(image))['image'])
        for i in range(self.mixture_width):
            image_aug = image.copy()
            depth = self.mixture_depth if self.mixture_depth > 0 else np.random.randint(
                1, 4)
            for _ in range(depth):
                op = np.random.choice(aug_list)
                image_aug = op(image_aug, self.aug_severity)
        # Preprocessing commutes since all coefficients are convex
        mix += ws[i] * preprocess(image=np.array(image_aug))['image']

        mixed = (1 - m) * preprocess(image=np.array(image))['image'] + m * mix
        return mixed

    def int_parameter(self, level, maxval):
        """Helper function to scale `val` between 0 and maxval .
        Args:
            level: Level of the operation that will be between [0, `PARAMETER_MAX`].
            maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.
        Returns:
            An int that results from scaling `maxval` according to `level`.
        """
        return int(level * maxval / 10)

    def float_parameter(self, level, maxval):
        """Helper function to scale `val` between 0 and maxval.
        Args:
            level: Level of the operation that will be between [0, `PARAMETER_MAX`].
            maxval: Maximum value that the operation can have. This will be scaled to
            level/PARAMETER_MAX.
        Returns:
            A float that results from scaling `maxval` according to `level`.
        """
        return float(level) * maxval / 10.

    def sample_level(self, n):
        return np.random.uniform(low=0.1, high=n)

    def autocontrast(self, pil_img, _):
        return ImageOps.autocontrast(pil_img)

    def equalize(self, pil_img, _):
        return ImageOps.equalize(pil_img)

    def posterize(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), 4)
        return ImageOps.posterize(pil_img, 4 - level)

    def rotate(self, pil_img, level):
        degrees = self.int_parameter(self.sample_level(level), 30)
        if np.random.uniform() > 0.5:
            degrees = -degrees
        return pil_img.rotate(degrees, resample=Image.BILINEAR)

    def solarize(self, pil_img, level):
        level = self.int_parameter(self.sample_level(level), 256)
        return ImageOps.solarize(pil_img, 256 - level)

    def shear_x(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                 Image.AFFINE, (1, level, 0, 0, 1, 0),
                                 resample=Image.BILINEAR)

    def shear_y(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 0.3)
        if np.random.uniform() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                 Image.AFFINE, (1, 0, 0, level, 1, 0),
                                 resample=Image.BILINEAR)

    def translate_x(self, pil_img, level):
        level = self.int_parameter(
            self.sample_level(level), self.IMAGE_SIZE / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                 Image.AFFINE, (1, 0, level, 0, 1, 0),
                                 resample=Image.BILINEAR)

    def translate_y(self, pil_img, level):
        level = self.int_parameter(
            self.sample_level(level), self.IMAGE_SIZE / 3)
        if np.random.random() > 0.5:
            level = -level
        return pil_img.transform((self.IMAGE_SIZE, self.IMAGE_SIZE),
                                 Image.AFFINE, (1, 0, 0, 0, 1, level),
                                 resample=Image.BILINEAR)

    # operation that overlaps with ImageNet-C's test set
    def color(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 1.8) + 0.1
        return ImageEnhance.Color(pil_img).enhance(level)

    # operation that overlaps with ImageNet-C's test set
    def contrast(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 1.8) + 0.1
        return ImageEnhance.Contrast(pil_img).enhance(level)

    # operation that overlaps with ImageNet-C's test set
    def brightness(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 1.8) + 0.1
        return ImageEnhance.Brightness(pil_img).enhance(level)

    # operation that overlaps with ImageNet-C's test set
    def sharpness(self, pil_img, level):
        level = self.float_parameter(self.sample_level(level), 1.8) + 0.1
        return ImageEnhance.Sharpness(pil_img).enhance(level)


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
