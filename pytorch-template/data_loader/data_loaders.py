from torchvision import datasets, transforms
from torchvision.transforms.transforms import Normalize, Resize
from base import BaseDataLoader
from custom_dataset import *
from albumentations.pytorch import ToTensorV2
import albumentations as A


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


class MaskImageDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=True):
        # self.transform = A.Compose([
        #     A.Resize(224, 224),
        #     A.HorizontalFlip(),
        #     A.Normalize(
        #         mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
        #     ToTensorV2()
        # ])
        self.transform = A.Compose([
            A.OneOf(transform_list, p=0.5),
            A.Resize(240, 240),
            # A.HorizontalFlip(),
            A.Normalize(mean=(0.445, 0.47, 0.52), std=(0.248, 0.24, 0.235)),
            ToTensorV2()
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path

        # d_type, resize, data_dir, csv_path, transforms, train
        self.dataset = CustomDatasetFromImages(
            self.data_dir, self.csv_path, self.transform, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MaskImageValidDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=False):
        self.transform = A.Compose([
            A.Resize(240, 240),
            A.Normalize(mean=(0.445, 0.47, 0.52), std=(0.248, 0.24, 0.235)),
            ToTensorV2()
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.dataset = CustomValidDatasetFromImages(
            self.data_dir, self.csv_path, self.transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
