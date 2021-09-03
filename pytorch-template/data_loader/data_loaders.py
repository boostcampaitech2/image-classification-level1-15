from torchvision import datasets, transforms
from base import BaseDataLoader
from custom_dataset import *
from albumentations.pytorch import ToTensorV2
import albumentations as A


class MaskImageDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True, validation_split=0.0, num_workers=2, training=True):
        self.transform = A.Compose([
            A.ColorJitter(brightness=(0.2, 2), contrast=(
                0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3)),
            A.HorizontalFlip(),
            A.augmentations.transforms.ToGray(),
            A.Resize(224, 224),
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
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
            A.Resize(224, 224),
            A.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            ToTensorV2()
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.dataset = CustomValidDatasetFromImages(
            self.data_dir, self.csv_path, self.transform)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
