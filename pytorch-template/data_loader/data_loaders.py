from torchvision import datasets, transforms
from base import BaseDataLoader
from custom_dataset import *
import albumentations.pytorch


class MaskImageDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True, validation_split=0.2, num_workers=2, training=True):
        self.transform = albumentations.Compose([
            albumentations.ColorJitter(brightness=(0.2, 2), contrast=(
                0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3)),
            albumentations.RandomCrop(300, 300),
            albumentations.HorizontalFlip(),
            # albumentations.augmentations.transforms.ToGray(),
            albumentations.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            albumentations.Resize(self.resize, self.resize),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path

        # d_type, resize, data_dir, csv_path, transforms, train
        self.dataset = CustomDatasetFromImages(
            self.data_dir, self.csv_path, self.transform, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class TestDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
