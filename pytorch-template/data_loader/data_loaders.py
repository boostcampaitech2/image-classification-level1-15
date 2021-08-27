from torchvision import datasets, transforms
from base import BaseDataLoader
from custom_dataset import *
import albumentations.pytorch


class MaskImageDataLoader(BaseDataLoader):
<<<<<<< HEAD
    def __init__(self, resize, data_dir, csv_path, batch_size, shuffle=True, validation_split=0.2, num_workers=2, training=True):
        self.resize = resize
=======
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True, validation_split=0.2, num_workers=2, training=True):
>>>>>>> 8e4be59d69bce1c679f013825971d55568d5218c
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
<<<<<<< HEAD
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

#
# Todo Base Data Loader 에서 training parameter에 의해 validset에 augementation 되는지 확인하기

#self, resize, data_dir, csv_path, transform


class AgeSmoothingDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True, validation_split=0, num_workers=2, training=False):
        self.transform = albumentations.Compose([
            # albumentations.ColorJitter(brightness=(0.2, 2), contrast=(
            #     0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3)),
            # albumentations.RandomCrop(300, 300),
            # albumentations.HorizontalFlip(),
            # albumentations.augmentations.transforms.ToGray(),
            albumentations.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            albumentations.Resize(224, 224),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path

        self.dataset = AgeLabel50To60Smoothing(
            self.data_dir, self.csv_path, self.transform, train=training)
=======
>>>>>>> 8e4be59d69bce1c679f013825971d55568d5218c
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
