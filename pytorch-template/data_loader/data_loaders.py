from torchvision import datasets, transforms
from base import BaseDataLoader
from custom_dataset import *


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """

    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(
            self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)


class MaskImageDataLoader(BaseDataLoader):
    def __init__(self, data_dir, d_type, csv_path, resize, batch_size, shuffle=True, validation_split=0.2, num_workers=4, training=True):
        self.resize = resize
        self.d_type = d_type
        self.transform = transforms.Compose([
            # Albumentation 으로 변경
            transforms.RandomCrop(300),
            transforms.Resize((resize, resize)),
            transforms.ColorJitter(brightness=(0.2, 2), contrast=(
                0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2))
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path

        # d_type, resize, data_dir, csv_path, transforms, train
        self.dataset = CustomDatasetFromImages2(
            self.d_type, self.resize, self.data_dir, self.csv_path, self.transform, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
