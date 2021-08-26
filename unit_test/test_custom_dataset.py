import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from torchvision import transforms

# https://github.com/utkuozbulak/pytorch-custom-dataset-examples#incorporating-pandas


class CustomDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transforms):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.data_dir = data_dir
        self.csv_path = csv_path

        # Transforms
        self.transforms = transforms
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)
        # Image paths
        self.image_arr = np.asarray(
            self.data_dir + self.data_info['id'] + '_' +
            self.data_info['gender'] + '_' +
            self.data_info['race'] + '_' +
            self.data_info['age'].astype(str) + '/' +
            self.data_info['path'])
        # Labels
        self.label_arr = np.asarray(self.data_info['label'])
        # Calculate len
        self.data_len = len(self.data_info.index)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform image to tensor
        transform_image = self.transforms(img_as_img)
        # Get label(class) of the image based on the cropped pandas column
        single_image_label = self.label_arr[index]
        return (transform_image, single_image_label)

    def __len__(self):
        return self.data_len


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

c = CustomDatasetFromImages(
    "input/data/train/images/", "train_csv_with_labels.csv", transform)
print(c[0])
