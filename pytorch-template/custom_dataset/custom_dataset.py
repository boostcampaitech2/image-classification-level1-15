import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os


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

        # Write CSV with Labeling
        # 초기 한 번만 실행
        # labeling = Labeling()
        # labeling.write_train_csv_with_labels()

        # Transforms
        self.transforms = transforms
        # Read the csv file
        self.data_info = pd.read_csv(csv_path)

        # Image paths
        # 하드코딩 극혐 죄송합니다
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


"""
       id  gender   race age                path  label
0  000001  female  Asian  45           mask3.jpg      4
1  000001  female  Asian  45           mask2.jpg      4
2  000001  female  Asian  45           mask5.jpg      4
3  000001  female  Asian  45           mask4.jpg      4
4  000001  female  Asian  45  incorrect_mask.jpg     10
5  000001  female  Asian  45          normal.jpg     16
6  000001  female  Asian  45           mask1.jpg      4
"""


class Labeling():
    def __init__(self):
        self.train_csv_path = 'data/input/data/train'
        self.train_csv = self.read_train_csv()
        self.columns = self.get_columns(self.train_csv)
        self.image_directory_names = self.parse_train_image_directory_names()
        self.train_csv_with_labels = self.make_train_csv_with_labels(
            self.image_directory_names)

    def read_train_csv(self):
        return pd.read_csv(os.path.join(self.train_csv_path, 'train.csv'))

    def get_columns(self, train_csv):
        return train_csv.columns.values

    def parse_train_image_directory_names(self):
        return self.train_csv['path'].values

    def make_train_csv_with_labels(self, image_directory_names):
        rows = []

        for image_directory_name in image_directory_names:
            row = self.make_train_csv_row(image_directory_name)
            rows.extend(row)

        train_csv_with_labels = pd.DataFrame(rows)
        train_csv_with_labels.columns = [
            'id', 'gender', 'race', 'age', 'path', 'label']
        return train_csv_with_labels

    def make_train_csv_row(self, image_directory_name):
        image_path = os.path.join(
            'data/input/data/train/images/' + image_directory_name)
        files = os.listdir(image_path)
        row = [image_directory_name.split('_') for _ in range(len(files))]

        for index in range(len(files)):
            file_name = files[index].split('.')[0]
            row[index].append(files[index])
            label = self.make_label(row[index], file_name)
            row[index].append(label)

        return row

    def make_label(self, row, file_name):
        _, gender, _, age, _ = row
        age = int(age)
        label = 0

        if file_name.startswith('mask'):
            if gender == 'male':
                if age < 30:
                    label = 0
                elif 30 <= age < 60:
                    label = 1
                elif age >= 60:
                    label = 2
            elif gender == 'female':
                if age < 30:
                    label = 3
                elif 30 <= age < 60:
                    label = 4
                elif age >= 60:
                    label = 5
        elif file_name.startswith('incorrect'):
            if gender == 'male':
                if age < 30:
                    label = 6
                elif 30 <= age < 60:
                    label = 7
                elif age >= 60:
                    label = 8
            elif gender == 'female':
                if age < 30:
                    label = 9
                elif 30 <= age < 60:
                    label = 10
                elif age >= 60:
                    label = 11
        elif file_name.startswith('normal'):
            if gender == 'male':
                if age < 30:
                    label = 12
                elif 30 <= age < 60:
                    label = 13
                elif age >= 60:
                    label = 14
            elif gender == 'female':
                if age < 30:
                    label = 15
                elif 30 <= age < 60:
                    label = 16
                elif age >= 60:
                    label = 17

        return label

    def get_train_csv_with_labels(self):
        return self.train_csv_with_labels

    def write_train_csv_with_labels(self):
        train_csv_with_labels = self.get_train_csv_with_labels()
        train_csv_with_labels.to_csv(
            'data/train_csv_with_labels.csv', index=False)
