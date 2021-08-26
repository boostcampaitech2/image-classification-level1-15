import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import os
from torchvision import datasets, transforms


# https://github.com/utkuozbulak/pytorch-custom-dataset-examples#incorporating-pandas
class CustomDatasetFromImages(Dataset):
    def __init__(self, data_dir, csv_path, transforms, train=True):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.train = train

        self.data_dir = data_dir + 'train/images/' if self.train else data_dir + 'eval/images/'
        self.csv_path = csv_path if self.train else data_dir + 'eval/info.csv'

        # Write CSV with Labeling
        # 초기 한 번만 실행
        # labeling = Labeling()
        # labeling.write_train_csv_with_multi_labels()

        # Transforms
        self.transforms = transforms
        # Read the csv file
        self.data_info = pd.read_csv(self.csv_path)

        if self.train:  # Train Dataset
            # Image paths
            self.image_arr = np.asarray(
                self.data_dir + self.data_info['id'] + '_' +
                self.data_info['gender'] + '_' +
                self.data_info['race'] + '_' +
                self.data_info['age'].astype(str) + '/' +
                self.data_info['path'])
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
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = np.array(Image.open(single_image_name))
            # Transform image to tensor
            if self.transforms is not None:
                transform_image = self.transforms(image=img_as_img)
            # Get label(class) of the image based on the cropped pandas column
            single_image_label = self.label_arr[index]
            single_gender_label = self.gender_label_arr[index]
            single_age_label = self.age_label_arr[index]
            single_mask_label = self.mask_label_arr[index]
            return (transform_image['image'], single_image_label,
                    single_gender_label, single_age_label, single_mask_label)
        else:
            # Get image name from the pandas df
            single_image_name = self.image_arr[index]
            # Open image
            img_as_img = np.array(Image.open(single_image_name))
            # Transform image to tensor
            if self.transforms is not None:
                transform_image = self.transforms(image=img_as_img)
            return transform_image['image']

    def __len__(self):
        return self.data_len


class CustomDatasetFromImages2(Dataset):
    def __init__(self, d_type, resize, data_dir, csv_path, transform, train):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.d_type = d_type
        self.is_train = train

        if self.is_train:
            self.transforms = transform  # augmuent
        else:
            self.transforms = transforms.Compose([
                # Albumentation 으로 변경
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.2, 0.2, 0.2))
            ])

        self.data_dir = data_dir + 'train/images/'

        self.csv_path = csv_path

        self.data_info = pd.read_csv(self.csv_path)
        self.image_arr = np.asarray(
            self.data_dir + self.data_info['id'] + '_' +
            self.data_info['gender'] + '_' +
            self.data_info['race'] + '_' +
            self.data_info['age'].astype(str) + '/' +
            self.data_info['path'])
        # Labels

        self.label = None

        self.label_arr = np.asarray(self.data_info['label'])

        if self.d_type == 'gender':
            self.label = np.asarray(self.data_info['gender_label'])
        elif self.d_type == 'age':
            self.label = np.asarray(self.data_info['age_label'])
        elif self.d_type == 'mask':
            self.label = np.asarray(self.data_info['mask_label'])

    def __getitem__(self, index):
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        transform_image = self.transforms(img_as_img)
        # print(self.transforms)
        target = self.label[index]
        return transform_image, target

    def __len__(self):
        return len(self.data_info)


class AgeLabel50To60Smoothing(Dataset):
    def __init__(self, resize, data_dir, csv_path, transform, train):

        self.is_train = train

        if self.is_train:
            self.transforms = transform  # augmuent
        else:
            self.transforms = transforms.Compose([
                # Albumentation 으로 변경
                transforms.Resize((resize, resize)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),
                                     std=(0.2, 0.2, 0.2))
            ])

        self.data_dir = data_dir + 'train/images/'

        self.csv_path = csv_path

        self.data_info = pd.read_csv(self.csv_path)
        self.image_arr = np.asarray(
            self.data_dir + self.data_info['id'] + '_' +
            self.data_info['gender'] + '_' +
            self.data_info['race'] + '_' +
            self.data_info['age'].astype(str) + '/' +
            self.data_info['path'])
        # Labels

        self.label = None
        self.age_info = self.data_info['age']

        self.label_arr = np.asarray(self.data_info['label'])
        self.label = np.asarray(self.data_info['age_label'])

    def __getitem__(self, index):
        target = None
        if self.age_info[index] <= 50:
            target = self.label[index]
        elif self.age_info[index] >= 60:
            target = self.label[index]
        else:
            # 51~ 59
            floating_age = (self.age_info[index] - 50) / 10
            target = self.age_info + floating_age

        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        transform_image = self.transforms(img_as_img)
        # print(self.transforms)
        target = self.label[index]
        return transform_image, target

    def __len__(self):
        return len(self.data_info)


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

# """
#        id  gender   race age                path  label  gender_label  age_label  mask_label
# 0  000001  female  Asian  45           mask3.jpg      4             1          1           0
# 1  000001  female  Asian  45           mask2.jpg      4             1          1           0
# 2  000001  female  Asian  45           mask5.jpg      4             1          1           0
# 3  000001  female  Asian  45           mask4.jpg      4             1          1           0
# 4  000001  female  Asian  45  incorrect_mask.jpg     10             1          1           1
# 5  000001  female  Asian  45          normal.jpg     16             1          1           2
# 6  000001  female  Asian  45           mask1.jpg      4             1          1           0
# """


class Labeling():
    def __init__(self):
        self.train_csv_path = 'data/input/data/train'
        self.train_csv = self.read_train_csv()
        self.columns = self.get_columns(self.train_csv)
        self.image_directory_names = self.parse_train_image_directory_names()
        self.train_csv_with_labels = self.make_train_csv_with_labels(
            self.image_directory_names)
        self.train_csv_with_multi_labels = self.make_multi_labels(
            self.train_csv_with_labels)

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

    def categorize_age(self, n):
        if int(n) < 30:
            return 0
        elif 30 <= int(n) < 60:
            return 1
        elif int(n) >= 60:
            return 2

    def categorize_mask(self, s):
        if s.startswith('mask'):
            return 0
        elif s.startswith('incorrect'):
            return 1
        elif s.startswith('normal'):
            return 2
        else:
            print(s)

    def make_multi_labels(self, train_csv_with_labels):
        train_csv_with_multi_labels = train_csv_with_labels

        train_csv_with_multi_labels['gender_label'] = (
            train_csv_with_labels['gender'] == 'female').astype(int)

        train_csv_with_multi_labels['age_label'] = train_csv_with_labels['age'].apply(
            self.categorize_age)

        train_csv_with_multi_labels['mask_label'] = train_csv_with_labels['path'].apply(
            self.categorize_mask)

        print(train_csv_with_multi_labels.head(7))
        return train_csv_with_multi_labels

    def get_train_csv_with_multi_labels(self):
        return self.train_csv_with_multi_labels

    def write_train_csv_with_multi_labels(self):
        train_csv_with_multi_labels = self.get_train_csv_with_multi_labels()
        train_csv_with_multi_labels.to_csv(
            'data/train_csv_with_multi_labels.csv', index=False)
