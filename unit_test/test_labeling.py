import os
from unicodedata import category
import pandas as pd
from torch._C import dtype


"""
       id  gender   race age                path  label  gender_label  age_label  mask_label
0  000001  female  Asian  45           mask3.jpg      4             1          1           0
1  000001  female  Asian  45           mask2.jpg      4             1          1           0
2  000001  female  Asian  45           mask5.jpg      4             1          1           0
3  000001  female  Asian  45           mask4.jpg      4             1          1           0
4  000001  female  Asian  45  incorrect_mask.jpg     10             1          1           1
5  000001  female  Asian  45          normal.jpg     16             1          1           2
6  000001  female  Asian  45           mask1.jpg      4             1          1           0
"""


class Labeling():
    def __init__(self):
        self.train_csv_path = './input/data/train'
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
            self.train_csv_path, 'images/' + image_directory_name)
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
        train_csv_with_labels.to_csv('train_csv_with_labels.csv', index=False)

    """
           id  gender   race age                path  label  gender_label  age_label mask_label
    0  000001  female  Asian  45           mask3.jpg      4
    1  000001  female  Asian  45           mask2.jpg      4
    2  000001  female  Asian  45           mask5.jpg      4
    3  000001  female  Asian  45           mask4.jpg      4
    4  000001  female  Asian  45  incorrect_mask.jpg     10
    5  000001  female  Asian  45          normal.jpg     16
    6  000001  female  Asian  45           mask1.jpg      4
    """

    def age_categorize(self, n):
        if int(n) < 30:
            return 0
        elif 30 <= int(n) < 60:
            return 1
        elif int(n) >= 60:
            return 2

    def mask_categorize(self, s):
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
            self.age_categorize)

        train_csv_with_multi_labels['mask_label'] = train_csv_with_labels['path'].apply(
            self.mask_categorize)

        print(train_csv_with_multi_labels.head(7))
        return train_csv_with_multi_labels

    def get_train_csv_with_multi_labels(self):
        return self.train_csv_with_multi_labels

    def write_train_csv_with_multi_labels(self):
        train_csv_with_multi_labels = self.get_train_csv_with_multi_labels()
        train_csv_with_multi_labels.to_csv(
            'train_csv_with_multi_labels.csv', index=False)


"""
gender_label
  - male = 0
  - female = 1
"""

labeling = Labeling()

labeling.write_train_csv_with_multi_labels()
# a = pd.read_csv('train_csv_with_labels.csv')
# print(a.head(7))
