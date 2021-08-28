import pandas as pd
import numpy as np
import random
from collections import defaultdict
from itertools import chain
import os


class MakeTrainAndValidCsvWithLabeling():
    def __init__(self, split_ratio=0.2):
        self.split_ratio = split_ratio
        self.train_csv_path = 'data/input/data/train'
        self.train_csv = self.read_train_csv()
        self.columns = self.get_columns(self.train_csv)
        self.image_directory_names = self.parse_train_image_directory_names()
        self.train_csv_with_labels = self.make_train_csv_with_labels(
            self.image_directory_names)
        self.train_csv_with_multi_labels = self.make_multi_labels(
            self.train_csv_with_labels)
        self.label_count = self.count_label()
        self.valid_csv_with_multi_labels = self.make_valid_csv_with_multi_labels()
        self.splitted_train_csv_with_multi_labels = self.make_splitted_train_csv_with_multi_labels()

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
            assert "Dataset have incorrect images"

    def make_multi_labels(self, train_csv_with_labels):
        train_csv_with_multi_labels = train_csv_with_labels

        train_csv_with_multi_labels['gender_label'] = (
            train_csv_with_labels['gender'] == 'female').astype(int)

        train_csv_with_multi_labels['age_label'] = train_csv_with_labels['age'].apply(
            self.categorize_age)

        train_csv_with_multi_labels['mask_label'] = train_csv_with_labels['path'].apply(
            self.categorize_mask)

        print(train_csv_with_multi_labels)
        return train_csv_with_multi_labels

    def get_train_csv_with_multi_labels(self):
        return self.train_csv_with_multi_labels

    def write_train_csv_with_multi_labels(self):
        train_csv_with_multi_labels = self.get_train_csv_with_multi_labels()
        train_csv_with_multi_labels.to_csv(
            'data/train_csv_with_multi_labels.csv', index=False)

    def count_label(self):
        label_count = self.get_train_csv_with_multi_labels()['label']
        label_count = label_count.value_counts().sort_values().drop_duplicates()
        return label_count

    def pick_normal_and_incorrect_mask_index(self):
        infrequent_classes = self.label_count[:6].index
        return infrequent_classes

    def stratification(self, train_csv_with_multi_labels, infrequent_classes):
        valid_csv_length = int(
            len(train_csv_with_multi_labels) * self.split_ratio / 7)
        valid_image_dir = []
        count_summation = 0

        for class_num in infrequent_classes:
            if class_num == infrequent_classes[-1]:
                group_count = valid_csv_length - count_summation
            else:
                group_count = round(
                    self.label_count[class_num] * self.split_ratio)

            random.seed(42)
            group_train_csv_with_multi_labels = train_csv_with_multi_labels[
                train_csv_with_multi_labels['label'] == class_num]
            index = random.sample(
                list(group_train_csv_with_multi_labels.index), group_count)
            group_folder_path = train_csv_with_multi_labels.iloc[index]['id'].values + '_' +\
                train_csv_with_multi_labels.iloc[index]['gender'].values + '_' +\
                train_csv_with_multi_labels.iloc[index]['race'].values + '_' +\
                train_csv_with_multi_labels.iloc[index]['age'].values

            valid_image_dir.append(group_folder_path)
            count_summation += group_count

        return valid_image_dir

    def make_valid_csv_with_multi_labels(self):
        valid_folder_list = self.stratification(
            self.train_csv_with_multi_labels, self.pick_normal_and_incorrect_mask_index())
        valid_folder_list = list(chain(*valid_folder_list))

        self.train_csv_with_multi_labels['folder'] = self.train_csv_with_multi_labels['id'] + '_' +\
            self.train_csv_with_multi_labels['gender'] + '_' +\
            self.train_csv_with_multi_labels['race'] + '_' +\
            self.train_csv_with_multi_labels['age']

        valid_csv_with_multi_labels = self.train_csv_with_multi_labels[
            self.train_csv_with_multi_labels['folder'].isin(valid_folder_list)]
        return valid_csv_with_multi_labels

    def make_splitted_train_csv_with_multi_labels(self):
        valid_index = self.valid_csv_with_multi_labels.index
        train_index = [
            i for i in self.train_csv_with_multi_labels.index if i not in valid_index]
        splitted_csv_with_multi_labels = self.train_csv_with_multi_labels.iloc[train_index]
        return splitted_csv_with_multi_labels

    def get_valid_csv_with_multi_labels(self):
        return self.valid_csv_with_multi_labels

    def get_splitted_train_csv_with_multi_labels(self):
        return self.splitted_train_csv_with_multi_labels

    def write_valid_csv_with_multi_labels(self):
        valid_csv_with_multi_labels = self.get_valid_csv_with_multi_labels()
        valid_csv_with_multi_labels.to_csv(
            'data/splitted_valid_csv_10.csv', index=False)

    def write_splitted_train_csv_with_multi_labels(self):
        splitted_train_csv_with_multi_labels = self.get_splitted_train_csv_with_multi_labels()
        splitted_train_csv_with_multi_labels.to_csv(
            'data/splitted_train_csv_10.csv', index=False)


splitter = MakeTrainAndValidCsvWithLabeling(0.10)
splitter.write_valid_csv_with_multi_labels()
splitter.write_splitted_train_csv_with_multi_labels()
