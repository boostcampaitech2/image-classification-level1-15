import os
import pandas as pd

# data_dir = "/opt/ml/level1-15/pytorch-template/data/input/data/train/AgeRecognitionDataset"
data_dir = "/opt/ml/level1-15/pytorch-template/data/input/data/train/resize"
d = pd.DataFrame()

paths = []
genders = []
ages = []
for image in os.listdir(data_dir):
    image_id, extension = image.split('.')
    _, _, _, gender, age = image_id.split('_')
    # if gender == 'FEMAIL':
    paths.append(os.path.join(data_dir, image))
    # genders.append(1)
    if int(age) < 30:
        ages.append(0)
    elif 30 <= int(age) < 60:
        ages.append(1)
    else:
        ages.append(2)
    # if gender == 'MALE':
    # paths.append(os.path.join(data_dir, image))
    # genders.append(0)
    # if int(age) < 30:
    #     ages.append(0)
    # elif 30 <= int(age) < 60:
    #     ages.append(1)
    # else:
    #     ages.append(2)

# labels = []
# for folder in os.listdir(data_dir):
#     if folder == '25-30' or folder == '6-20':
#         for p in os.listdir(os.path.join(data_dir, folder)):
#             paths.append(os.path.join(data_dir + '/' + folder, p))
#             labels.append(0)
#     elif folder == '42-48':
#         for p in os.listdir(os.path.join(data_dir, folder)):
#             paths.append(os.path.join(data_dir + '/' + folder, p))
#             labels.append(1)
#     elif folder == '60-98':
#         for p in os.listdir(os.path.join(data_dir, folder)):
#             paths.append(os.path.join(data_dir + '/' + folder, p))
#             labels.append(2)

columns = ['path', 'gender', 'age']

d['path'] = paths
# d['gender_label'] = genders
d['age_label'] = ages

# print(d['age_label'].value_counts())
d.to_csv('data/add_age_data_2.csv', index=False)


# d['age_label']
