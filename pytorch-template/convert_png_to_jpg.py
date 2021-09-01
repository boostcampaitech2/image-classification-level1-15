import os
from PIL import Image

root_dir = '/opt/ml/level1-15/pytorch-template/data/input/data/train/train_horizon'

# 006224_male_Asian_20 : png

not_jpg_file = []
for (root, dirs, files) in os.walk(root_dir):
    if len(files) > 0:
        for file_name in files:
            t = file_name.split('.')[-1]
            if len(t) > 4:
                continue

            if t != 'jpg' and file_name[0] != '.':
                not_jpg_file.append(f'{root}/{file_name}')

for imagedir in not_jpg_file:
    im = Image.open(imagedir)
    im.save(imagedir.split('.')[0] + '.jpg')
