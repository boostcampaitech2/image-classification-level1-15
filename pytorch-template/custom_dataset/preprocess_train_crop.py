import os
import cv2
from PIL import Image
import pandas as pd
import torch
from torchvision import transforms
import albumentations as A
import albumentations.pytorch
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from facenet_pytorch import MTCNN
from retinaface import RetinaFace

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)

'''
트레인셋 crop
'''

info = pd.read_csv(
    '/opt/ml/image-classification-level1-15/pytorch-template/data/input/data/train/train.csv')
folder_path = '/opt/ml/image-classification-level1-15/pytorch-template/data/input/data/train/images/'
image_folders = os.listdir(folder_path)
for image_folder in image_folders:
    files = [s for s in os.listdir(
        folder_path+image_folder) if not s.startswith('.')]

    for image_folder in image_folders:
        files = [s for s in os.listdir(
            folder_path+image_folder) if not s.startswith('.')]
        for file in files:
            img_path = folder_path + image_folder+'/' + file

            img = Image.open(img_path)
            boxes, probs = mtcnn.detect(img)

            if not isinstance(boxes, np.ndarray):

                retina_dectection = RetinaFace.detect_faces(img_path)
                if type(retina_dectection) == dict:
                    print('retina')
                    xmin = int(
                        retina_dectection["face_1"]["facial_area"][0]) - 30
                    ymin = int(
                        retina_dectection["face_1"]["facial_area"][1]) - 30
                    xmax = int(
                        retina_dectection["face_1"]["facial_area"][2]) + 30
                    ymax = int(
                        retina_dectection["face_1"]["facial_area"][3]) + 30

                    if xmin < 0:
                        xmin = 0
                    if ymin < 0:
                        ymin = 0
                    if xmax > 384:
                        xmax = 384
                    if ymax > 512:
                        ymax = 512
                    img = A.Crop(x_min=xmin, y_min=ymin, x_max=xmax,
                                 y_max=ymax)(image=np.array(img))
                    img = img['image']

                elif type(retina_dectection) == tuple:
                    print('averate')
                    xmin = 80
                    ymin = 50
                    xmax = 80 + 220
                    ymax = 50 + 320

                    img = A.Crop(x_min=xmin, y_min=ymin, x_max=xmax,
                                 y_max=ymax)(image=np.array(img))
                    img = img['image']
            else:
                xmin = int(boxes[0, 0])-30
                ymin = int(boxes[0, 1])-30
                xmax = int(boxes[0, 2])+30
                ymax = int(boxes[0, 3])+30
                if xmin < 0:
                    xmin = 0
                if ymin < 0:
                    ymin = 0
                if xmax > 384:
                    xmax = 384
                if ymax > 512:
                    ymax = 512
                img = A.Crop(x_min=xmin, y_min=ymin, x_max=xmax,
                             y_max=ymax)(image=np.array(img))
                img = img['image']
            file_name = '_'.join(file.split('/'))
            img = transforms.ToPILImage()(img)
            img.save(
                '/opt/ml/image-classification-level1-15/pytorch-template/data/input/data/train/train_total_crop/'+image_folder+'_'+file_name)
