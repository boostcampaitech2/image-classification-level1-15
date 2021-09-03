import os
from PIL import Image
from torchvision import transforms
import albumentations as A
import numpy as np
from albumentations.core.transforms_interface import ImageOnlyTransform
from tqdm import tqdm

crop_path = '/opt/ml/image-classification-level1-15/pytorch-template/data/input/data/train/train_total_crop/'
holizen_flip_path = '/opt/ml/image-classification-level1-15/pytorch-template/data/input/data/train/train_horizon/'
image_names = os.listdir(crop_path)
print(image_names)
for image_name in tqdm(image_names):
    file_path = crop_path + image_name
    img = Image.open(file_path)
    img_hf = A.HorizontalFlip(p=1)(image=np.array(img))
    img_hf = img_hf['image']
    img_hf = transforms.ToPILImage()(img_hf)
    img_hf.save(holizen_flip_path + image_name)
