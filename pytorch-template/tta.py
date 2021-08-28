import torch
import albumentations.pytorch
import timm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

image_path = "/opt/ml/image-classification-level1-15/pytorch-template/data/input/data/eval/images/0a2bd33bf76d7426f3d6ca0b7fbe03ee431159b4.jpg"
image = np.array(Image.open(image_path)) / 255  # 이미지를 읽고 min max scaling
image = cv2.resize(image, (384, 384))  # Vision Transformer base model input size
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).to(torch.float32)

print(image.shape)
transform = albumentations.Compose([
            albumentations.ColorJitter(brightness=(0.2, 2), contrast=(
                0.3, 2), saturation=(0.2, 2), hue=(-0.3, 0.3)),
            albumentations.HorizontalFlip(),
            albumentations.augmentations.transforms.ToGray(),
            albumentations.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            albumentations.Resize(384, 384),
            albumentations.pytorch.transforms.ToTensorV2()
        ]

model = timm.create_model("vit_base_patch16_384", pretrained=True)
# model = timm.create_model("seresnet50", pretrained=True)

imagenet_labels = dict(enumerate(open('classes.txt')))  # ImageNet class name
fig = plt.figure(figsize=(20, 20))
columns = 3
rows = 3
for i, transformer in enumerate(transforms):  # custom transforms

    augmented_image = transformer.augment_image(image)
    output = model(augmented_image)
    predicted = imagenet_labels[output.argmax(1).item()].strip()
    
    augmented_image = np.array((augmented_image*255).squeeze()).transpose(1, 2, 0).astype(np.uint8)
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(augmented_image)
    plt.title(predicted)

plt.show()