from torchvision import datasets, transforms
from base import BaseDataLoader
from custom_dataset import *
import albumentations.pytorch


class MaskImageDataLoader(BaseDataLoader):
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True, validation_split=0.2, num_workers=2, training=True):
        self.transform = albumentations.Compose([
            #albumentations.CLAHE(always_apply=False, p=0.5, clip_limit=(1, 15), tile_grid_size=(8, 8)),
            albumentations.CoarseDropout(always_apply=False, p=0.5, max_holes=20, max_height=15, max_width=15, min_holes=1, min_height=8, min_width=8),
            albumentations.Cutout(always_apply=False, p=0.5, num_holes=10, max_h_size=10, max_w_size=10),
            albumentations.Downscale(always_apply=False, p=0.5, scale_min=0.699999988079071, scale_max=0.9900000095367432, interpolation=2),
            albumentations.ElasticTransform(always_apply=False, p=0.5, alpha=0.20000000298023224, sigma=3.359999895095825, alpha_affine=2.009999990463257, interpolation=1, border_mode=1, value=(0, 0, 0), mask_value=None, approximate=False),
            # albumentations.Equalize(always_apply=False, p=0.5, mode='cv', by_channels=False),
            albumentations.GaussNoise(always_apply=False, p=0.5, var_limit=(0.0, 26.849998474121094)),
            albumentations.GridDistortion(always_apply=False, p=0.5, num_steps=1, distort_limit=(-0.029999999329447746, 0.05000000074505806), interpolation=2, border_mode=0, value=(0, 0, 0), mask_value=None),
            albumentations.HorizontalFlip(always_apply=False, p=0.5),
            albumentations.ISONoise(always_apply=False, p=0.5, intensity=(0.05000000074505806, 0.12999999523162842), color_shift=(0.009999999776482582, 0.26999998092651367)),
            albumentations.ImageCompression(always_apply=False, p=0.5, quality_lower=56, quality_upper=100, compression_type=1),
            albumentations.RandomBrightness(always_apply=False, p=0.5, limit=(-0.20000000298023224, 0.14999999105930328)),
            albumentations.Normalize(
                mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
            albumentations.Resize(300, 300),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
        self.data_dir = data_dir
        self.csv_path = csv_path

        # d_type, resize, data_dir, csv_path, transforms, train
        self.dataset = CustomDatasetFromImages(
            self.data_dir, self.csv_path, self.transform, train=training)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)
