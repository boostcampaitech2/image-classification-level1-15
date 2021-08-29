# AugMix & TTA

[AugMix](https://github.com/google-research/augmix)와 TTA를 적용했습니다.

![TTA 예시사진](https://user-images.githubusercontent.com/49181231/131235250-40ffab97-1644-45ac-bac8-7e502b8bc66e.png)

## Warning

- Image Size에 주의해주세요.
    - AugMix의 Default 값은 224로 되어있습니다.
    - `custom_dataset.py - CustomDatasetFromImages - init` 함수에서 `self.IMAGE_SIZE`와 `self.preprocess`에 들어있는 `RandomResizedCrop` 값을 사용을 원하시는 사이즈로 변경해야합니다. 또한 `data_loader - MaskImageDataLoader와 MaskImageValidDataLoader`의 `self.transform -> Resize` 크기도 똑같은걸로 변경해주셔야 합니다.


## AugMix

사전에 지정한 Augmentation과 MixCut을 이미지에 적용합니다.

```python
# custom_dataset.py - CustomDatasetFromImages
self.preprocess = A.Compose([
    A.RandomResizedCrop(224, 224),
    A.HorizontalFlip(),
    A.Normalize(mean, std),
    ToTensorV2(),
])

self.augmentations = [
    self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
    self.translate_x, self.translate_y
]

self.augmentations_all = [
    self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x, self.shear_y,
    self.translate_x, self.translate_y, self.color, self.contrast, self.brightness, self.sharpness
]
```

Train Phase에서는 데이터로더에서 preprocess, augmentations, augmentations_all 가 각각 적용된 tuple 타입의 이미지를 받아오게 됩니다. 그리고 해당 튜플 이미지는 각각 `trainer.py` 의 `_train_epoch` 에서 `logits_clean, logits_aug1, logits_aug2` 라는 이름으로 사용되며 softmax 연산을 통해 확률로 변환한 뒤 `kl_div`로 loss를 계산합니다.

*Validation Phase 에는 포함되지 않습니다.* 

## TTA

```python
def TTA(model1, model2, model3, images, device):
    for i in range(len(images)):
        if i == 0:
            preds_gender = model1(images[i].to(device)).to(device)
            preds_age = model2(images[i].to(device)).to(device)
            preds_mask = model3(images[i].to(device)).to(device)
        else:
            pred_gender = model1(images[i].to(device)).to(device)
            pred_age = model2(images[i].to(device)).to(device)
            pred_mask = model3(images[i].to(device)).to(device)

            preds_gender = torch.cat((preds_gender, pred_gender), dim=0)
            preds_age = torch.cat((preds_age, pred_age), dim=0)
            preds_mask = torch.cat((preds_mask, pred_mask), dim=0)

    return torch.mean(preds_gender, dim=0), torch.mean(preds_age, dim=0), torch.mean(preds_mask, dim=0)
```

Evaluation Phase 에서 사용되며 현재 코드에는 AugMix 를 그대로 사용하고 있습니다.

위와 같이 세 개의 이미지(clean, aug1, aug2)가 결합된 튜플을 받아서 각각 모델의 output을 예측한 뒤
예측값들의 평균으로 라벨을 계산합니다.

변경을 위해서는 아래 list에 augmentation 함수를 추가해주시면 됩니다.

```python
# custom_dataset.py - CustomDatasetFromImages
self.test_augmentations = [
    self.autocontrast, self.translate_x, self.rotate
]

self.test_augmentations_all = [
    self.autocontrast, self.equalize, self.posterize, self.rotate, self.solarize, self.shear_x
]
```

또한 Evaluation 시 Tensorboard에 이미지들을 기록하게 해두었으니 확인 바랍니다!
- 현재 TTA_Evaluation 이란 이름으로 로그 생성