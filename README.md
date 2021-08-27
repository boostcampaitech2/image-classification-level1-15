# image-classification-level1-15
imHello World
Hello Worldage-classification-level1-15 created by GitHub Classroom
# Baseline

## 수정사항

- config 파일을 전부 config 폴더에 옮겼습니다.
- 쓸모없는 파일 삭제 및 디렉토리 정리했습니다.
- config 파일 수정 만으로 원하는 사전학습 모델을 불러와 원하는 태스크 별 라벨을 학습할 수 있도록 하였습니다.
- evaluation 시 학습모델을 저장한 폴더와 모델 이름만 eval_config.json 파일에 작성하면 자동으로 best_model 경로를
  불러와서 테스트할 수 있도록 만들었습니다.
- albumentation 을 추가하였습니다.

## Prev

- data 폴더에 input 폴더(이미지포함) 저장
- conda 환경을 강요합니다.

```
conda install --file packagelist.txt
conda activate base
conda install -c conda-forge albumentations
```

## USAGE
1. edit config/config.json (or age_config, gender_config, mask_config)
2. python train.py -c config/config.json
3. edit config/eval_config.json
4. python eval_multi_model.py -c config/eval_config.json

## Edit config

```
{
    ## 모델을 저장할 폴더명
    "name": "GenderClf",
    "n_gpu": 1,
    "arch": {
        # 모델 이름 (수정 x)
        "type": "Model",
        "args": {
            # 예측할 클래스 개수
            "num_classes": 2,
            # 예측할 라벨 이름 [gender, age, mask, total]
            "label_name": "gender",
            # 불러올 사전학습모델 이름 (timm 에 있는 것만 가능)
            "pretrained_model": "efficientnet_b1",
            # 사전학습 모델의 output feature
            "pretrained_out_feature": 1000
        }
    },
    "data_loader": {
        # 데이터로더 이름
        "type": "MaskImageDataLoader",
        "args": {
            "data_dir": "data/input/data/",
            "batch_size": 64,
            "shuffle": true,
            "validation_split": 0.2,
            "num_workers": 2,
            "csv_path": "data/train_csv_with_multi_labels.csv"
        }
    },
    "optimizer": {
        "type": "SGD",
        "args": {
            "lr": 0.01,
            "momentum": 0.9
        }
    },
    "loss": "cross_entropy_loss",
    "metrics": [
        "accuracy",
        "f1"
    ],
    "lr_scheduler": {
        "type": "StepLR",
        "args": {
            "step_size": 50,
            "gamma": 0.1
        }
    },
    "trainer": {
        "epochs": 3,
        "save_dir": "saved/",
        "save_period": 1,
        "verbosity": 2,
        "monitor": "min val_loss",
        "early_stop": 2,
        "tensorboard": true
    }
}
```
