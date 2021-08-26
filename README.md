# Baseline

## USAGE
1. edit config.json (or age_config, gender_config, mask_config)
2. python train.py -c config.json
3. edit test_config.json
4. python eval_multi_model.py -c test_config.json

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
