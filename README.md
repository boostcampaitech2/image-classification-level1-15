# 떡볶이조  

## Prev

- data 폴더에 input 폴더(이미지포함) 저장
- conda 환경을 강요합니다.

```
source /opt/conda/bin/activate
conda install --file packagelist.txt
conda install -c conda-forge albumentations
conda install -c conda-forge timm
apt-get install libgl1-mesa-glx
```

## USAGE
### Train Single models
1. edit config/(gender_config, none_mask_age_config, mask_age_config, incorrect_mask_age_config, mask_config).json
2. bash start.sh

### Train Ensemble models
1. edit config/age_ensemble_config.json
2. edit model paths in model.py
3. python train.py -c config/age_ensemble_config.json

3. edit config/total_config.json
4. edit model paths in model.py
5. python train.py -c config/total_config.json

### Evaluation
1. edit config/eval_config.json
2. python eval_multi_model.py -c config/eval_config.json `or` python eval_total.py


## Warning
You must edit model paths for load.
```python
# model.py
# 73 line
def init_model(self):
    gender_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_gender/0830_105923/checkpoint-epoch8.pth"
    age_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/age_ensemble/model_best.pth"
    mask_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_mask/0830_121746/checkpoint-epoch11.pth"

# 139 line
def init_model(self):
    none_mask_age_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/none_mask_age/checkpoint-epoch36.pth"
    incorrect_mask_age_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/incorrect_mask_age/checkpoint-epoch33.pth"
    mask_age_model_path = "/opt/ml/level1-15/pytorch-template/saved/models/mask_age/checkpoint-epoch43.pth"
```
