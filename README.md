# Baseline

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
1. edit config/(age_config, gender_config, mask_config).json
2. bash start.sh
3. edit config/eval_config.json
4. python eval_multi_model.py -c config/eval_config.json
