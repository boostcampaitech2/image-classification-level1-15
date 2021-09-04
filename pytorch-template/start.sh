#!/bin/bash

nohup python train.py -c "config/gender_config.json" &&
    nohup python train.py -c "config/mask_age_config.json" && 
    nohup python train.py -c "config/none_mask_age_config.json" && 
    nohup python train.py -c "config/incorrect_mask_age_config.json" && 
    nohup python train.py -c "config/mask_config.json" &
