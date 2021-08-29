#!/bin/bash

nohup python train.py -c "config/gender_config.json" && 
nohup python train.py -c "config/age_config.json" && 
nohup python train.py -c "config/mask_config.json"&