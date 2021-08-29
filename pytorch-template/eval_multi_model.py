import argparse
import torch
from tqdm import tqdm
import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import pandas as pd
from torchvision import models
import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from logger import TensorboardWriter


def init_models(config):
    models = [
        config.eval_init_obj('arch', module_arch, 1),
        config.eval_init_obj('arch', module_arch, 2),
        config.eval_init_obj('arch', module_arch, 3)
    ]
    return models


def get_latest_saved_model_paths(config):
    checkpoint_path = "/opt/ml/level1-15/pytorch-template/saved/models/"
    save_paths = [
        checkpoint_path + config['save_directory_name']['gender'],
        checkpoint_path + config['save_directory_name']['age'],
        checkpoint_path + config['save_directory_name']['mask']
    ]

    latest_saved_directory = [
        sorted(os.listdir(save_paths[0]))[-1],
        sorted(os.listdir(save_paths[1]))[-1],
        sorted(os.listdir(save_paths[2]))[-1]
    ]

    latest_saved_model_paths = [
        save_paths[0] + "/" + latest_saved_directory[0] + "/model_best.pth",
        save_paths[1] + "/" + latest_saved_directory[1] + "/model_best.pth",
        save_paths[2] + "/" + latest_saved_directory[2] + "/model_best.pth"
    ]
    return latest_saved_model_paths


def get_saved_model_state_dict(latest_saved_model_paths):
    checkpoint1 = torch.load(latest_saved_model_paths[0])
    state_dict1 = checkpoint1['state_dict']
    checkpoint2 = torch.load(latest_saved_model_paths[1])
    state_dict2 = checkpoint2['state_dict']
    checkpoint3 = torch.load(latest_saved_model_paths[2])
    state_dict3 = checkpoint3['state_dict']

    return [state_dict1, state_dict2, state_dict3]


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


def main(config):
    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=2,
        csv_path=config['data_loader']['args']['csv_path'],
    )

    print()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model1, model2, model3 = init_models(config)
    latest_saved_model_paths = get_latest_saved_model_paths(config)
    state_dict1, state_dict2, state_dict3 = get_saved_model_state_dict(
        latest_saved_model_paths)

    model1.load_state_dict(state_dict1)
    model2.load_state_dict(state_dict2)
    model3.load_state_dict(state_dict3)

    # prepare model for testing
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)

    model1.eval()
    model2.eval()
    model3.eval()

    submission = pd.read_csv(
        config['data_loader']['args']['data_dir'] + 'eval/info.csv')

    logger = config.get_logger('trainer', config['trainer']['verbosity'])
    writer = TensorboardWriter(config.log_dir, logger, 'true')

    gender_preds = []
    age_preds = []
    mask_preds = []
    with torch.no_grad():
        for i, images in enumerate(tqdm(data_loader)):
            pred_gender, pred_age, pred_mask = TTA(
                model1, model2, model3, images, device)
            gender_preds.append(pred_gender.cpu().numpy())
            age_preds.append(pred_age.cpu().numpy())
            mask_preds.append(pred_mask.cpu().numpy())

            writer.add_image('input', make_grid(
                images[0].cpu(), nrow=8, normalize=True))
            writer.add_image('input', make_grid(
                images[1].cpu(), nrow=8, normalize=True))
            writer.add_image('input', make_grid(
                images[2].cpu(), nrow=8, normalize=True))

    pred_gender = np.mean(gender_preds, axis=0)
    pred_mask = np.mean(mask_preds, axis=0)
    pred_age = np.mean(age_preds, axis=0)

    pred_mask = pred_mask.argmax(axis=-1)
    pred_gender = pred_gender.argmax(axis=-1)
    pred_age = pred_age.argmax(axis=-1)

    all_predictions = pred_mask * 6 + pred_gender * 3 + pred_age
    submission['ans'] = all_predictions
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
