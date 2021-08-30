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


def init_models(config):
    models = [
        config.eval_init_obj('arch', module_arch, 1),
        config.eval_init_obj('arch', module_arch, 2),
        config.eval_init_obj('arch', module_arch, 3)
    ]
    return models


def get_latest_saved_model_paths(config):
    checkpoint_path = "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/"
    save_paths = [
        checkpoint_path + config['save_directory_name']['gender'],
        checkpoint_path + config['save_directory_name']['age'],
        checkpoint_path + config['save_directory_name']['mask']
    ]

    # print(save_paths)
    # for path in save_paths:
    #     print(path)
    #     print(11111)
    #     print(sorted(os.listdir(path)))

    # TODO 예외처리
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

    gender_pred = []
    age_pred = []
    mask_pred = []
    with torch.no_grad():
        for i, image in enumerate(tqdm(data_loader)):
            image = image.to(device)

            # 0:male, 1:female
            output1 = model1(image)
            # 0: age < 30, 1: 30 <= age < 60, 2: 60 <= age
            output2 = model2(image)
            # 0: mask, 2: incorrect, 3: normal
            output3 = model3(image)  # 마스크 착용여부

            pred1 = output1.argmax(dim=-1)
            pred2 = output2.argmax(dim=-1)
            pred3 = output3.argmax(dim=-1)

            gender_pred.extend(pred1.cpu().numpy())
            age_pred.extend(pred2.cpu().numpy())
            mask_pred.extend(pred3.cpu().numpy())

    CLASS_DICT = {
        '000': 0, '001': 1, '002': 2, '010': 3, '011': 4, '012': 5,
        '100': 6, '101': 7, '102': 8, '110': 9, '111': 10, '112': 11,
        '200': 12, '201': 13, '202': 14, '210': 15, '211': 16, '212': 17
    }

    preds = zip(gender_pred, age_pred, mask_pred)
    labels = [CLASS_DICT[''.join(map(str, [mask, gender, age]))]
              for gender, age, mask in preds]
    submission['ans'] = labels
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
