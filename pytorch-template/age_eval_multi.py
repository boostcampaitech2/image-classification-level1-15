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
        config.eval_init_obj('arch', module_arch, 3),
        config.eval_init_obj('arch', module_arch, 4),
        config.eval_init_obj('arch', module_arch, 5)
    ]
    return models


def get_latest_saved_model_paths(config):
    checkpoint_path = "/opt/ml/level1-15/pytorch-template/saved/models/"
    save_paths = [
        checkpoint_path + config['save_directory_name']['gender'],
        checkpoint_path + config['save_directory_name']['none_mask_age'],
        checkpoint_path + config['save_directory_name']['incorrect_mask_age'],
        checkpoint_path + config['save_directory_name']['mask_age'],
        checkpoint_path + config['save_directory_name']['mask']
    ]

    latest_saved_directory = [
        sorted(os.listdir(save_paths[0]))[-1],
        sorted(os.listdir(save_paths[1]))[-1],
        sorted(os.listdir(save_paths[2]))[-1],
        sorted(os.listdir(save_paths[3]))[-1],
        sorted(os.listdir(save_paths[4]))[-1]
    ]

    latest_saved_model_paths = [
        "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_gender/0830_105923/checkpoint-epoch8.pth",
        "/opt/ml/level1-15/pytorch-template/saved/models/none_mask_age/0831_024947/checkpoint-epoch36.pth",
        "/opt/ml/level1-15/pytorch-template/saved/models/incorrect_mask_age/0831_024928/checkpoint-epoch33.pth",
        "/opt/ml/level1-15/pytorch-template/saved/models/mask_age/0831_031913/checkpoint-epoch43.pth",
        "/opt/ml/level1-15/pytorch-template/saved/models/multi_augmentation_mask/0830_121746/checkpoint-epoch11.pth"
    ]
    return latest_saved_model_paths


def get_saved_model_state_dict(latest_saved_model_paths):
    checkpoint1 = torch.load(latest_saved_model_paths[0])
    state_dict1 = checkpoint1['state_dict']
    checkpoint2 = torch.load(latest_saved_model_paths[1])
    state_dict2 = checkpoint2['state_dict']
    checkpoint3 = torch.load(latest_saved_model_paths[2])
    state_dict3 = checkpoint3['state_dict']
    checkpoint4 = torch.load(latest_saved_model_paths[3])
    state_dict4 = checkpoint4['state_dict']
    checkpoint5 = torch.load(latest_saved_model_paths[4])
    state_dict5 = checkpoint5['state_dict']

    return [state_dict1, state_dict2, state_dict3, state_dict4, state_dict5]


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

    model1, model2, model3, model4, model5 = init_models(config)
    latest_saved_model_paths = get_latest_saved_model_paths(config)
    state_dict1, state_dict2, state_dict3, state_dict4, state_dict5 = get_saved_model_state_dict(
        latest_saved_model_paths)

    model1.load_state_dict(state_dict1)
    model2.load_state_dict(state_dict2)
    model3.load_state_dict(state_dict3)
    model4.load_state_dict(state_dict4)
    model5.load_state_dict(state_dict5)

    # prepare model for testing
    model1 = model1.to(device)
    model2 = model2.to(device)
    model3 = model3.to(device)
    model4 = model4.to(device)
    model5 = model5.to(device)

    model1.eval()
    model2.eval()
    model3.eval()
    model4.eval()
    model5.eval()

    submission = pd.read_csv(
        config['data_loader']['args']['data_dir'] + 'eval/info.csv')

    gender_pred = []
    incorrect_mask_age_pred = []
    mask_age_pred = []
    none_mask_age_pred = []
    mask_pred = []
    with torch.no_grad():
        for i, image in enumerate(tqdm(data_loader)):
            image = image.to(device)

            # 0:male, 1:female
            output1 = model1(image)
            output2 = model2(image)
            # 0: mask, 2: incorrect, 3: normal
            output3 = model3(image)  # 마스크 착용여부
            output4 = model4(image)
            output5 = model5(image)

            # # 0: age < 30, 1: 30 <= age < 60, 2: 60 <= age
            # output2 = model2(image)
            pred1 = output1.argmax(dim=-1)
            pred2 = output2.argmax(dim=-1)
            pred3 = output3.argmax(dim=-1)
            pred4 = output4.argmax(dim=-1)
            pred5 = output5.argmax(dim=-1)

            gender_pred.extend(pred1.cpu().numpy())
            none_mask_age_pred.extend(pred2.cpu().numpy())
            incorrect_mask_age_pred.extend(pred3.cpu().numpy())
            mask_age_pred.extend(pred4.cpu().numpy())
            mask_pred.extend(pred5.cpu().numpy())

    CLASS_DICT = {
        '000': 0, '001': 1, '002': 2, '010': 3, '011': 4, '012': 5,
        '100': 6, '101': 7, '102': 8, '110': 9, '111': 10, '112': 11,
        '200': 12, '201': 13, '202': 14, '210': 15, '211': 16, '212': 17
    }

    age_pred = []
    for i in range(len(mask_pred)):
        if mask_pred[i] == 0:
            age_pred.append(mask_age_pred[i])
        elif mask_pred[i] == 1:
            age_pred.append(incorrect_mask_age_pred[i])
        elif mask_pred[i] == 2:
            age_pred.append(none_mask_age_pred[i])

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
