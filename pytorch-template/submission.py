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
import timm
from model.model import *


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

    model1 = GenderClassifier()
    model2 = AgeClassifier()
    model3 = MaskClassifier()

    checkpoint1 = torch.load(
        "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/GenderClassificationEfficientnet/0825_044850/model_best.pth")
    state_dict1 = checkpoint1['state_dict']
    checkpoint2 = torch.load(
        "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/AgeClassificationEfficientnet/0825_040108/model_best.pth")
    state_dict2 = checkpoint2['state_dict']
    checkpoint3 = torch.load(
        "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/MaskClassificationEfficientnet/0825_032519/model_best.pth")
    state_dict3 = checkpoint3['state_dict']

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

            output1 = model1(image)
            output2 = model2(image)
            output3 = model3(image)

            pred1 = output1.argmax(dim=-1)
            pred2 = output2.argmax(dim=-1)
            pred3 = output3.argmax(dim=-1)

            gender_pred.extend(pred1.cpu().numpy())
            age_pred.extend(pred2.cpu().numpy())
            mask_pred.extend(pred3.cpu().numpy())

    all_predictions = []
    for i in range(len(mask_pred)):
        if mask_pred[i] == 0:
            if gender_pred[i] == 0:
                if age_pred[i] == 0:
                    all_predictions.append(0)
                elif age_pred[i] == 1:
                    all_predictions.append(1)
                elif age_pred[i] == 2:
                    all_predictions.append(2)
            elif gender_pred[i] == 1:
                if age_pred[i] == 0:
                    all_predictions.append(3)
                elif age_pred[i] == 1:
                    all_predictions.append(4)
                elif age_pred[i] == 2:
                    all_predictions.append(5)
        elif mask_pred[i] == 1:
            if gender_pred[i] == 0:
                if age_pred[i] == 0:
                    all_predictions.append(6)
                elif age_pred[i] == 1:
                    all_predictions.append(7)
                elif age_pred[i] == 2:
                    all_predictions.append(8)
            elif gender_pred[i] == 1:
                if age_pred[i] == 0:
                    all_predictions.append(9)
                elif age_pred[i] == 1:
                    all_predictions.append(10)
                elif age_pred[i] == 2:
                    all_predictions.append(11)
        elif mask_pred[i] == 2:
            if gender_pred[i] == 0:
                if age_pred[i] == 0:
                    all_predictions.append(12)
                elif age_pred[i] == 1:
                    all_predictions.append(13)
                elif age_pred[i] == 2:
                    all_predictions.append(14)
            elif gender_pred[i] == 1:
                if age_pred[i] == 0:
                    all_predictions.append(15)
                elif age_pred[i] == 1:
                    all_predictions.append(16)
                elif age_pred[i] == 2:
                    all_predictions.append(17)
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
