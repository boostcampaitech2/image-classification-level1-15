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
from model.model import ViT1, ViT2, ViT3


def main(config):

    data_loader = getattr(module_data, config['data_loader']['type'])(
        config['data_loader']['args']['data_dir'],
        batch_size=64,
        shuffle=False,
        validation_split=0.0,
        training=False,
        num_workers=4,
        csv_path=config['data_loader']['args']['csv_path'],
    )

    print()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model1 = ViT1()
    model2 = ViT2()
    model3 = ViT3()

    checkpoint1 = torch.load(
        "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/GenderViTSGD/0826_183028/model_best.pth")
    # orderddict Object
    state_dict1 = checkpoint1['state_dict']
    checkpoint2 = torch.load(
        "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/AgeViTSGD/0826_163531/model_best.pth")
    state_dict2 = checkpoint2['state_dict']
    checkpoint3 = torch.load(
        "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/MaskViTSGD/0826_234749/model_best.pth")
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

            output1 = model1(image)  # gender 0=male, 1=female
            output2 = model2(image)  # age
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

    all_predictions = []

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
