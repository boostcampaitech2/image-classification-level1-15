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
import numpy as np
from scipy.stats import mode
from torchvision import transforms
from torchvision.transforms import Resize, ToTensor, Normalize
import os
from custom_dataset import *
from torch.utils.data import Dataset, DataLoader


def main(config):
    #data_dir, d_type, csv_path, resize,
    # data_loader = getattr(module_data, config['data_loader']['type'])(
    #     data_dir=config['data_loader']['args']['data_dir'],
    #     d_type=config['data_loader']['args']['d_type'],
    #     resize=config['data_loader']['args']['resize'],
    #     batch_size=64,
    #     shuffle=False,
    #     validation_split=0.0,
    #     training=False,
    #     num_workers=2,
    #     csv_path=config['data_loader']['args']['csv_path'],
    # )

    # 테스트 데이터셋 폴더 경로를 지정해주세요.

    test_dir = '/opt/ml/input/data/eval'
    # meta 데이터와 이미지 경로를 불러옵니다.
    submission = pd.read_csv(os.path.join(test_dir, 'info.csv'))
    image_dir = os.path.join(test_dir, 'images')

    # Test Dataset 클래스 객체를 생성하고 DataLoader를 만듭니다.
    image_paths = [os.path.join(image_dir, img_id)
                   for img_id in submission.ImageID]

    # 모델과 같은 transform을 줄것!!

    transform = transforms.Compose([
        Resize((300, 300), Image.BILINEAR),
        ToTensor(),
        Normalize(mean=(0.5, 0.5, 0.5), std=(0.2, 0.2, 0.2)),
    ])
    dataset = TestDataset(image_paths, transform)

    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=150,
        num_workers=3,
        drop_last=False
    )
    # gender_
    # age
    # mask
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    backbones = config["arch"]["args"]["backbone"]

    all_pred = []

    gender_model = Model(label_name="gender", pretrained_model=backbones[2])
    age_model = SmoothTestingModel(
        num_classes=3, pretrained_model=backbones[1])
    mask_model = Model(num_classes=3, label_name="age",
                       pretrained_model=backbones[2])

    gender_model.to(device)
    age_model.to(device)
    mask_model.to(device)

    gender_model.eval()
    age_model.eval()
    mask_model.eval()

    #save_path = "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/"

    CLASS_DICT = {
        '000': 0, '001': 1, '002': 2, '010': 3, '011': 4, '012': 5,
        '100': 6, '101': 7, '102': 8, '110': 9, '111': 10, '112': 11,
        '200': 12, '201': 13, '202': 14, '210': 15, '211': 16, '212': 17
    }
    submission = pd.read_csv(
        config['data_loader']['args']['data_dir'] + 'eval/info.csv')
    all_preds = []
    for g_e in config["arch"]['args']["backbone_point"][0]:
        for a_e in config["arch"]['args']["backbone_point"][1]:
            for m_e in config["arch"]['args']["backbone_point"][2]:
                checkpoint1 = torch.load(
                    "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/GenderClf/0826_183918/model_best.pth")
                state_dict1 = checkpoint1['state_dict']
                checkpoint2 = torch.load(
                    "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/SmoothTestingModel/0826_192838/model_best.pth")
                state_dict2 = checkpoint2['state_dict']
                checkpoint3 = torch.load(
                    "/opt/ml/image-classification-level1-15/pytorch-template/saved/models/MaskClf/0826_200650/model_best.pth")
                state_dict3 = checkpoint3['state_dict']
                gender_model.load_state_dict(state_dict1)
                age_model.load_state_dict(state_dict2)
                mask_model.load_state_dict(state_dict3)

                gender_pred = []
                age_pred = []
                mask_pred = []
                with torch.no_grad():
                    for i, image in enumerate(tqdm(loader)):
                        # print(image)
                        image = image.to(device)

                        # gender 0=male, 1=female
                        output1 = gender_model(image)
                        output2 = age_model(image)  # age
                        output3 = mask_model(image)  # 마스크 착용여부

                        pred1 = output1.argmax(dim=-1)
                        pred2 = output2.argmax(dim=-1)
                        pred3 = output3.argmax(dim=-1)

                        gender_pred.extend(pred1.cpu().numpy())
                        age_pred.extend(pred2.cpu().numpy())
                        mask_pred.extend(pred3.cpu().numpy())

                preds = zip(gender_pred, age_pred, mask_pred)
                # print(preds)
                labels = [CLASS_DICT[''.join(map(str, [mask, gender, age]))]
                          for gender, age, mask in preds]
                # print(labels)
                all_preds.append(labels)

    all_preds_array = np.array(all_preds)
    voting_arr = mode(all_preds_array, axis=0)

    '''
    
    '''
    # print(all_preds_array)
    submission['ans'] = voting_arr[0][0]
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
