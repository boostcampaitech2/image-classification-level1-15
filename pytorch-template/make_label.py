import os
import argparse

import pandas as pd

"""
ABC 
    A mask :   0 - wear,     1 - incorrect,       2 - Not wear
    B gender : 0 - male,     1 - female
    C age :    0 - <30,      1 - >=30 and <60,    2 - >=60
"""

def get_mask_category(mask):
    if mask=='mask':
        return '0'
    elif mask=='incorrect':
        return '1'
    else:
        return '2'

def get_gender_category(gender):
    if gender=='male':
        return '0'
    else:
        return '1'

def get_age_category(age):
    if age<30:
        return '0'
    elif age<60:
        return '1'
    else:
        return '2'

def get_class_id(mask, gender, age):
    return get_mask_category(mask)+get_gender_category(gender)+get_age_category(age)

CLASS_DICT = {
    '000' : 0, '001' : 1, '002' : 2, '010' : 3, '011' : 4, '012' : 5,
    '100' : 6, '101' : 7, '102' : 8, '110' : 9, '111' : 10, '112' : 11,
    '200' : 12, '201' : 13, '202' : 14, '210' : 15, '211' : 16, '212' : 17
}

def make_label_csv(data_dir, csv_path):
    '''
    Make label.csv in the csv folder
    Args:
        data_dir : path of data diretory
        csv_path : path of csv
    '''
    csv_df = pd.read_csv(csv_path)

    labeled_csv = [['id', 'gender', 'race', 'age', 'mask', 'path', 'label']]
    save_path = os.path.split(csv_path)[0]

    for row in csv_df.values:
        _id, gender, race, age, path = row
        imgs_folder = os.path.join(data_dir, path)
        imgs = os.listdir(imgs_folder)

        for img in imgs:
            mask = ''
            if img.startswith('mask'):
                mask = 'mask'
            elif img.startswith('incorrect'):
                mask = 'incorrect'
            else:
                mask = 'normal'

            mask_category = get_mask_category(mask)
            gender_category = get_gender_category(gender)
            age_categoy = get_age_category(age)
            class_id = mask_category + gender_category + age_categoy
            new_row = [_id, 
                        gender_category, 
                        race,
                        age_categoy,
                        mask_category,
                        os.path.join(path,img), 
                        CLASS_DICT[class_id]]
            labeled_csv.append(new_row)

    labeled_csv = pd.DataFrame(labeled_csv)
    labeled_csv.to_csv(os.path.join(save_path, 'label.csv'), index=False, header=False)


if __name__=='__main__':
    parser = argparse.ArgumentParser('make label')
    parser.add_argument('d', type=str, help='image directory')
    parser.add_argument('c', type=str, help='train.csv file path')
    args = parser.parse_args()
    make_label_csv(args.d, args.c)
    print('success!!')
