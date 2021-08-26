from custom_dataset import AgeLabel50To60Smoothing

data = AgeLabel50To60Smoothing(
    resize=260, data_dir="/opt/ml/image-classification-level1-15/pytorch-template/data/input/data/",
    csv_path="/opt/ml/image-classification-level1-15/pytorch-template/data/train_csv_with_multi_labels.csv", transform=None, train=True)

print(data[15][1])
