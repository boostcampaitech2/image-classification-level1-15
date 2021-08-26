from custom_dataset import AgeLabel50To60Smoothing

data = AgeLabel50To60Smoothing(
    resize=260, data_dir="data/input/data/", csv_path="data/train_csv_with_multi_labels.csv", transform=None, train=True)

data[0]
