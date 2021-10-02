import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split

import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
import seaborn as sns


class BcwDataset(data.Dataset):

    def __init__(self, csv_path, train=True):
        """
        Args:
            csv_path (string): Path to the csv file.
        """
        self.train = train
        self.df = pd.read_csv(csv_path)
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.area_mean = None

        self.df = self.df.drop('Unnamed: 32', axis=1)
        self.df = self.df.drop('id', axis=1)
        # sequence adjustment
        radius_mean = self.df['radius_mean']
        self.df = self.df.drop('radius_mean', axis=1)
        self.df['radius_mean'] = radius_mean
        perimeter_mean = self.df['perimeter_mean']
        self.df = self.df.drop('perimeter_mean', axis=1)
        self.df['perimeter_mean'] = perimeter_mean

        self.area_mean = self.df['area_mean']
        self.df = self.df.drop('area_mean', axis=1)

        le = LabelEncoder()
        self.df['diagnosis'] = le.fit_transform(self.df['diagnosis'])

        x = self.df.iloc[:, 1:]
        y = self.df.iloc[:, 0]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=16)

        sc = StandardScaler()

        x_train = sc.fit_transform(x_train)
        x_test = sc.fit_transform(x_test)

        self.train_data = x_train  # numpy array
        self.test_data = x_test

        self.train_labels = y_train.tolist()
        self.test_labels = y_test.tolist()

        print(csv_path, "train", len(self.train_data), "test", len(self.test_data))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            data, label = self.train_data[index], self.train_labels[index]
        else:
            data, label = self.test_data[index], self.test_labels[index]

        return data, label


class BcwSetup(DatasetSetup):

    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.size_bottom_out = 2

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        base_dataset = BcwDataset(file_path)
        train_labeled_idxs, train_unlabeled_idxs = train_val_split(base_dataset.train_labels,
                                                                   int(n_labeled / self.num_classes),
                                                                   self.num_classes)
        train_labeled_dataset = BcwLabeled(file_path, train_labeled_idxs, train=True)
        train_unlabeled_dataset = BcwUnlabeled(file_path, train_unlabeled_idxs, train=True)
        train_complete_dataset = BcwLabeled(file_path, None, train=True)
        test_dataset = BcwLabeled(file_path, train=False)
        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def get_transforms(self):
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transforms_

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        _liver_dataset = BcwDataset(file_path, train)
        return _liver_dataset

    def clip_one_party_data(self, x, half):
        x = x[:, :half]
        return x


class BcwLabeled(BcwDataset):

    def __init__(self, file_path, indexs=None, train=True):
        super(BcwLabeled, self).__init__(file_path, train=train)
        if indexs is not None:
            self.train_data = self.train_data[indexs]
            self.train_labels = np.array(self.train_labels)[indexs]
        self.train_data = np.array(self.train_data, np.float32)
        self.test_data = np.array(self.test_data, np.float32)


class BcwUnlabeled(BcwDataset):

    def __init__(self, file_path, indexs=None, train=True):
        super(BcwUnlabeled, self).__init__(file_path, train=train)
        if indexs is not None:
            self.train_data = self.train_data[indexs]
            # self.train_labels = np.array(self.label_original)[indexs]
        self.train_data = np.array(self.train_data, np.float32)
        self.test_data = np.array(self.test_data, np.float32)
        self.train_labels = np.array([-1 for i in range(len(self.train_labels))])


if __name__ == '__main__':
    path = 'D:/Datasets/BreastCancerWisconsin/wisconsin.csv'
    dataset_setup = BcwSetup()
    train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset =\
    dataset_setup.set_datasets_for_ssl(file_path=path, n_labeled=20, party_num=2)

    bcw_train_set = BcwDataset(path, train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=bcw_train_set,
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True
    )
    bcw_test_set = BcwDataset(path, train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=bcw_test_set,
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True
    )
    print("len train loader:", len(train_loader))
    for batch_id, (data, target) in enumerate(train_loader):
        print("batch_id:", batch_id)
        print("batch datasets:", data)
        print("batch datasets shape:", data.shape)
        print("batch target:", target)
        break
    print("\n\n test-->")
    print("len test loader:", len(test_loader))
    for batch_id, (data, target) in enumerate(test_loader):
        print("batch_id:", batch_id)
        print("batch datasets:", data)
        print("batch target:", target)
        break

