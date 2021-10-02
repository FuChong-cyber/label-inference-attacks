"""
thanks: https://github.com/swapniel99/criteo/blob/master/criteo.py
"""
import torch.utils.data as data
from csv import DictReader
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split

import itertools

import warnings

warnings.filterwarnings("ignore")

D = 2 ** 13  # number of weights use for learning
BATCH_SIZE = 1000
header = ['Label', 'i1', 'i2', 'i3', 'i4', 'i5', 'i6', 'i7', 'i8', 'i9', 'i10', 'i11', 'i12', 'i13', 'c1', 'c2', 'c3',
          'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11', 'c12', 'c13', 'c14', 'c15', 'c16', 'c17', 'c18', 'c19',
          'c20', 'c21', 'c22', 'c23', 'c24', 'c25', 'c26']


def get_csv_row_by_index(reader, index):
    row = itertools.islice(reader, index - 1, index).__next__()
    return row


class Criteo(data.Dataset):

    def __init__(self, processed_csv_file_path, batch_size=BATCH_SIZE, train=True, total_samples_num=1e5, test_size=0.2):
        """
        Args:
            processed_csv_file_path (string): Path to the criteo.csv file.
        """
        self.total_samples_num = total_samples_num
        self.test_size = test_size
        self.train_samples_num = int(self.total_samples_num * (1 - self.test_size))
        self.test_samples_num = int(self.total_samples_num * self.test_size)
        self.train_batches_num = int(self.train_samples_num / batch_size)
        self.test_batches_num = int(self.test_samples_num / batch_size)
        self.train = train
        self.processed_csv_file_path = processed_csv_file_path
        self.batch_size = batch_size

        df_labels = pd.read_csv(processed_csv_file_path, nrows=self.total_samples_num, usecols=['label'])
        y_val = df_labels.astype('long')
        self.labels = y_val.values.reshape(-1, batch_size)

    def __len__(self):
        # print(f'The Criteo DATALOADER\'s batch quantity. Batch size is {self.batch_size}:')
        if self.train:
            return self.train_batches_num
        else:
            return self.test_batches_num

    def __getitem__(self, index):
        # index is for batch
        if self.train:
            index = index
        else:
            index = index + self.train_batches_num

        temp_df = pd.read_csv(self.processed_csv_file_path, skiprows=index * self.batch_size, nrows=self.batch_size)
        temp_df = temp_df.drop(temp_df.columns[-1], axis=1)
        feature_names = temp_df.columns.tolist()
        x_feature = temp_df[feature_names]
        feat_ = x_feature.values
        label_ = self.labels[index]

        feat_ = torch.tensor(feat_)
        label_ = torch.tensor(label_)

        return feat_, label_


class CriteoSetup(DatasetSetup):

    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.size_bottom_out = 4

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        train_labeled_dataset = CriteoLabeled(file_path, n_labeled, train=True)
        train_unlabeled_dataset = CriteoUnlabeled(file_path, n_labeled, train=True)
        train_complete_dataset = Criteo(file_path, train=True)
        test_dataset = Criteo(file_path, train=False)
        print("#Labeled:", len(train_labeled_dataset),
              "#Unlabeled:", len(train_unlabeled_dataset))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def get_transforms(self):
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transforms_

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        _dataset = Criteo(file_path, batch_size=BATCH_SIZE, train=train)
        return _dataset

    def clip_one_party_data(self, x, half):
        x = x[:, :half]
        return x


class CriteoLabeled(Criteo):

    def __init__(self, file_path, n_labeled, train=True):
        super(CriteoLabeled, self).__init__(file_path, batch_size=100, train=train, total_samples_num=n_labeled, test_size=0.)


class CriteoUnlabeled(Criteo):

    def __init__(self, file_path, n_labeled, train=True):
        super(CriteoUnlabeled, self).__init__(file_path, batch_size=100, train=train, total_samples_num=1e6 - n_labeled, test_size=0.)
        self.n_labeled = n_labeled

    def __getitem__(self, index):
        index += self.n_labeled
        feat_, label_ = super().__getitem__(index)
        return feat_, label_


if __name__ == "__main__":

    path = 'D:/Datasets/Criteo/criteo.csv'
    dataset = Criteo(path, batch_size=5, train=True)
    print('dataset constructed')
    print(f'len dataset:{len(dataset)}')

    feat, label = dataset[10]
    print(f"len feat:{len(feat)}")
    print(f"feat:{feat}")
    print(f"label:{label}")

    # data_loader = DataLoader(dataset, 4)
    # print('dataloader constructed')
    for feat, label in dataset:
        print(f"feat.shape:{feat.shape}")
        print(f"label:{label}")
        # break
