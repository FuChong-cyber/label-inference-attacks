import torch.utils.data as data
import numpy as np
import pandas as pd
import torch
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, StandardScaler
from torchvision import transforms

from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split

import warnings
warnings.filterwarnings("ignore")


def regularit(df):
    newDataFrame = pd.DataFrame(index=df.index)
    columns = df.columns.tolist()
    for c in columns:
        d = df[c]
        MAX = d.max()
        MIN = d.min()
        newDataFrame[c] = ((d - MIN) / (MAX - MIN)).tolist()
    return newDataFrame


class LiverDataset(data.Dataset):

    def __init__(self, csv_file, train=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.train = train
        self.df = pd.read_csv(csv_file)
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        liver_df = self.df.copy()

        # liver_df = liver_df.drop(["Baseline histological Grading"], axis=1)

        categorical_columns = ['Gender', 'Fever', 'Nausea/Vomting', 'Headache ', 'Diarrhea ',
                               'Fatigue & generalized bone ache ', 'Jaundice ', 'Epigastric pain ']
        # 'Baselinehistological staging'
        scale_columns = ['Age ', 'BMI', 'WBC', 'RBC', 'HGB', 'Plat', 'AST 1', 'ALT 1', 'ALT4', 'ALT 12', 'ALT 24',
                         'ALT 36',
                         'ALT 48',
                         'ALT after 24 w', 'RNA Base', 'RNA 4']
        powertransform = ['RNA 12', 'RNA EOT', 'RNA EF']

        label_encoder = preprocessing.LabelEncoder()
        for i in categorical_columns:
            liver_df[i] = label_encoder.fit_transform(liver_df[i])
        liver_df[scale_columns] = StandardScaler().fit_transform(liver_df[scale_columns])
        liver_df[powertransform] = PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(
            liver_df[powertransform])

        y_val = liver_df['Baselinehistological staging']
        y_val.replace([1, 2, 3, 4], [0, 1, 2, 3], inplace=True)
        y_val = y_val.astype('long')

        liver_df = liver_df.drop(["Baselinehistological staging"], axis=1)

        x_feature = liver_df.columns.tolist()
        x_val = liver_df[x_feature]
        pd.set_option('display.max_columns', None)

        # x_val = regularit(x_val)

        # self.sample_original = x_val.values
        # self.label_original = y_val.values

        x_train, x_test, y_train, y_test = train_test_split(x_val, y_val, test_size=0.2, random_state=42)

        # num_one_class = 600
        # sampling_dict = {1: num_one_class,
        #                  2: num_one_class,
        #                  3: num_one_class,
        #                  0: num_one_class}
        #
        # sm = SMOTE(sampling_strategy=sampling_dict, random_state=42)
        # x_val_new, y_val_new = sm.fit_sample(x_train, y_train)
        # x_train = x_val_new
        # y_train = y_val_new

        self.data_column_name = x_train.columns.values.tolist()  # list
        self.label_column_name = x_test.columns.values.tolist()
        self.train_data = x_train.values  # numpy array
        self.test_data = x_test.values

        self.train_labels = y_train.values
        self.test_labels = y_test.values

        print(csv_file, "train", len(self.train_data), "test", len(self.test_data))

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


class LiverSetup(DatasetSetup):

    def __init__(self):
        super().__init__()
        self.num_classes = 4
        self.size_bottom_out = 4

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        base_dataset = LiverDataset(file_path)
        train_labeled_idxs, train_unlabeled_idxs = train_val_split(base_dataset.train_labels,
                                                                   int(n_labeled / self.num_classes),
                                                                   self.num_classes)
        train_labeled_dataset = LiverLabeled(file_path, train_labeled_idxs, train=True)
        train_unlabeled_dataset = LiverUnlabeled(file_path, train_unlabeled_idxs, train=True)
        train_complete_dataset = LiverLabeled(file_path, None, train=True)
        test_dataset = LiverLabeled(file_path, train=False)
        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def get_transforms(self):
        transforms_ = transforms.Compose([
            transforms.ToTensor(),
        ])
        return transforms_

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        _liver_dataset = LiverDataset(file_path, train)
        return _liver_dataset

    def clip_one_party_data(self, x, half):
        x = x[:, :half]
        return x


class LiverLabeled(LiverDataset):

    def __init__(self, file_path, indexs=None, train=True):
        super(LiverLabeled, self).__init__(file_path, train=train)
        if indexs is not None:
            self.train_data = self.train_data[indexs]
            self.train_labels = np.array(self.train_labels)[indexs]
        self.train_data = np.array(self.train_data, np.float32)
        self.test_data = np.array(self.test_data, np.float32)


class LiverUnlabeled(LiverDataset):

    def __init__(self, file_path, indexs=None, train=True):
        super(LiverUnlabeled, self).__init__(file_path, train=train)
        if indexs is not None:
            self.train_data = self.train_data[indexs]
            # self.train_labels = np.array(self.label_original)[indexs]
        self.train_data = np.array(self.train_data, np.float32)
        self.test_data = np.array(self.test_data, np.float32)
        self.train_labels = np.array([-1 for i in range(len(self.train_labels))])


if __name__ == '__main__':

    filename = 'D:/datasets/HCV-Egy-Data/HCV-Egy-Data.csv'
    liver_train_set = LiverDataset(filename, train=True)
    train_loader = torch.utils.data.DataLoader(
        dataset=liver_train_set,
        batch_size=128, shuffle=True,
        num_workers=4, pin_memory=True
    )
    liver_test_set = LiverDataset(filename, train=False)
    test_loader = torch.utils.data.DataLoader(
        dataset=liver_test_set,
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
