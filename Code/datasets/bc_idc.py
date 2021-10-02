# Code extensively uses the kernel https://www.kaggle.com/bonhart/pytorch-cnn-from-scratch!
# Thank you https://www.kaggle.com/bonhart
# Libraries
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from glob import glob
from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split


class IdcSetup(DatasetSetup):

    def __init__(self):
        super().__init__()
        self.num_classes = 2
        self.size_bottom_out = 5

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num):
        base_dataset = IdcDataset(file_path, party_num=party_num)
        transforms_ = self.get_transforms()
        train_labeled_idxs, train_unlabeled_idxs = train_val_split(base_dataset.train_labels,
                                                                   int(n_labeled / self.num_classes),
                                                                   self.num_classes)
        train_labeled_dataset = IdcLabeled(file_path, party_num, train_labeled_idxs, train=True, transform=transforms_)
        train_unlabeled_dataset = IdcUnlabeled(file_path, party_num, train_unlabeled_idxs, train=True,
                                               transform=transforms_)
        train_complete_dataset = IdcLabeled(file_path, party_num, None, train=True, transform=transforms_)
        test_dataset = IdcLabeled(file_path, party_num, None, train=False, transform=transforms_)

        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def get_normalize_transform(self):
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    def get_transforms(self):
        normalize_ = self.get_normalize_transform()
        transforms_ = transforms.Compose([transforms.ToPILImage(),
                                          transforms.ToTensor(),
                                          normalize_
                                          ])
        return transforms_

    def get_transformed_dataset(self, file_path, party_num, train):
        transforms_ = self.get_transforms()
        idc_dataset_ = IdcDataset(file_path, party_num, train, transform=transforms_)
        return idc_dataset_

    def clip_one_party_data(self, x, half=1):
        input_img_tuple = tuple(x[:, i:i+1, :, :, :] for i in range(0, half))
        input_tensor_adversary = torch.cat(input_img_tuple, dim=4).squeeze(1)
        return input_tensor_adversary


class IdcDataset(Dataset):

    def __init__(self, file_path, party_num,
                 train=True, transform=None):
        """
        Args:
            file_path (string): Path to the csv file with annotations.
        """
        self.train = train
        self.party_num = party_num
        self.transform = transform
        self.train_data = []
        self.train_labels = []
        self.test_data = []
        self.test_labels = []
        self.num_classes = 2
        # get 0/1 img paths list
        img_paths_list_zero = []
        img_paths_list_one = []
        img_paths_list = glob(file_path + '/IDC_regular_ps50_idx5/**/*.png', recursive=True)
        for img_path in img_paths_list:
            if img_path[-5] == "0":
                img_paths_list_zero.append(img_path)
            elif img_path[-5] == "1":
                img_paths_list_one.append(img_path)
        # pack paths as path_groups_zero/one according to party_num
        groups_num_zero = int(len(img_paths_list_zero) / party_num)
        groups_num_one = int(len(img_paths_list_one) / party_num)
        path_groups_zero = []
        path_groups_one = []
        for group_zore_id in range(groups_num_zero):
            path_groups_zero.append(
                img_paths_list_zero[group_zore_id * party_num: group_zore_id * party_num + party_num])
        for group_one_id in range(groups_num_one):
            path_groups_one.append(img_paths_list_one[group_one_id * party_num: group_one_id * party_num + party_num])
        # merge groups zero/one
        labels_group = [0] * groups_num_zero
        labels_group.extend([1] * groups_num_one)
        path_groups_zero.extend(path_groups_one)
        path_groups = path_groups_zero
        # Splitting data into train and val
        X_train, X_test, Y_train, Y_test = train_test_split(path_groups, labels_group, stratify=labels_group,
                                                            test_size=0.2,
                                                            random_state=1)
        # y_val = y_val.astype('long')
        self.train_data = X_train
        self.test_data = X_test
        self.train_labels = Y_train
        self.test_labels = Y_test
        print(file_path, "train", len(self.train_data), "test", len(self.test_data))

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def __getitem__(self, index):
        if self.train:
            path_group, label = self.train_data[index], self.train_labels[index]
        else:
            path_group, label = self.test_data[index], self.test_labels[index]
        images_list = []
        for img_id in range(self.party_num):
            img_path = path_group[img_id]
            image = cv2.imread(img_path)
            image = cv2.resize(image, (50, 50))
            if self.transform is not None:
                image = self.transform(image)
            images_list.append(image)
        images = torch.stack(tuple(image for image in images_list), 0)
        return images, label


class IdcLabeled(IdcDataset):

    def __init__(self, file_path, party_num, indexs=None, train=True, transform=None):
        super(IdcLabeled, self).__init__(file_path, train=train, party_num=party_num, transform=transform)
        if indexs is not None:
            temp_train_data = []
            for id in indexs:
                temp_train_data.append(self.train_data[id])
            self.train_data = temp_train_data
            self.train_labels = np.array(self.train_labels)[indexs]


class IdcUnlabeled(IdcLabeled):

    def __init__(self, file_path, party_num, indexs, train=True, transform=None):
        super(IdcUnlabeled, self).__init__(file_path, party_num, indexs=indexs, train=train, transform=transform)
        self.train_labels = np.array([-1 for i in range(len(self.train_labels))])


if __name__ == '__main__':
    idcDataset = IdcDataset()
    train_loader = idcDataset.get_trainloader()
    print("len train loader:", len(train_loader))
    for batch_id, (data, target) in enumerate(train_loader):
        print("batch_id:", batch_id)
        print("batch datasets:", data)
        print("batch datasets shape:", data.shape)
        print("batch target:", target)
        break
    print("\n\n test-->")
    test_loader = idcDataset.get_testloader()
    print("len test loader:", len(test_loader))
    for batch_id, (data, target) in enumerate(test_loader):
        print("batch_id:", batch_id)
        print("batch datasets:", data)
        print("batch target:", target)
        break
