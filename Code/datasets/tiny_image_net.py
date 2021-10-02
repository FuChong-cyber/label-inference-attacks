# Thank you https://github.com/leemengtaiwan/tiny-imagenet/blob/master/TinyImageNet.py
# Libraries

import os
import glob

import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split, image_format_2_rgb

EXTENSION = 'JPEG'
NUM_IMAGES_PER_CLASS = 500
CLASS_LIST_FILE = 'wnids.txt'
VAL_ANNOTATION_FILE = 'val_annotations.txt'


class TinyImageNet(Dataset):
    """Tiny ImageNet data set available from `http://cs231n.stanford.edu/tiny-imagenet-200.zip`.
    Parameters
    ----------
    root: string
        Root directory including `train`, `test` and `val` subdirectories.
    split: string
        Indicating which split to return as a data set.
        Valid option: [`train`, `test`, `val`]
    transform: torchvision.transforms
        A (series) of valid transformation(s).
    in_memory: bool
        Set to True if there is enough memory (about 5G) and want to minimize disk IO overhead.
    """

    def __init__(self, root, split='train', transform=None, in_memory=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.transform = transform
        self.in_memory = in_memory
        self.split_dir = os.path.join(root, self.split)
        self.image_paths = sorted(glob.iglob(os.path.join(self.split_dir, '**', '*.%s' % EXTENSION), recursive=True))
        self.labels = {}  # fname - label number mapping
        self.images = []  # used for in-memory processing

        # build class label - number mapping
        with open(os.path.join(self.root, CLASS_LIST_FILE), 'r') as fp:
            self.label_texts = sorted([text.strip() for text in fp.readlines()])
        self.label_text_to_number = {text: i for i, text in enumerate(self.label_texts)}

        if self.split == 'train':
            for label_text, i in self.label_text_to_number.items():
                for cnt in range(NUM_IMAGES_PER_CLASS):
                    self.labels['%s_%d.%s' % (label_text, cnt, EXTENSION)] = i
        elif self.split == 'val':
            with open(os.path.join(self.split_dir, VAL_ANNOTATION_FILE), 'r') as fp:
                for line in fp.readlines():
                    terms = line.split('\t')
                    file_name, label_text = terms[0], terms[1]
                    self.labels[file_name] = self.label_text_to_number[label_text]

        # read all images into torch tensor in memory to minimize disk IO overhead
        if self.in_memory:
            self.images = [self.read_image(path) for path in self.image_paths]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        file_path = self.image_paths[index]

        if self.in_memory:
            img = self.images[index]
        else:
            img = self.read_image(file_path)

        if self.split == 'test':
            return img
        else:
            # file_name = file_path.split('/')[-1]
            return img, self.labels[os.path.basename(file_path)]

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = self.split
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img


class TinyImageNetSetup(DatasetSetup):

    def __init__(self):
        super().__init__()
        self.num_classes = 200
        self.size_bottom_out = 200

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num=None):
        transforms_ = self.get_transforms()
        base_dataset = TinyImageNet(file_path)
        train_labeled_idxs, train_unlabeled_idxs = train_val_split(list(base_dataset.labels.values()),
                                                                   int(n_labeled / self.num_classes),
                                                                   self.num_classes)
        train_labeled_dataset = TinyImageNetLabeled(file_path, train_labeled_idxs, split='train',
                                                    transform=transforms_)
        train_unlabeled_dataset = TinyImageNetUnlabeled(file_path, train_unlabeled_idxs, split='train',
                                                        transform=transforms_)
        train_complete_dataset = TinyImageNetLabeled(file_path, None, split='train', transform=transforms_)
        test_dataset = TinyImageNetLabeled(file_path, split='val', transform=transforms_)
        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def get_normalize_transform(self):
        normalize_ = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
        return normalize_

    def get_transforms(self):
        normalize = self.get_normalize_transform()
        transforms_ = transforms.Compose([
            transforms.Lambda(image_format_2_rgb),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            normalize
        ])
        return transforms_

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        if train:
            split = 'train'
        else:
            split = 'val'
        transforms_ = self.get_transforms()
        _tiny_imagenet_dataset = TinyImageNet(file_path, split, transform=transforms_)
        return _tiny_imagenet_dataset

    def clip_one_party_data(self, x, half):
        x = x[:, :, :, :half]
        return x


class TinyImageNetLabeled(TinyImageNet):

    def __init__(self, root, indexs=None, split='train', transform=None):
        super(TinyImageNetLabeled, self).__init__(root, split=split,
                                                  transform=transform)
        if indexs is not None:
            temp_image_paths = []
            for id in indexs:
                temp_image_paths.append(self.image_paths[id])
            self.image_paths = temp_image_paths


class TinyImageNetUnlabeled(TinyImageNetLabeled):

    def __init__(self, root, indexs, split='train',
                 transform=None):
        super(TinyImageNetUnlabeled, self).__init__(root, indexs, split=split,
                                                    transform=transform)
        for key in self.labels.keys():
            self.labels[key] = -1


if __name__ == '__main__':
    normalize_imagenet = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])

    augmentation = transforms.RandomApply([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.RandomResizedCrop(64)], p=.8)

    training_transform = transforms.Compose([
        transforms.Lambda(image_format_2_rgb),
        augmentation,
        transforms.ToTensor(),
        normalize_imagenet])

    test_transform = transforms.Compose([
        transforms.Lambda(image_format_2_rgb),
        transforms.ToTensor(),
        normalize_imagenet])

    dataset_train = TinyImageNet(root='D:\\Datasets\\tiny-imagenet-200',
                                 split='train')
    dataset_test = TinyImageNet(root='D:\\Datasets\\tiny-imagenet-200',
                                split='val')
    train_loader = torch.utils.data.DataLoader(
        dataset=dataset_train,
        batch_size=1, shuffle=True,
        num_workers=4, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=dataset_test,
        batch_size=1, shuffle=False,
        num_workers=4, pin_memory=True
    )
    print("len train loader:", len(train_loader))
    for batch_id, (data, target) in enumerate(train_loader):
        print("batch_id:", batch_id)
        # print("batch datasets:", data)
        print("batch datasets shape:", data.shape)
        print("batch target:", target)
        break
    print("\n\n test-->")
    print("len test loader:", len(test_loader))
    for batch_id, (data, target) in enumerate(test_loader):
        print("batch_id:", batch_id)
        # print("batch datasets:", data)
        print("batch datasets shape:", data.shape)
        print("batch target:", target)
        break
    for data, target in test_loader:
        # print("batch datasets:", data)
        print("batch datasets shape:", data.shape)
        print("batch target:", target)
        break
