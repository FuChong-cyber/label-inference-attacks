import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from datasets.dataset_setup import DatasetSetup
from my_utils.utils import train_val_split, image_format_2_rgb


class CINIC10L(Dataset):
    def __init__(self, root, split='train', transform=None):
        super().__init__()
        image_folder = torchvision.datasets.ImageFolder(root=root + '/' + split)
        self.targets = image_folder.targets
        self.image_paths = image_folder.imgs
        self.transform = transform

    def __getitem__(self, index):
        file_path, label = self.image_paths[index]
        img = self.read_image(file_path)
        return img, label

    def __len__(self):
        return len(self.image_paths)

    def read_image(self, path):
        img = Image.open(path)
        return self.transform(img) if self.transform else img


class Cinic10LSetup(DatasetSetup):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.size_bottom_out = 10

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num):
        transforms_ = self.get_transforms()
        base_dataset = CINIC10L(file_path)
        train_labeled_idxs, train_unlabeled_idxs = train_val_split(base_dataset.targets,
                                                                   int(n_labeled / self.num_classes),
                                                                   self.num_classes)
        train_labeled_dataset = CINIC10LLabeled(file_path, train_labeled_idxs, split='train', transform=transforms_)
        train_unlabeled_dataset = CINIC10LUnlabeled(file_path, train_unlabeled_idxs, split='train',
                                                    transform=transforms_)
        train_complete_dataset = CINIC10LLabeled(file_path, None, split='train', transform=transforms_)
        test_dataset = CINIC10LLabeled(file_path, split='test', transform=transforms_)
        print("#Labeled:", len(train_labeled_idxs), "#Unlabeled:", len(train_unlabeled_idxs))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_dataset

    def get_normalize_transform(self):
        normalize_cinic = transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404],
                                               std=[0.24205776, 0.23828046, 0.25874835])
        return normalize_cinic

    def get_transforms(self):
        normalize = self.get_normalize_transform()
        transforms_ = transforms.Compose([
            transforms.Lambda(image_format_2_rgb),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize
        ])
        return transforms_

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        if train:
            split = 'train'
        else:
            split = 'test'
        transforms_ = self.get_transforms()
        _cinic10_dataset = CINIC10L(file_path, split, transform=transforms_)
        return _cinic10_dataset

    def clip_one_party_data(self, x, half):
        x = x[:, :, :, :half]
        return x


class CINIC10LLabeled(CINIC10L):

    def __init__(self, root, indexs=None, split='train',
                 transform=None):
        super(CINIC10LLabeled, self).__init__(root, split=split,
                                              transform=transform
                                              )
        if indexs is not None:
            temp_image_paths = []
            for id in indexs:
                temp_image_paths.append(self.image_paths[id])
            self.image_paths = temp_image_paths


class CINIC10LUnlabeled(CINIC10LLabeled):

    def __init__(self, root, indexs, split='train',
                 transform=None):
        super(CINIC10LUnlabeled, self).__init__(root, indexs, split=split,
                                                transform=transform
                                                )
        temp_image_paths = []
        for image_path, label in self.image_paths:
            temp_image_paths.append((image_path, -1))
        self.image_paths = temp_image_paths


if __name__ == '__main__':
    dataset = CINIC10L(root='D:/Datasets/CINIC10L')
    print("s")
