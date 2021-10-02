from datasets import bc_idc, cifar10, cifar100, cinic10, yahoo, tiny_image_net, criteo, breast_cancer_wisconsin
import torchvision.datasets as datasets


def get_dataset_by_name(dataset_name):
    dict_dataset = {
        'BC_IDC': bc_idc.IdcDataset,
        'CIFAR10': datasets.CIFAR10,
        'CIFAR100': datasets.CIFAR100,
        'CINIC10L': cinic10.CINIC10L,
        'Yahoo': yahoo.YahooSetup(),
        'TinyImageNet': tiny_image_net.TinyImageNet,
        'Criteo': criteo.Criteo,
        'BCW': breast_cancer_wisconsin.BcwDataset
    }
    dataset = dict_dataset[dataset_name]
    return dataset


def get_datasets_for_ssl(dataset_name, file_path, n_labeled, party_num=None):
    dataset_setup = get_dataset_setup_by_name(dataset_name)
    train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset = \
        dataset_setup.set_datasets_for_ssl(file_path, n_labeled, party_num)
    return train_labeled_set, train_unlabeled_set, test_set, train_complete_dataset


def get_dataset_setup_by_name(dataset_name):
    dict_dataset_setup = {
        'BC_IDC': bc_idc.IdcSetup(),
        'CIFAR10': cifar10.Cifar10Setup(),
        'CIFAR100': cifar100.Cifar100Setup(),
        'CINIC10L': cinic10.Cinic10LSetup(),
        'Yahoo': yahoo.YahooSetup(),
        'TinyImageNet': tiny_image_net.TinyImageNetSetup(),
        'Criteo': criteo.CriteoSetup(),
        'BCW':breast_cancer_wisconsin.BcwSetup()
    }
    dataset_setup = dict_dataset_setup[dataset_name]
    return dataset_setup
