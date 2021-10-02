from datasets.dataset_setup import DatasetSetup
from models import read_data_text


class YahooSetup(DatasetSetup):
    def __init__(self):
        super().__init__()
        self.num_classes = 10
        self.size_bottom_out = 10

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num):
        train_labeled_dataset, train_unlabeled_dataset, val_dataset, test_dataset, n_labels = \
            read_data_text.get_data(file_path, int(n_labeled / 10))
        train_complete_labeled_dataset, _, _, _, _ = \
            read_data_text.get_data(file_path, 5000)
        print("#Labeled:", len(train_labeled_dataset), "#Unlabeled:", len(train_unlabeled_dataset))
        return train_labeled_dataset, train_unlabeled_dataset, test_dataset, train_complete_labeled_dataset

    def get_transformed_dataset(self, file_path, party_num=None, train=True):
        if train:
            train_complete_labeled_dataset, _, _, _, _ = \
                read_data_text.get_data(file_path, 5000)
            return train_complete_labeled_dataset
        else:
            _, _, _, test_dataset, _ = \
                read_data_text.get_data(file_path, 10)
            return test_dataset


if __name__ == '__main__':
    dataset_setup = YahooSetup()
    train_dataset = dataset_setup.get_transformed_dataset(file_path='D:/Datasets/yahoo_answers_csv/',train=True)
    print("s")
