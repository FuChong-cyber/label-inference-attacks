class DatasetSetup:
    def __init__(self):
        self.num_classes = None
        self.size_bottom_out = None

    def set_datasets_for_ssl(self, file_path, n_labeled, party_num):
        pass

    def get_transforms(self):
        pass

    def get_normalize_transform(self):
        pass

    def get_transformed_dataset(self, file_path, party_num, train):
        pass

    def clip_one_party_data(self, x, half):
        """
        :param x:
        :param half: how many features the adversary has. For the IDC dataset, it means how many pics
        the adversary has.
        :return:
        """
        pass
