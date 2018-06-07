import os
import numpy as np
from torch.utils import data
from utils.data_os import pickle_load

# get data feature
def get_data_feature(data, data_feature_df):
    data_feature = []
    data_feature[0] = np.array(data_feature_df[data_feature_df.photo_id==data[0]]).tolist()
    for i in range(len(data[1])):
        data_feature[1][i] = np.array(data_feature_df[data_feature_df.photo_id==data[1][i]]).tolist()
    data_feature[-1] = np.array(data_feature_df[data_feature_df.photo_id==data[-1]]).tolist()
    return data_feature


# the dataset used for train and validation
class KuaishouDataset(data.Dataset):
    '''
    Args:
        root (string): Root directory of dataset where ``processed/training.pkl``
            and  ``processed/test.pkl`` exist.
        train (bool, optional): If True, creates dataset from ``training.pkl``,
            otherwise from ``test.pkl``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    '''

    data_type = 'train'
    file = 'train_dataset.pkl'

    classes = ['0 - no click', '1 - click']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}

    @property
    def targets(self):
        if self.train:
            return self.train_labels
        else:
            return self.val_labels

    def __init__(self, root, train = True, transform=None, target_transform = None):
        # TODO
        # 1. Initialize file path or list of file names.

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train # training set or test set

        self.data_path = os.path.join(self.root, self.data_type, self.file)
        # exists check
        if not self._check_exists(self):
            raise RuntimeError('Dataset not found.' +
                               ' You should check it...')

        data_set = pickle_load(self.data_path)
        self.train_photo_feature_df = data_set[2]
        if self.train:
            train_set = data_set[0]
            self.train_data = [x[:3] for x in train_set]
            self.train_labels = [x[3] for x in train_set]
        else:
            val_set = data_set[1]
            self.val_data = [x[:3] for x in val_set]
            self.val_labels = [x[3] for x in val_set]


    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data

        if self.train:
            data = self.train_data[index]
            data_feature = get_data_feature(data, self.train_photo_feature_df)
            target = self.train_labels[index]
        else:
            data = self.val_data[index]
            data_feature = get_data_feature(data, self.train_photo_feature_df)
            target = self.val_labels[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target, data_feature

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)

    def _check_exists(self):
        return os.path.exists(self.data_path)

    # data info
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str




# the dataset used for test
class KuaishouDetection(data.Dataset):
    '''
    Args:
        root (string): Root directory of dataset where ``processed/training.pkl``
            and  ``processed/test.pkl`` exist.
        train (bool, optional): If True, creates dataset from ``training.pkl``,
            otherwise from ``test.pkl``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """
    '''

    data_type = 'test'
    file = 'test_dataset.pkl'

    classes = ['0 - no click', '1 - click']
    class_to_idx = {_class: i for i, _class in enumerate(classes)}


    def __init__(self, root, transform=None):
        # TODO
        # 1. Initialize file path or list of file names.

        self.root = os.path.expanduser(root)
        self.transform = transform

        self.data_path = os.path.join(self.root, self.data_type, self.file)
        # exists check
        if not self._check_exists(self):
            raise RuntimeError('Dataset not found.' +
                               ' You should check it...')

        data_set = pickle_load(self.data_path)
        self.test_data = data_set[0]
        self.test_photo_feature_df = data_set[1]

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        # 这里需要注意的是，第一步：read one data，是一个data

        data = self.test_data[index]
        data_feature = get_data_feature(data, self.test_photo_feature_df)

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.test_data)

    def _check_exists(self):
        return os.path.exists(self.data_path)

    # data info
    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'test'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
