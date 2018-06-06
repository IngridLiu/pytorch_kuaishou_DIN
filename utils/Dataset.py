
from torch.utils import data
from utils.data_os import pickle_load

class KuaishouDataset(data.Dataset):
    def __init__(self, root, train = True, transform=None, target_transform = None):
        # TODO
        # 1. Initialize file path or list of file names.

        self.root = root
        self.trandform = transform
        self.target_transform = target_transform
        self.train = train # training set or test set

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You should check it...')

        data_set = pickle_load(root + "train/dataset.pkl")
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
            target = self.train_labels[index]
        else:
            data = self.val_data[index]
            target = self.val_labels[index]

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        if self.train:
            return len(self.train_data)
        else:
            return len(self.val_data)
