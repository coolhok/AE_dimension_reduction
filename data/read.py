import numpy as np
from torch.utils.data import Dataset, DataLoader


class feature_dset(Dataset):
    def __init__(self,
                 feature_path,
                 label_path):
        self.featute = np.loadtxt(feature_path, delimiter=",")
        self.label_list = np.loadtxt(label_path, delimiter=",")

    def __getitem__(self, index):
        featute = self.featute[index]
        label = self.label_list[index]
        return featute, label

    def __len__(self):
        return self.label_list.shape[0]
