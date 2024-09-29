import torch
from torch.utils.data import Dataset

class Preprocessor:
    """
        Class which converts the data to a new representation and back.
        It standardizes the varying bits and applies logarithm to the FER.
    """
    def fit(self, x, y, indices, bit_means):
        # Samples x Varying_bits
        self._x_mean = x.reshape(-1).mean()
        self._x_std = x.reshape(-1).std()
        self._y_mean_log = torch.log(y).mean() # can be any value depending on the dataset
        self._y_std_log = torch.log(y).std()
        self._indices = indices
        self._bit_means = bit_means # Used as non-varying bits can be 1 or 0
        self._orig_bits = len(x)

    def transform(self, x, y):
        x_ = (x[:, self._indices] - self._x_mean) / self._x_std
        y_ = (torch.log(y) - self._y_mean_log) / self._y_std_log
        return x_, y_

    def inverse_transform(self, x=None, y=None):
        if x is not None:
            x = self._bit_means.repeat(x.shape[0], 1)
            x[:, self._indices] = x[:, self._indices]*self._x_std + self._x_mean
        if y is not None:
            y = torch.exp((y * self._y_std_log) + self._y_mean_log)
        return x, y
    

class SimpleDataset(Dataset):
    def __init__(self, all_data, all_targets, device):
        super(Dataset, self).__init__()
        self.data = all_data.to(device) # features(bits)
        self.targets = all_targets.to(device) # labels (FER)

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]
        return img, target

    def __len__(self) -> int:
        return self.data.shape[0]