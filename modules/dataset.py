"""
Dataset module.
"""

__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve'
__email__ = "saul.alonso.monsalve@cern.ch"

import torch
from torch.utils.data import Dataset
from glob import glob
from modules.constants import *


class FittingDataset(Dataset):

    def __init__(self, root, to_tensor=True, normalize=True, shuffle=False):
        self.root = root
        self.data_files = self.processed_file_names
        if shuffle:
            random.shuffle(self.data_files)
        self.total_events = self.__len__()
        self.normalize = normalize
        self.to_tensor = to_tensor

    @property
    def processed_dir(self):
        return f'{self.root}' + '/*'

    @property
    def processed_file_names(self):
        return sorted(glob(f'{self.processed_dir}/*.pt'))

    def __len__(self):
        return len(self.data_files)

    @staticmethod
    def apply_norm(X):
        for i in range(3):
            X[:, i] -= DETECTOR_RANGES[i][0]
        X[:, :3] /= (DETECTOR_RANGES[0][1] - DETECTOR_RANGES[0][0])
        if X.shape[1] > 3:
            X[:, 3] = (X[:, 3] - CHARGE_RANGE[0]) / \
                      (CHARGE_RANGE[1] - CHARGE_RANGE[0])

    def __getitem__(self, idx):
        # load particle
        data = torch.load(self.data_files[idx])
        x = data['reco_hits']
        y = data['true_nodes']

        # normalise
        x[x[:, 3] > 500, 3] = 500  # trim energy
        if self.normalize:
            self.apply_norm(x)
            self.apply_norm(y)

        # convert to tensors
        if self.to_tensor:
            x = torch.tensor(x).float()
            y = torch.tensor(y).float()

        del data
        return x, y
