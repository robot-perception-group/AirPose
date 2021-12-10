"""
This file contains the definition of different heterogeneous datasets used for training
"""
import torch
import numpy as np

from .aerialpeople import aerialpeople_crop
from .h36m import h36m_crop_train

class MixedDataset(torch.utils.data.Dataset):

    def __init__(self):
        self.dataset_list = ['h36m', 'aerialpeople']
        self.dataset_dict = {'h36m': 0, 'aerialpeople': 1}
        self.datasets = [h36m_crop_train(), aerialpeople_crop()]
        total_length = sum([len(ds) for ds in self.datasets])
        self.length = max([len(ds) for ds in self.datasets])
        """
        Data distribution inside each batch:
        50% H36M - 50% aerialpeople
        """
        self.partition = np.array([0.5,1.])

    def __getitem__(self, index):
        p = np.random.rand()
        for i in range(2):
            if p <= self.partition[i]:
                # temporarily aerialdataset
                return self.datasets[0][index % len(self.datasets[0])]

    def __len__(self):
        return self.length
