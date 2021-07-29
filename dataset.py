import os
import config
import math
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

class HoroscopeDataset(Dataset):
    def __init__(self, filename, features, labels, segments):
        self.h5f = h5py.File(filename, 'r')
        self.data = self.h5f[features]
        self.labels = self.h5f[labels]
        self.segments = self.h5f[segments]
        self.examples = [(self.data, self.labels, self.segments)]
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, item):
        texts = torch.tensor(self.data[item], dtype=torch.long)
        labels = torch.tensor(self.labels[item], dtype=torch.long)
        segments = torch.tensor(self.segments[item], dtype=torch.long)
        return texts, labels, segments