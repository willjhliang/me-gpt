
import torch
from torch.utils.data import Dataset as TorchDataset

from utils import encode
import config

class Dataset(TorchDataset):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            text = f.read()
        self.data = torch.tensor(encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - config.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+config.window_size]
        y = self.data[idx+1:idx+config.window_size+1]
        return x, y
