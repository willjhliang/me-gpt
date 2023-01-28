
import torch
from torch.utils.data import Dataset as TorchDataset

import config

class Dataset(TorchDataset):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            text = f.read()
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.data = torch.tensor(self.encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - config.window_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+config.window_size]
        y = self.data[idx+1:idx+config.window_size+1]
        return x, y

    def encode(self, text):
        return [self.stoi[c] for c in text]
    
    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])