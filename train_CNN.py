import os, glob, math, argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class DatasetWindow(Dataset):
    def __init__(self, data_file, seq_len=168):
        self.seq_len = seq_len
        data = np.load(data_file)
        