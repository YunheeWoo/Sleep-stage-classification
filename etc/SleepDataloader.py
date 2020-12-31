import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
import csv
from utils import *

class SleepDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform

        self.__makedata__(csv_file)

    def __loader__(self, img_path):
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __makedata__(self, csv_file):
        dirs = csv2list(csv_file)
        for d in dirs:
            imgs = os.listdir(self.root_dir+'/'+d)
            for img in imgs:
                self.samples.append(self.root_dir+'/'+d+'/'+img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path= self.samples[index]
        target = int(path[-5])
        sample = self.__loader__(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target