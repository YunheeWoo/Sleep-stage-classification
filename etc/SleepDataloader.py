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
from PIL import Image
from utils import *

class SleepDataset(Dataset):

    def __init__(self, csv_file, root_dir, signals, color=None, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.signals = signals
        self.color = color
        self.transform = transform

        self.__make_samples__(csv_file)

    def __loader__(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if not self.color == None
                img = img.convert('L')
            return img

    def __makeimg__(self, idx):
        img_size = len(self.signals)
        dst = None

        img = __loader__(self.root_dir / signals[0] / self.samples[idx])
        if not self.color == None:
            dst = Image.new('L', (img.width, img.height * img_size))
        else:
            dst = Image.new('RGB', (img.width, img.height * img_size))
        

    def __make_samples__(self, csv_file):
        dirs = csv2list(csv_file)
        for d in dirs:
            imgs = os.listdir(self.root_dir / signals[0] / d)
            for img in imgs:
                self.samples.append(img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path= self.samples[index]
        target = int(path[-5])
        sample = self.__loader__(path)
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target