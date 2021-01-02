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
import PIL.ImageOps 
from util import *
from pathlib import Path

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
            if not self.color == None:
                img = img.convert('L')
            img = PIL.ImageOps.invert(img)
            return img

    def __get_img__(self, idx):
        img_size = len(self.signals)
        dst = None

        img = self.__loader__(self.root_dir / self.signals[0] / self.samples[idx])

        if not self.color == None:
            dst = Image.new('L', (img.width, img.height * img_size))
        else:
            dst = Image.new('RGB', (img.width, img.height * img_size))
        
        dst.paste(img, (0, 0))

        for signal in self.signals[1:]:
            img = self.__loader__(self.root_dir / signal / self.samples[idx])
            dst.paste(img, (0, img.height*self.signals.index(signal)))

        return dst
        
    def __make_samples__(self, csv_file):
        dirs = csv2list(csv_file)
        for d in dirs:
            imgs = os.listdir(self.root_dir / self.signals[0] / d[0])
            for img in imgs:
                self.samples.append(d[0] + '/' + img)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.__get_img__(idx)
        target = int(self.samples[idx][-5])
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


#dataset = SleepDataset("/home/eslab/wyh/data/test.csv", Path("/home/eslab/wyh/data/img/2000x100/min-max-cut"), ["C3-M2", "E1-M2", "E2-M1"], color="L")

#print(dataset[0])