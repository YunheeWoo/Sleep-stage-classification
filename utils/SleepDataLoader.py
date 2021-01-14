import os
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
from .util import *
from pathlib import Path
import random

class SleepDataset(Dataset):

    def __init__(self, csv_file, root_dir, signals, inv=True, color=None, shuffle=False, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.signals = signals
        self.color = color
        self.transform = transform
        self.inv = inv
        self.shuffle = shuffle

        self.__make_samples__(csv_file)

    def __loader__(self, path):
        with open(path, 'rb') as f:
            # Open image
            img = Image.open(f)

            # Convert image to grayscale
            if self.color == "L" or self.color == "RGB":
                img = img.convert('L')

            # If image has 'A' channel, remove it
            if img.mode == 'RGBA':
                r,g,b,a = img.split()
                img = Image.merge('RGB', (r,g,b))

            # If invert is True, invert the image(default is True)
            if self.inv:
                img = PIL.ImageOps.invert(img)

            return img

    def __get_img__(self, idx):
        img_size = len(self.signals)
        dst = None

        signal_shuffle = self.signals.copy()

        if self.shuffle:
            random.shuffle(signal_shuffle)

        # Loader the first signal
        img = self.__loader__(self.root_dir / signal_shuffle[0] / self.samples[idx])

        # Make the canvas
        if self.color == "L":
            dst = Image.new('L', (img.width, img.height * img_size))
        elif self.color == "RGB":
            dst = Image.new('L', (img.width, img.height))
        else:
            dst = Image.new('RGB', (img.width, img.height * img_size))

        # Paste the images to canvas
        if not self.color == "RGB":
            dst.paste(img, (0, 0))

            for signal in signal_shuffle[1:]:
                img = self.__loader__(self.root_dir / signal / self.samples[idx])
                dst.paste(img, (0, img.height*signal_shuffle.index(signal)))
        
        else:
            img2 = self.__loader__(self.root_dir / self.signals[1] / self.samples[idx])
            img3 = self.__loader__(self.root_dir / self.signals[2] / self.samples[idx])

            dst = Image.merge('RGB', (img,img2,img3))

        return dst
        
    def __make_samples__(self, csv_file):
        # Read file list from csv file (each patient)
        dirs = csv2list(csv_file)

        # From patient read image list it has.(Because sequence is different from each patients)
        for d in dirs:
            imgs = os.listdir(self.root_dir / self.signals[0] / d[0])

            # Add image name to list
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


#dataset = SleepDataset("/home/eslab/wyh/test.csv", Path("/home/eslab/wyh/data/img/butter/2000x100/t-02/mean-std-discard"), ["C3-M2", "E1-M2", "E2-M1"])

#print(len(dataset))
