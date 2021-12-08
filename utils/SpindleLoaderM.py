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
import albumentations as A

class SpinleLoaderM(Dataset):

    def __init__(self, csv_file, root_dir, signals=None, shuffle=False, transform=None, dim=1, toimg=False, inv=True, rand=False):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        self.inv = inv
        self.shuffle = shuffle
        self.dim = dim
        self.toimg = toimg
        self.color ='L'
        self.color = None
        self.rand = rand
        self.signals = signals

        self.__load_samples__(csv_file)

    def __load_samples__(self, csv_file):
        self.samples = csv2list(csv_file)
        self.samples.sort()

    def __load_img__(self, idx, signal):
        path = self.root_dir / signal /self.samples[idx][0]
        img = None
        
        #with open(path, 'rb') as f:
            # Open image
        img = Image.open(path)

        
        # Convert image to grayscale
        if self.color == "L":
            img = img.convert('L')

        if self.color == None:
            img = img.convert('RGB')
        
        # If image has 'A' channel, remove it
        if img.mode == 'RGBA':
            r,g,b,a = img.split()
            img = Image.merge('RGB', (r,g,b))

        # If invert is True, invert the image(default is True)
        if self.inv:
            img = PIL.ImageOps.invert(img)

        return img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        target = int(self.samples[idx][0][-6])
        sample = []

        for signal in self.signals:
            sample.append(self.__load_img__(idx,signal))

        if self.transform is not None:
            for idx, s in enumerate(sample):
                sample[idx] = self.transform(s)

        return sample[0], sample[1], sample[2], sample[3], sample[4], sample[5], target
