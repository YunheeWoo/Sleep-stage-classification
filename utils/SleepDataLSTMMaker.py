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

class SleepDataLSTMMaker(Dataset):

    def __init__(self, csv_file, root_dir, signals, transform=None, classes=5):
        self.samples = []
        self.root_dir = root_dir
        self.signals = signals
        self.transform = transform
        self.classes = classes

        self.__load_samples__(csv_file)

    def __load_samples__(self, csv_file):
        patients = csv2list(csv_file)

        """
        ##########
        dirs_temp = csv2list(csv_file)
        dirs = []
        for d in dirs_temp:
            #if int(d[0][-2]) == 0 or int(d[0][-2]) == 1:
            if int(d[0][-2]) == 2:
            #if int(d[0][-2]) == 3:
                dirs.append(d)
        ##########
        """

        for patient in patients:
            imgs = os.listdir(self.root_dir / self.signals[0] / patient[0])
            imgs.sort()

            # Add image name to list

            for img_idx, img in enumerate(imgs):
                if img == imgs[0] or img == imgs[1] or img == imgs[-1] or img == imgs[-2]:
                    continue

                seq_data = []
                
                for seq_idx in range(-2, 3):
                    seq_data.append(patient[0] + '/' + imgs[img_idx+seq_idx])

                self.samples.append(seq_data)

    def __load_img_(self, fname):

        dst = Image.new('L', (448, 224))

        for s_idx in range(len(self.signals)):
            with open(self.root_dir / self.signals[s_idx] / fname, 'rb') as f:
                img = Image.open(f)

                dst.paste(img, (0, 32*s_idx))

        return dst

    def __len__(self):
        return len(self.samples)

    def __prefix__(self, sample):
        #img_width = sample.shape(1)
        width, _ = sample.size

        img_width = width

        #sample = Image.fromarray(sample.astype('uint8'), 'L')

        return sample

    def __getitem__(self, idx):
        target = None
        sample = []

        for s in self.samples[idx]:
            sample.append(self.__load_img_(s))
            #target = int(self.samples[idx][2][-5])

            if self.classes == 5:
                target = int(self.samples[idx][2][-5])
            elif self.classes == 10:
                if len(self.samples[idx]) < 20:
                    target = int(self.samples[idx][2][-5]) + 5
                elif len(self.samples[idx]) > 20:
                    target = int(self.samples[idx][2][-5])

        #########################################################

        #sample = np.where(sample > 255., 0., sample)

        #sample = sample.astype(np.uint8)

        #print(np.max(sample))
        #print(type(np.max(sample)))

        #sample = Image.fromarray(sample.astype('uint8'), 'L')

        # 1d t0 3d
        #sample = sample[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        #sample = Image.fromarray(sample.astype('uint8'), 'RGB')
        
        #########################################################

        #for idx, s in enumerate(sample):
        #    sample[idx] = self.__prefix__(s)

        if self.transform is not None:
                for idx, s in enumerate(sample):
                    sample[idx] = self.transform(s)

        return sample[0], sample[1], sample[2], sample[3], sample[4], target
