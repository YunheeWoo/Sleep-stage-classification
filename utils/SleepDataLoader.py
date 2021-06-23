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

class SleepDataset(Dataset):

    def __init__(self, csv_file, root_dir, shuffle=False, transform=None, dim=1, toimg=False, inv=True):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        self.inv = inv
        self.shuffle = shuffle
        self.dim = dim
        self.toimg = toimg

        self.__load_samples__(csv_file)

    def __load_samples__(self, csv_file):
        dirs = csv2list(csv_file)

        for d in dirs:

            # Train only normal & mild
            #if not ((int(d[0][-2]) == 0)or(int(d[0][-2]) == 1)):
            #    continue

            # Train only moderate
            #if not int(d[0][-2]) == 2:
            #    continue

            # Train only severe
            #if not int(d[0][-2]) == 3:
            #    continue


            imgs = os.listdir(self.root_dir / d[0])

            # Add image name to list
            for img in imgs:
                self.samples.append(d[0] + '/' + img)

    def __load_img__(self, idx):

        path = self.root_dir / self.samples[idx]
        
        with open(path, 'rb') as f:
            # Open image
            img = Image.open(f)

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
        sample = None
        target = None
        
        sample = self.__load_img__(idx)
        target = int(self.samples[idx][-5])

        """
        if len(self.samples[idx]) < 20:
            target = int(self.samples[idx][-5]) + 5
        elif len(self.samples[idx]) > 20:
            target = int(self.samples[idx][-5])
        """

        #########################################################

        width, _ = sample.size

        img_width = width

        
        eeg = np.zeros((img_width,69))
        eeg.fill(0)
        eog = np.zeros((img_width,34))
        eog.fill(10)
        emg = np.zeros((img_width,17))
        emg.fill(20)
        ecg = np.zeros((img_width,17))
        ecg.fill(30)
        flow = np.zeros((img_width,52))
        flow.fill(40)
        chest = np.zeros((img_width,18))
        chest.fill(50)
        abdomen = np.zeros((img_width,17))
        abdomen.fill(60)

        """
        eeg = np.zeros((img_width,139))
        eeg.fill(0)
        eog = np.zeros((img_width,70))
        eog.fill(10)
        emg = np.zeros((img_width,36))
        emg.fill(20)
        ecg = np.zeros((img_width,36))
        ecg.fill(30)
        flow = np.zeros((img_width,104))
        flow.fill(40)
        chest = np.zeros((img_width,36))
        chest.fill(50)
        abdomen = np.zeros((img_width,35))
        abdomen.fill(60)
        """
        
        test = np.concatenate((eeg, eog), axis=1)
        test = np.concatenate((test, emg), axis=1)
        test = np.concatenate((test, ecg), axis=1)
        test = np.concatenate((test, flow), axis=1)
        test = np.concatenate((test, chest), axis=1)
        test = np.concatenate((test, abdomen), axis=1)
        test = test.T
        sample += test
        

        #sample = np.where(sample > 255., 0., sample)

        #sample = sample.astype(np.uint8)

        if self.dim == 3:
            # 1d t0 3d
            sample = sample[:, :, None] * np.ones(3, dtype=int)[None, None, :]
            if self.toimg == True:
                sample = Image.fromarray(sample.astype('uint8'), 'RGB')

        elif self.dim == 1:
            if self.toimg == True:
                sample = Image.fromarray(sample.astype('uint8'), 'L')

        #sample /= 255.

        #########################################################
        if self.transform is not None:
                sample = self.transform(sample)
                #sample = self.transform(image=sample)['image']

        return sample, target
