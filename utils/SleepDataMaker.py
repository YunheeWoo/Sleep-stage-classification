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

class SleepDataMaker(Dataset):

    def __init__(self, csv_file, root_dir, signals=None, transform=None, dim=1, toimg=False, classes=5):
        self.samples = []
        self.root_dir = root_dir
        self.transform = transform
        self.dim = dim
        self.toimg = toimg
        self.color ='L'
        self.signals = signals
        self.classes = classes


        self.__load_fnames__(csv_file)

    def __load_fnames__(self, csv_file):
        patients = csv2list(csv_file)

        for patient in patients:

            # Train only normal & mild
            #if not ((int(d[0][-2]) == 0)or(int(d[0][-2]) == 1)):
            #    continue

            # Train only moderate
            #if not int(d[0][-2]) == 2:
            #    continue

            # Train only severe
            #if not int(d[0][-2]) == 3:
            #    continue


            imgs = os.listdir(self.root_dir / self.signals[0] / patient[0])

            # Add image name to list
            for img in imgs:
                self.samples.append(patient[0] + '/' + img)

    def __load_img__(self, idx):

        dst = Image.new('L', (448, 224))

        for s_idx in range(len(self.signals)):
            with open(self.root_dir / self.signals[s_idx] / self.samples[idx], 'rb') as f:
                img = Image.open(f)

                dst.paste(img, (0, 32*s_idx))

        return dst

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = None
        target = None
        
        sample = self.__load_img__(idx)
        target = int(self.samples[idx][-5])

        if self.classes == 5:
            target = int(self.samples[idx][-5])
        elif self.classes == 10:
            if len(self.samples[idx]) < 20:
                target = int(self.samples[idx][-5]) + 5
            elif len(self.samples[idx]) > 20:
                target = int(self.samples[idx][-5])
        

        #########################################################

        width, _ = sample.size

        img_width = width
        """
        eeg1 = np.zeros((img_width,17))
        eeg1.fill(0)
        eeg2 = np.zeros((img_width,17))
        eeg2.fill(0)
        eeg3 = np.zeros((img_width,17))
        eeg3.fill(0)
        eeg4 = np.zeros((img_width,18))
        eeg4.fill(0)
        eog1 = np.zeros((img_width,17))
        eog1.fill(0)
        eog2 = np.zeros((img_width,17))
        eog2.fill(0)
        emg = np.zeros((img_width,17))
        emg.fill(0)
        ecg = np.zeros((img_width,17))
        ecg.fill(400)
        flow = np.zeros((img_width,52))
        flow.fill(0)
        chest = np.zeros((img_width,18))
        chest.fill(400)
        abdomen = np.zeros((img_width,17))
        abdomen.fill(0)

        test = np.concatenate((eeg1, eeg2), axis=1)
        test = np.concatenate((test, eeg3), axis=1)
        test = np.concatenate((test, eeg4), axis=1)
        test = np.concatenate((test, eog1), axis=1)
        test = np.concatenate((test, eog2), axis=1)
        test = np.concatenate((test, emg), axis=1)
        test = np.concatenate((test, ecg), axis=1)
        test = np.concatenate((test, flow), axis=1)
        test = np.concatenate((test, chest), axis=1)
        test = np.concatenate((test, abdomen), axis=1)
        test = test.T
        sample += test
        """
        """
        eeg1 = np.zeros((img_width,11))
        eeg1.fill(0)
        eeg2 = np.zeros((img_width,12))
        eeg2.fill(0)
        eeg3 = np.zeros((img_width,12))
        eeg3.fill(0)
        eeg4 = np.zeros((img_width,12))
        eeg4.fill(0)
        eeg5 = np.zeros((img_width,12))
        eeg5.fill(0)
        eeg6 = np.zeros((img_width,12))
        eeg6.fill(400)
        eog1 = np.zeros((img_width,11))
        eog1.fill(0)
        eog2 = np.zeros((img_width,12))
        eog2.fill(400)
        emg = np.zeros((img_width,12))
        emg.fill(0)
        ecg = np.zeros((img_width,12))
        ecg.fill(400)
        flow = np.zeros((img_width,35))
        flow.fill(0)
        chest = np.zeros((img_width,12))
        chest.fill(0)
        abdomen = np.zeros((img_width,12))
        abdomen.fill(0)
        sat1 = np.zeros((img_width,23))
        sat1.fill(0)
        sat2 = np.zeros((img_width,24))
        sat2.fill(400)

        test = np.concatenate((eeg1, eeg2), axis=1)
        test = np.concatenate((test, eeg3), axis=1)
        test = np.concatenate((test, eeg4), axis=1)
        test = np.concatenate((test, eeg5), axis=1)
        test = np.concatenate((test, eeg6), axis=1)
        test = np.concatenate((test, eog1), axis=1)
        test = np.concatenate((test, eog2), axis=1)
        test = np.concatenate((test, emg), axis=1)
        test = np.concatenate((test, ecg), axis=1)
        test = np.concatenate((test, flow), axis=1)
        test = np.concatenate((test, chest), axis=1)
        test = np.concatenate((test, abdomen), axis=1)
        test = np.concatenate((test, sat1), axis=1)
        test = np.concatenate((test, sat2), axis=1)
        test = test.T
        sample += test
        """
        """
        eeg = np.zeros((img_width,71))
        eeg.fill(0)
        eog = np.zeros((img_width,23))
        eog.fill(10)
        emg = np.zeros((img_width,12))
        emg.fill(20)
        ecg = np.zeros((img_width,12))
        ecg.fill(30)
        flow = np.zeros((img_width,35))
        flow.fill(40)
        chest = np.zeros((img_width,12))
        chest.fill(50)
        abdomen = np.zeros((img_width,12))
        abdomen.fill(60)
        sat1 = np.zeros((img_width,23))
        sat1.fill(70)
        sat2 = np.zeros((img_width,24))
        sat2.fill(70)
        
        
        test = np.concatenate((eeg, eog), axis=1)
        test = np.concatenate((test, emg), axis=1)
        test = np.concatenate((test, ecg), axis=1)
        test = np.concatenate((test, flow), axis=1)
        test = np.concatenate((test, chest), axis=1)
        test = np.concatenate((test, abdomen), axis=1)
        test = np.concatenate((test, sat1), axis=1)
        test = np.concatenate((test, sat2), axis=1)
        test = test.T
        sample += test
        """

        #sample = np.where(sample > 399., 0., sample)

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
