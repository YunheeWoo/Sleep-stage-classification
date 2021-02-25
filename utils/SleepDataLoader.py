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

    def __init__(self, csv_file, root_dir, signals=None, inv=False, color=None, shuffle=False, transform=None):
        self.samples = []
        self.root_dir = root_dir
        self.signals = signals
        self.color = color
        self.transform = transform
        self.inv = inv
        self.shuffle = shuffle

        if self.signals == None:
            self.__load_samples__(csv_file)
        else:
            self.__make_samples__(csv_file)

    def __load_samples__(self, csv_file):
        dirs = csv2list(csv_file)

        for d in dirs:
            imgs = os.listdir(self.root_dir / d[0])

            # Add image name to list
            for img in imgs:
                self.samples.append(d[0] + '/' + img)

    def __make_samples__(self, csv_file):
        # Read file list from csv file (each patient)
        dirs = csv2list(csv_file)

        # From patient read image list it has.(Because sequence is different from each patients)
        for d in dirs:
            imgs = os.listdir(self.root_dir / self.signals[0] / d[0])

            # Add image name to list
            for img in imgs:
                self.samples.append(d[0] + '/' + img)

    def __loader__(self, path):
        with open(path, 'rb') as f:
            # Open image
            img = Image.open(f)

            # Convert image to grayscale
            if self.color == "L" or self.color == "RGB":
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = None
        target = None
        if self.signals == None:
            sample = self.__load_img__(idx)
            target = int(self.samples[idx][-5])
            
        else:
            sample = self.__get_img__(idx)
            target = int(self.samples[idx][-5])

        #########################################################
        
        #eeg = np.zeros((224,32*4))
        #eeg.fill(0)
        #eog = np.zeros((224,32*2))
        #eog.fill(300)
        #emg = np.zeros((224,32*1))
        #emg.fill(300)

        #eeg = np.zeros((224,28*4))
        #eeg.fill(10)
        #eog = np.zeros((224,28*2))
        #eog.fill(20)
        #emg = np.zeros((224,28*1))
        #emg.fill(500)
        #ecg = np.zeros((224,28*1))
        #ecg.fill(40)

        eeg = np.zeros((224,69))
        eeg.fill(0)
        eog = np.zeros((224,34))
        eog.fill(10)
        emg = np.zeros((224,17))
        emg.fill(20)
        ecg = np.zeros((224,17))
        ecg.fill(30)
        flow = np.zeros((224,52))
        flow.fill(40)
        chest = np.zeros((224,18))
        chest.fill(50)
        abdomen = np.zeros((224,17))
        abdomen.fill(60)

        test = np.concatenate((eeg, eog), axis=1)
        test = np.concatenate((test, emg), axis=1)
        test = np.concatenate((test, ecg), axis=1)
        test = np.concatenate((test, flow), axis=1)
        test = np.concatenate((test, chest), axis=1)
        test = np.concatenate((test, abdomen), axis=1)
        #print(np.max(sample))
        #print(type(np.max(sample)))
        test = test.T
        sample += test

        #sample = np.where(sample > 255., 0., sample)

        #sample = sample.astype(np.uint8)

        #print(np.max(sample))
        #print(type(np.max(sample)))

        #sample = Image.fromarray(sample.astype('uint8'), 'L')
        # 1d t0 3d
        sample = sample[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        
        #########################################################
        if self.transform is not None:
                sample = self.transform(sample)

        return sample, target

"""
dataset = SleepDataset("/home/eslab/wyh/test_full.csv", Path("/home/eslab/wyh/data/img/resize/2000x100-224x32/t-02/mean-std-discard/"), ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "EMG"], color='L', inv=True)

#print(len(dataset))
cnt = 0

clss = 1

#temp = [300, 900, 1800, 3600, 4800]

temp = range(200)

for d in dataset:
    if d[1] == clss:
        cnt += 1
        if cnt in temp:
            #d[0].save(str(clss)+"_"+str(cnt)+".png")
            PIL_image = Image.fromarray(d[0].astype('uint8'), 'L')
            PIL_image.save(str(clss)+"_"+str(cnt).zfill(6)+".png")
            
            if cnt == temp[-1]:
                break

"""

#dataset =  SleepDataset("/home/eslab/wyh/test_full.csv", Path("/home/eslab/wyh/data/img/resize/1920x1080-224x224/t-02/mean-std-cut/"), color='L')

#Image.fromarray(np.uint8(dataset[0][0].save("test.png"))).convert('RGB').save("test.png")
#PIL_image = Image.fromarray(dataset[0][0].astype('uint8'), 'L')
#PIL_image.save("test11.png")
#dataset[0][0].save("test11.png")
#print(dataset[0][0].mode)

"""
dataset = SleepDataset("/home/eslab/wyh/train_full.csv", Path("/home/eslab/wyh/data/img/resize/2000x100-224x32/t-02/mean-std-discard/"), ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "EMG", "ECG"], inv=True, color="L", #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.Resize([224,224]),
                                    #transforms.RandomHorizontalFlip(),
                                    #transforms.RandomVerticalFlip(),
                                    #transforms.ToTensor(), 
                                    #transforms.Normalize(mean=[0.0044], std=[0.0396]),
                                    #transforms.Normalize(mean=[0.5], std=[0.5]),
                            ]))

dataset[0][0].save("test.png")
"""