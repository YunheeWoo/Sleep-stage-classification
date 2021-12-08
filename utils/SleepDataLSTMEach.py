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

class SleepDataLSTMEach(Dataset):

    def __init__(self, root_dir, signals=None, inv=False, color=None, shuffle=False, transform=None, train=False):
        self.samples = []
        self.all_samples = []
        self.root_dir = root_dir
        self.signals = signals
        self.color = color
        self.transform = transform
        self.inv = inv
        self.shuffle = shuffle
        self.train = train
        #self.cnt_sample = [0,0,0,0,0]
        self.cnt_sample = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
        self.change_cnt = 0

        self.__load_samples__()

    def __load_samples__(self):

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

        imgs = os.listdir(self.root_dir)

        imgs.sort()

        # Add image name to list

        for img_idx, img in enumerate(imgs):
            if img == imgs[0] or img == imgs[1] or img == imgs[-1] or img == imgs[-2]:
                continue

            self.samples.append([self.root_dir / imgs[img_idx-2], self.root_dir / imgs[img_idx-1], self.root_dir / imgs[img_idx], self.root_dir / imgs[img_idx+1], self.root_dir / imgs[img_idx+2]])
            if int(imgs[img_idx][-5]) != int(imgs[img_idx-1][-5]):
                self.change_cnt += 1
            if self.train == True:
                if int(imgs[img_idx][-5]) == 3:
                    self.samples.append([self.root_dir / imgs[img_idx-2], self.root_dir / imgs[img_idx-1], self.root_dir / imgs[img_idx], self.root_dir / imgs[img_idx+1], self.root_dir / imgs[img_idx+2]])
            self.cnt_sample[int(str(self.root_dir)[-2])][int(imgs[img_idx][-5])] += 1

    def __load_img_by_name__(self, filename):

        path = self.root_dir / filename
        
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

    def __prefix__(self, sample):
        #img_width = sample.shape(1)
        width, _ = sample.size

        img_width = width

        eeg = np.zeros((img_width,81))
        eeg.fill(0)
        eog = np.zeros((img_width,40))
        eog.fill(10)
        emg = np.zeros((img_width,21))
        emg.fill(20)
        flow = np.zeros((img_width,20))
        flow.fill(40)
        chest = np.zeros((img_width,21))
        chest.fill(50)
        abdomen = np.zeros((img_width,20))
        abdomen.fill(60)
        sat = np.zeros((img_width,21))
        sat.fill(50)
        
        test = np.concatenate((eeg, eog), axis=1)
        test = np.concatenate((test, emg), axis=1)
        test = np.concatenate((test, flow), axis=1)
        test = np.concatenate((test, chest), axis=1)
        test = np.concatenate((test, abdomen), axis=1)
        test = np.concatenate((test, sat), axis=1)
        #print(np.max(sample))
        #print(type(np.max(sample)))
        test = test.T
        sample += test

        sample = np.where(sample > 255., 0., sample)

        #sample = Image.fromarray(sample.astype('uint8'), 'L')

        return sample

    def __getitem__(self, idx):
        #sample = None
        target = None
        sample = []
        if self.signals == None:
            for s in self.samples[idx]:
                sample.append(self.__load_img_by_name__(s))
                target = int(str(self.samples[idx][2])[-5])
        else:
            sample = self.__get_img__(idx)
            target = int(self.samples[idx][-5])

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

        for idx, s in enumerate(sample):
            sample[idx] = self.__prefix__(s)

        if self.transform is not None:
                for idx, s in enumerate(sample):
                    sample[idx] = self.transform(s)

        return sample[0], sample[1], sample[2], sample[3], sample[4], target

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

#cnt = [0,0,0,0,0]

#dataset =  SleepDataLSTMset("/home/eslab/wyh/test_downsampling.csv", Path("/home/eslab/wyh/data/img/resize/1920x1080-448x224/t-02/mean-std-cut/"), None, inv=False, color="L", train=False)

#print(dataset.cnt_sample)

"""
for d in dataset:
    cnt[int(d[-1])] += 1
#Image.fromarray(np.uint8(dataset[0][0].save("test.png"))).convert('RGB').save("test.png")
#PIL_image = Image.fromarray(dataset[0][0].astype('uint8'), 'L')
#PIL_image.save("test210316-1.png")
#dataset[0][0].save("test11.png")
#print(dataset[0])

print(cnt)
"""