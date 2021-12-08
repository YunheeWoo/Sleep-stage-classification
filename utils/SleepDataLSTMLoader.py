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

class SleepDataLSTMset(Dataset):

    def __init__(self, csv_file, root_dir, signals=None, inv=False, color=None, shuffle=False, transform=None, train=False):
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

        if self.signals == None:
            self.__load_samples__(csv_file)
        else:
            self.__make_samples__(csv_file)

    def __load_samples__(self, csv_file):
        dirs = csv2list(csv_file)

        skip_list = ['A2019-NX-01-1064_3_', 'A2019-NX-01-0867_3_', 'A2019-NX-01-0581_3_', 'A2020-NX-01-0337_3_', '055_3_', 'A2019-NX-01-0154_3_', 'A2020-NX-01-0753_3_', 'A2019-NX-01-0437_2_', 'A2019-NX-01-1606_3_', 'A2019-NX-01-1331_3_', 'A2019-NX-01-1195_3_', '334_0_', 'A2019-NX-01-1128_3_', 'A2019-NX-01-1133_2_', 'A2019-NX-01-0299_3_', '252_1_', 'A2019-NX-01-0417_3_']
        #skip_list = []
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

        for d in dirs:
            if d[0] in skip_list:
                continue

            imgs = os.listdir(self.root_dir / d[0])

            imgs.sort()

            # Add image name to list

            for img_idx, img in enumerate(imgs):
                if img == imgs[0] or img == imgs[1] or img == imgs[-1] or img == imgs[-2]:
                    continue

                self.samples.append([d[0] + '/' + imgs[img_idx-2], d[0] + '/' + imgs[img_idx-1], d[0] + '/' + imgs[img_idx], d[0] + '/' + imgs[img_idx+1], d[0] + '/' + imgs[img_idx+2]])
                if self.train == True:
                    if int(imgs[img_idx][-5]) == 3:
                        self.samples.append([d[0] + '/' + imgs[img_idx-2], d[0] + '/' + imgs[img_idx-1], d[0] + '/' + imgs[img_idx], d[0] + '/' + imgs[img_idx+1], d[0] + '/' + imgs[img_idx+2]])
                self.cnt_sample[int(d[0][-2])][int(imgs[img_idx][-5])] += 1

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

    def __prefix__(self, sample):
        #img_width = sample.shape(1)
        width, _ = sample.size

        img_width = width

        """
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
        """
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
        """
        """
        # new_14
        eeg = np.zeros((img_width,78))
        eeg.fill(0)
        eog = np.zeros((img_width,26))
        eog.fill(10)
        emg = np.zeros((img_width,14))
        emg.fill(20)
        ecg = np.zeros((img_width,13))
        ecg.fill(20)
        flow = np.zeros((img_width,40))
        flow.fill(40)
        chest = np.zeros((img_width,13))
        chest.fill(50)
        abdomen = np.zeros((img_width,13))
        abdomen.fill(60)
        sat = np.zeros((img_width,27))
        sat.fill(70)
        """
        # new_13
        eeg = np.zeros((img_width,84))
        eeg.fill(0)
        eog = np.zeros((img_width,28))
        eog.fill(10)
        emg = np.zeros((img_width,14))
        emg.fill(20)
        flow = np.zeros((img_width,42))
        flow.fill(40)
        chest = np.zeros((img_width,14))
        chest.fill(50)
        abdomen = np.zeros((img_width,14))
        abdomen.fill(60)
        sat = np.zeros((img_width,28))
        sat.fill(50)
        
        """
        # new_11
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
        """
        """
        # new_09 
        eeg = np.zeros((img_width,55))
        eeg.fill(0)
        eog = np.zeros((img_width,19))
        eog.fill(10)
        emg = np.zeros((img_width,18))
        emg.fill(20)
        flow = np.zeros((img_width,57))
        flow.fill(40)
        chest = np.zeros((img_width,18))
        chest.fill(50)
        abdomen = np.zeros((img_width,19))
        abdomen.fill(60)
        sat = np.zeros((img_width,38))
        sat.fill(70)
        """
        
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

        sample = np.where(sample > 255., 255., sample)
        
        #sample = Image.fromarray(sample.astype('uint8'), 'L')

        return sample

    def __getitem__(self, idx):
        #sample = None
        target = None
        sample = []
        if self.signals == None:
            """
            matching = []
            setoffs = [-2, -1, 0, 1, 2]

            for s in setoffs:
                target = self.samples[idx][:-10] + str(int(self.samples[idx][-10:-6])+s).zfill(4)

                matching.append([s for s in self.all_samples if target in s][0])
            
            for m in matching:
                sample.append(self.__load_img_by_name__(m))
            """
            for s in self.samples[idx]:
                sample.append(self.__load_img_by_name__(s))
                target = int(self.samples[idx][2][-5])
            """
            ##########
            if len(self.samples[idx][2]) < 20:
                target = int(self.samples[idx][2][-5]) + 5
            elif len(self.samples[idx][2]) > 20:
                target = int(self.samples[idx][2][-5])
            ##########
            """
            
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