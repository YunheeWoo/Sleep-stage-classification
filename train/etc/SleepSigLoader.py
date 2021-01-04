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

class SleepSigLoader(Dataset):

    def __init__(self, csv_file, root_dir, signals, method,color=None, transform=None):
        self.samples = {}
        self.targets = np.array([])
        self.root_dir = root_dir
        self.signals = signals
        self.color = color
        self.transform = transform
        self.method = method

        self.normalization = ["min-max-cut", "mean-std-cut", "min-max-discard", "min-max-discard", "original", "min-max", "mean-std"]
        self.signal_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]

        self.src_path = Path("/home/eslab/wyh/data/npy/original")
        self.ann_path = Path("/home/eslab/wyh/data/annotations")

        self.__make_samples__(csv_file)

    def __figure_to_array__(self, fig):
        fig.canvas.draw()
        return np.array(fig.canvas.renderer._renderer)

    def __normalize__(self, data):
        std = np.std(data)
        mean = np.mean(data)
        # preprocessing
        if self.method == "min-max-cut" and self.method == "mean-std-cut":
            cut_value = 192*1e-06
            data = np.where(data < -cut_value, -cut_value, data)
            data = np.where(data > cut_value, cut_value, data)
            

        if self.method == "min-max-discard" and self.method == "mean-std-discard":
            m = 3.5
            idx1 = (data - np.mean(data)) > m * np.std(data)
            idx2 = (data - np.mean(data)) < -m * np.std(data)
            data[idx1] = m * std
            data[idx2] = -m * std

        data = np.reshape(data,(-1,6000))

        return data

    def __loader__(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            if not self.color == None:
                img = img.convert('L')
            #img = PIL.ImageOps.invert(img)

            if img.mode == 'RGBA':
                r,g,b,a = img.split()
                rgb_image = Image.merge('RGB', (r,g,b))
                img = PIL.ImageOps.invert(rgb_image)
            else:
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
        datas = np.array([])
        patients = csv2list(csv_file)
        for signal in self.signals:
            for p in patients:
                data = np.load(self.src_path / (p[0]+".npy"))
                datas = np.append(datas, self.__normalize__(data[self.signal_list.index(signal)]))

                if signal == self.signals[0]:
                    ann = np.load(self.ann_path / (p[0]+".npy"))
                    self.targets = np.append(self.targets, ann)

            self.samples[signal] = datas

        print(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.__get_img__(idx)
        target = int(self.samples[idx][-5])
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


#dataset = SleepSigLoader("/home/eslab/wyh/data/val.csv", Path("/home/eslab/wyh/data/img/fail/min-max-cut"), ["C3-M2", "E1-M2", "E2-M1"], method="min-max-cut", color="L")

#print(np.array(dataset[0][0]).shape)