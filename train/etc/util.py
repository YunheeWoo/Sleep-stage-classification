import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.datasets import ImageFolder
from PIL import Image
import csv
from typing import Any, Callable, TypeVar, Generic, Sequence, List, Optional, Tuple

def csv2list(csv_file):
    f = open(csv_file, 'r')
    csvf = csv.reader(f)
    lst = []
    for item in csvf:
        lst.append(item)
    return lst

def makecsv(dir, f_name):
    f_list = os.listdir(dir)
    csvfile = open(f_name, 'w', newline="")
    csvwriter = csv.writer(csvfile)
    for item in f_list:
        csvwriter.writerow([item,])
    csvfile.close()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def count_labels(csv_file, path):
    lst = csv2list(csv_file)

    labels = [0,0,0,0,0]

    for l in lst:
        fs = os.listdir(path+"/"+l[0])
        for f in fs:
            labels[int(f[-5])] += 1

    print(labels)


#count_labels("/home/eslab/wyh/data/val.csv", "/home/eslab/wyh/data/img/fail/min-max-cut/EMG/")