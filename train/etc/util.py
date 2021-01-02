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

def pil_loader_grayscale(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

class Imageloader(ImageFolder):
    def __init__(
            self,
            root: str,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            loader: Callable[[str], Any] = pil_loader_grayscale,
            is_valid_file: Optional[Callable[[str], bool]] = None,
    ):
        super(Imageloader, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file)
        self.imgs = self.samples
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        target = int(path[-5])
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target
