import os
import argparse
import numpy as np
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import argparse
import multiprocessing
from PIL import Image
import PIL.ImageOps 

def img_resize(path):
    src_path = Path("/home/eslab/wyh/data/img/original/1920x1080/t-02/mean-std-cut/")
    dst_path = Path("/home/eslab/wyh/data/img/resize/1920x1080-336x224/t-02/mean-std-cut")

    img = None
    with open(src_path / path, 'rb') as f:
        # Open image
        img = Image.open(f)

        img = img.convert('L')

        #img = PIL.ImageOps.invert(img)

        img_resize = img.resize((336, 224))
        img_resize.save(dst_path / path)

if __name__ == '__main__':
    src_path = Path("/home/eslab/wyh/data/img/original/1920x1080/t-02/mean-std-cut/")
    dst_path = Path("/home/eslab/wyh/data/img/resize/1920x1080-336x224/t-02/mean-std-cut")

    patient = os.listdir(src_path)

    patients_pre = ['A2019-NX-01-0384_1_.npy', 'A2019-NX-01-0614_3_.npy', 'A2019-NX-01-0917_3_.npy']
    patient = [item for item in patient if item not in patients_pre]

    patient.sort()

    file_list = []

    for p in patient:
        img_list = os.listdir(src_path / p)

        #for s in signals:
        os.makedirs(dst_path / p, exist_ok=True)
        for img in img_list:
            #file_list.append(s+"/"+p+"/"+img)
            file_list.append(p+"/"+img)

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #pool = multiprocessing.Pool(processes=18)
    pool.map(img_resize, file_list)
    pool.close
    pool.join


    

