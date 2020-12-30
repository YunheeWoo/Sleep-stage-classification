import os
import argparse
import numpy as np

src_path = "/home/eslab/wyh/data/"
dst_path = src_path + "img/"

dirs = ['test/', 'train/', 'val/']

for d in dirs:
    patients = os.listdir(src_path+d)
    for p in patients:
        os.makedirs(dst_path+d+p, exist_ok=True)
        datas = os.listdir(src_path+d+p)
        for data in datas:
            epoch = np.load(src_path+d+p+'/'+data)

            