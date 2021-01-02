import os
import argparse
import numpy as np
from pathlib import Path
import datetime
import matplotlib.pyplot as plt

width = 2000
height = 100

def draw_img(data, path, ann, width, height, norm):

    # preprocessing
    if norm == "min-max-cut" and norm == "mean-std-cut":
        cut_value = 192*1e-06
        data = np.where(data < -cut_value, -cut_value, data)
        data = np.where(data > cut_value, cut_value, data)

    if norm == "min-max-discard" and norm == "mean-std-discard":
        m = 3.5
        std = np.std(data)
        idx1 = (data - np.mean(data)) > m * np.std(data)
        idx2 = (data - np.mean(data)) < -m * np.std(data)
        data[idx1] = m * std
        data[idx2] = -m * std

    data = np.reshape(data,(-1,6000))
    #annotation = np.load(annotation_path+file)

    if not data.shape[0] == ann.shape[0]:
        print("data %d and annotation %d do not match" %(data.shape[0], ann[0]))
        return

    img_num = 0

    for d_idx in range(data.shape[0]):
        plt.figure(figsize=(width/300,height/300), dpi=300)
        plt.ylim(np.min(data), np.max(data))
        plt.xlim(0,6000)
        plt.box(on=None)
        plt.axis('off')
        plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.plot(data[d_idx], linewidth=0.1, color="black")
        img_name = str(img_num).zfill(4) + "_" + str(ann[d_idx]) + ".png"
        plt.savefig(path / img_name)
        plt.close('all')
        plt.cla()
        plt.clf()
        img_num += 1

src_path = Path("/home/eslab/wyh/data/npy/original")
dst_path = Path("/home/eslab/wyh/data/img")
ann_path = Path("/home/eslab/wyh/data/annotations")

img_size = str(width) + "x" + str(height)

normalization = ["min-max-cut", "mean-std-cut", "min-max-discard", "min-max-discard", "original", "min-max", "mean-std"]

signal_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]

patients = os.listdir(src_path)
patients.sort()

for norm in normalization[1:]:
    print("**************************" + norm + " start **************************")
    for p in patients[320:]:
        print("===========" + p + " start ===========")

        datas = np.load(src_path / p)
        anns = np.load(ann_path / p)

        for sig_idx, signal in enumerate(signal_list):
            if signal == "F3-M2":
                print("> F3-M2 skip")
                continue
            if signal == "F4-M1":
                print("> F4-M1 skip")
                continue
            
            print("- " + signal + " start")
            print("\t", end="")
            print(datetime.datetime.now())

            os.makedirs(dst_path / img_size / norm / signal / p.split(".")[0], exist_ok=True)
            
            draw_img(datas[sig_idx], dst_path / img_size / norm / signal / p.split(".")[0], anns, width, height, norm)

            print("- " + signal + " end")
            print("\t", end="")
            print(datetime.datetime.now())


