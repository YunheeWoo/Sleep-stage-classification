import os
import argparse
import numpy as np
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='PyTorch Sleep Stage')
parser.add_argument('--idx', type=int, help='index')
parser.add_argument('--set', type=int, help='set')
args = parser.parse_args()

width = 1920
height = 83

def draw_img(data, path, ann, width, height, norm):

    std = np.std(data)
    mean = np.mean(data)

    y_min = None
    y_max = None

    # preprocessing
    if "cut" in norm:
        cut_value = 192*1e-06
        data = np.where(data < -cut_value, -cut_value, data)
        data = np.where(data > cut_value, cut_value, data)
        
    if "discard" in norm:
        m = 3.5
        idx1 = (data - np.mean(data)) > m * np.std(data)
        idx2 = (data - np.mean(data)) < -m * np.std(data)
        data[idx1] = m * std
        data[idx2] = -m * std

    if "min" in norm:
        data = (data - np.min(data)) / (np.max(data) - np.min(data))
        y_min = 0
        y_max = 1

    if "mean" in norm:
        data = (data - mean) / std

        if abs(np.min(data)) > abs(np.max(data)):
            y_min = -abs(np.min(data))
            y_max = abs(np.min(data))
        else:
            y_min = -abs(np.max(data))
            y_max = abs(np.max(data))

    data = np.reshape(data,(-1,3000))
    #annotation = np.load(annotation_path+file)

    if not data.shape[0] == len(ann):
        print("data %d and annotation %d do not match" %(data.shape[0], ann.shape[0]))
        return

    img_num = 0

    for d_idx in range(data.shape[0]):
        plt.figure(figsize=(width/300,height/300), dpi=300)
        #plt.ylim(y_min, y_max)
        plt.xlim(0,3000)
        plt.box(on=None)
        plt.axis('off')
        #plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.plot(data[d_idx], linewidth=0.2, color="black")
        img_name = str(img_num).zfill(4) + "_" + str(ann[d_idx]) + ".png"
        plt.savefig(path / img_name)
        plt.close('all')
        plt.cla()
        plt.clf()
        img_num += 1

src_path = Path("/data/ssd2/1-D_signal/100Hz_normed_divided/")
dst_path = Path("/data/ssd2/1-D_signal/img_test/")

allow_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]

img_size = str(width) + "x" + str(height) + "/t-02"

normalization = ["min-max-cut", "mean-std-cut", "min-max-discard", "mean-std-discard", "original", "min-max", "mean-std"]
signal_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]

patients = os.listdir(src_path)
patients.sort()
patients = patients[0:1]

idx = args.idx
threshold = args.set

for norm in normalization[4:5]:
    print(norm)
    print("**************************" + norm + " start **************************")
    #for p in patients[idx*40+threshold:(idx+1)*40]:
    for p in patients:

        print("====load datas====")

        split_list = os.listdir(src_path / p)

        split_list.sort()

        datas = np.load(src_path / p / split_list[0])

        anns = [int(split_list[0][-5])]

        for split in split_list[1:]:
            datas = np.append(datas, np.load(src_path / p / split))
            anns.append(split[-5])

        print("====finish to concat datas====")

        print("===========" + p + " start ===========")

        datas = np.reshape(datas,(9,-1))

        for sig_idx, signal in enumerate(signal_list):
            if not signal in allow_list:
                print("> " + signal + " skip")
                continue
            
            print("┌─ " + signal + " start")
            print("\t", end="")
            print(datetime.datetime.now())

            os.makedirs(dst_path / img_size / norm / signal / p, exist_ok=True)
            
            draw_img(datas[sig_idx], dst_path / img_size / norm / signal / p, anns, width, height, norm)

            print("└─ " + signal + " end")
            print("\t", end="")
            print(datetime.datetime.now())

