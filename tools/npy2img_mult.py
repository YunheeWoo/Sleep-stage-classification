import os
import argparse
import numpy as np
from pathlib import Path
import datetime
import matplotlib.pyplot as plt
import argparse
import multiprocessing

parser = argparse.ArgumentParser(description='PyTorch Sleep Stage')
parser.add_argument('--idx', type=int, help='index')
parser.add_argument('--set', type=int, help='set')
args = parser.parse_args()

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
        #m = 3.5
        m = 2
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

    #print(data.shape)
    data = np.reshape(data,(-1,6000))
    #annotation = np.load(annotation_path+file)

    if not data.shape[0] == ann.shape[0]:
        print("data %d and annotation %d do not match" %(data.shape[0], ann.shape[0]))
        return

    img_num = 0

    for d_idx in range(data.shape[0]):
        plt.figure(figsize=(width/300,height/300), dpi=300)
        plt.ylim(y_min, y_max)
        plt.xlim(0,6000)
        plt.box(on=None)
        plt.axis('off')
        #plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.plot(data[d_idx], linewidth=0.2, color="black")
        img_name = str(img_num).zfill(4) + "_" + str(int(ann[d_idx])) + ".png"
        plt.savefig(path / img_name)
        plt.close('all')
        plt.cla()
        plt.clf()
        img_num += 1

def makeimg(patients):

    width = 2000
    height = 100

    print(patients)

    src_path = Path("/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals/")
    dst_path = Path("/home/eslab/wyh/data/img/")
    ann_path = Path("/data/hdd1/dataset/Seoul_dataset/annotations/")

    allow_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "O1-M2", "O2-M1"]

    img_size = str(width) + "x" + str(height) + "/t-02"

    normalization = ["min-max-cut", "mean-std-cut", "min-max-discard", "mean-std-discard", "original", "min-max", "mean-std"]
    signal_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]

    for norm in normalization[3:4]:
        print("**************************" + norm + " start **************************")
        print("===========" + patients + " start ===========")

        datas = np.load(src_path / patients)
        anns = np.load(ann_path / patients)

        for sig_idx, signal in enumerate(signal_list):
            if not signal in allow_list:
                print("> " + signal + " skip")
                continue
            
            print("┌─ " + signal + " start")
            print("\t", end="")
            print(datetime.datetime.now())

            os.makedirs(dst_path / img_size / norm / signal / patients.split(".")[0], exist_ok=True)
            
            draw_img(datas[sig_idx], dst_path / img_size / norm / signal / patients.split(".")[0], anns, width, height, norm)

            print("└─ " + signal + " end")
            print("\t", end="")
            print(datetime.datetime.now())

if __name__ == '__main__':
    src_path = Path("/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals/")
    patients = os.listdir(src_path)
    patients_pre = ['A2019-NX-01-0384_1_.npy', 'A2019-NX-01-0614_3_.npy', 'A2019-NX-01-0917_3_.npy']
    patients = [item for item in patients if item not in patients_pre]
    patients.sort()

    print(len(patients))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pool.map(makeimg, patients)
    pool.close
    pool.join

