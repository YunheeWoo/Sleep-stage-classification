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

def smin(data):
    mm = data[0][0]
    for i in data[0]:
        if i < mm and i > np.min(data):
            mm = i
    return mm

def draw_img(data, path, ann, width, height, norm):

    std = np.std(data)
    mean = np.mean(data)

    y_min = None
    y_max = None

    # preprocessing
    if "cut" in norm:
        dic = {"C3-M2":2.50E-05, "C4-M1":2.50E-05, "O1-M2":2.50E-05, "O2-M1":2.50E-05, "E1-M2":3.50E-05, "E2-M1":3.50E-05, "EMG":2.50E-05, "ECG":0.0005, "Flow":1.5, "Thorax":0.000125, "Abdomen":0.000125}
        #cut_value = 192*1e-06
        cut_value = 2.5e-05    # C3-M2, C4-M1, O1-M2, O2-M1, EMG
        #cut_value = 3.5e-05    # E1-M2, E2-M1
        #cut_value = 0.0005     # ECG
        #cut_value = 1.5         # Flow
        #cut_value = 0.000125   # old Thorax, Abdomen

        #cut_value = 0.0001     # Thorax, Abdomen

        #cut_value = 100

        cut_value = 3.5e-05 * 3
        
        #data = np.where(data < -cut_value, -1, data)
        #data = np.where(data > cut_value, 1, data)
        
    if "discard" in norm:
        #m = 7
        m = 3.5
        #m = 2
        print(m * np.std(data))
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

        y_min = (-cut_value - mean) / std
        y_max = (cut_value - mean) / std

    #print(data.shape)
    data = np.reshape(data,(-1,6000))
    #data = np.reshape(data,(-1,750))
    #annotation = np.load(annotation_path+file)

    if not data.shape[0] == ann.shape[0]:
        print(data.shape)
        print(ann.shape)
        print("data %d and annotation %d do not match" %(data.shape[0], ann.shape[0]))
        return

    #print(str(np.max(data))+" "+str(smin(data)))

    img_num = 0

    for d_idx in range(data.shape[0]):
        #print(data[d_idx].shape)
        plt.figure(figsize=(width/300,height/300), dpi=300)
        plt.ylim(y_min, y_max)
        #plt.ylim(39, 100)
        plt.xlim(0,6000)
        #plt.xlim(0,750)
        plt.box(on=None)
        plt.axis('off')
        #plt.tight_layout()
        plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
        plt.plot(data[d_idx], linewidth=0.25, color="black")
        img_name = str(img_num).zfill(4) + "_" + str(int(ann[d_idx])) + ".png"
        plt.savefig(path / img_name)
        plt.close('all')
        plt.cla()
        plt.clf()
        img_num += 1

def makeimg(patients):

    width = 1920
    #height = 83
    #height = 166
    height = 249

    height = 83*3

    print(patients)


    src_path = Path("/data/ssd1/spindle_dataset/preprocessing_data/signals_200Hz/")

    dst_path = Path("/data/ssd1/spindle_dataset/preprocessing_data/image/")
    ann_path = Path("/data/ssd1/spindle_dataset/preprocessing_data/annotations/")

    allow_list = ["E1-M2", "E2-M1"]
    #allow_list = ["F3-M2", "F4-M1"]

    img_size = str(width) + "x" + str(height) + "/t-02"

    normalization = ["min-max-cut", "mean-std-cut", "min-max-discard", "mean-std-discard", "original", "min-max", "mean-std"]
    #signal_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "ECG", "F3-M2", "F4-M1", "Flow", "O1-M2", "O2-M1"]
    signal_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "ECG", "Flow", "O1-M2", "O2-M1"]
    #signal_list = ["Abdomen","Chest"]
    #signal_list = ["F3-M2", "F4-M1"]
    #signal_list = ["Saturation_40-100"]




    for norm in normalization[1:2]:
    #for norm in normalization[4:5]: # original
        print("**************************" + norm + " start **************************")
        print("===========" + patients + " start ===========")

        datas = np.load(src_path / patients)
        anns = np.load(ann_path / patients)

        print("=======")
        print(datas.shape)
        print(anns.shape)

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
    src_path = Path("/data/hdd2/dataset/1-D_signal/Seoul/signals_200Hz/")
    patients = os.listdir(src_path)
    #patients_pre = ['A2019-NX-01-0384_1_.npy', 'A2019-NX-01-0614_3_.npy', 'A2019-NX-01-0917_3_.npy']
    #patients = [item for item in patients if item not in patients_pre]
    #patients = ['A2019-NX-01-0001_3_.npy', 'A2019-NX-01-1000_3_.npy', 'A2019-NX-01-1350_3_.npy', 'A2019-NX-01-1440_3_.npy']
    patients.sort()

    print(len(patients))

    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    #pool = multiprocessing.Pool(processes=10)
    pool.map(makeimg, patients)
    pool.close
    pool.join

