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

width = 2000
height = 100

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
        plt.plot(data[d_idx], linewidth=0.05, color="black")
        img_name = str(img_num).zfill(4) + "_" + str(int(ann[d_idx])) + ".png"
        plt.savefig(path / img_name)
        plt.close('all')
        plt.cla()
        plt.clf()
        img_num += 1

#src_path = Path("/home/eslab/wyh/data/npy/original")
src_path = Path("/data/hdd1/dataset/Seoul_dataset/9channel_prefilter_butter/signals/")
#dst_path = Path("/home/eslab/wyh/data/img/")
dst_path = Path("/home/eslab/wyh/data/img/")
#ann_path = Path("/home/eslab/wyh/data/annotations")
ann_path = Path("/data/hdd1/dataset/Seoul_dataset/annotations/")

allow_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "O1-M2", "O2-M1"]

img_size = str(width) + "x" + str(height) + "/t-02"

normalization = ["min-max-cut", "mean-std-cut", "min-max-discard", "mean-std-discard", "original", "min-max", "mean-std"]
signal_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]

patients = os.listdir(src_path)
patients.sort()

#patients_pre = ['A2020-NX-01-0032_1_.npy', 'A2020-NX-01-0029_2_.npy', 'A2020-NX-01-0125_3_.npy', 'A2019-NX-01-1199_0_.npy', 'A2019-NX-01-0095_2_.npy', 'A2020-NX-01-0389_3_.npy', 'A2019-NX-01-0024_3_.npy', 'A2019-NX-01-0398_1_.npy', 'A2020-NX-01-0732_3_.npy', 'A2019-NX-01-1254_3_.npy', 'A2020-NX-01-0436_3_.npy', 'A2019-NX-01-0594_3_.npy', 'A2020-NX-01-0427_2_.npy', 'A2020-NX-01-0168_1_.npy', 'A2019-NX-01-0466_3_.npy', 'A2020-NX-01-0084_2_.npy', 'A2019-NX-01-0893_3_.npy', 'A2019-NX-01-0259_3_.npy', 'A2020-NX-01-0011_3_.npy', 'A2019-NX-01-0131_1_.npy', 'A2019-NX-01-0853_3_.npy', 'A2019-NX-01-1415_2_.npy', 'A2019-NX-01-1016_3_.npy', 'A2020-NX-01-0098_3_.npy', 'A2019-NX-01-1083_2_.npy', 'A2019-NX-01-0617_1_.npy', 'A2019-NX-01-0067_2_.npy', 'A2020-NX-01-0643_3_.npy', 'A2019-NX-01-1585_2_.npy', 'A2020-NX-01-0545_3_.npy', 'A2019-NX-01-1640_3_.npy', 'A2019-NX-01-0584_3_.npy', 'A2019-NX-01-0962_3_.npy', 'A2019-NX-01-0972_3_.npy', 'A2020-NX-01-0315_2_.npy', 'A2019-NX-01-0728_3_.npy', 'A2020-NX-01-0595_3_.npy', 'A2020-NX-01-0700_3_.npy', 'A2019-NX-01-0480_1_.npy', 'A2020-NX-01-0167_0_.npy', 'A2019-NX-01-0632_0_.npy', 'A2020-NX-01-0358_2_.npy', 'A2019-NX-01-1563_2_.npy', 'A2020-NX-01-0652_3_.npy', 'A2020-NX-01-0074_3_.npy', 'A2019-NX-01-0028_0_.npy', 'A2020-NX-01-0766_3_.npy', 'A2019-NX-01-1401_3_.npy', 'A2019-NX-01-1603_2_.npy', 'A2019-NX-01-1134_0_.npy', 'A2020-NX-01-0658_3_.npy', 'A2019-NX-01-1017_3_.npy', 'A2020-NX-01-0559_3_.npy', 'A2019-NX-01-0082_3_.npy', 'A2019-NX-01-0419_3_.npy', 'A2019-NX-01-0271_3_.npy', 'A2020-NX-01-0294_3_.npy', 'A2019-NX-01-0108_3_.npy', 'A2019-NX-01-0642_2_.npy', 'A2019-NX-01-0226_2_.npy', 'A2019-NX-01-1126_3_.npy', 'A2019-NX-01-0353_3_.npy', 'A2019-NX-01-1087_1_.npy', 'A2020-NX-01-0335_3_.npy', 'A2020-NX-01-0333_2_.npy', 'A2020-NX-01-0008_2_.npy', 'A2020-NX-01-0737_3_.npy', 'A2019-NX-01-1223_1_.npy', 'A2019-NX-01-0650_2_.npy', 'A2019-NX-01-1135_1_.npy', 'A2020-NX-01-0118_2_.npy', 'A2019-NX-01-0498_3_.npy', 'A2020-NX-01-0549_3_.npy', 'A2019-NX-01-0433_3_.npy', 'A2020-NX-01-0705_3_.npy', 'A2020-NX-01-0312_1_.npy', 'A2019-NX-01-1362_2_.npy', 'A2019-NX-01-0062_3_.npy', 'A2020-NX-01-0494_2_.npy', 'A2019-NX-01-1433_3_.npy', 'A2019-NX-01-0175_3_.npy', 'A2019-NX-01-0039_1_.npy', 'A2020-NX-01-0432_3_.npy', 'A2020-NX-01-0112_1_.npy', 'A2019-NX-01-0845_2_.npy', 'A2019-NX-01-1638_3_.npy', 'A2019-NX-01-1161_3_.npy', 'A2019-NX-01-1021_3_.npy', 'A2019-NX-01-1440_3_.npy', 'A2019-NX-01-1296_0_.npy', 'A2019-NX-01-1308_3_.npy', 'A2019-NX-01-1041_3_.npy', 'A2020-NX-01-0338_1_.npy', 'A2019-NX-01-0041_2_.npy', 'A2019-NX-01-1374_1_.npy', 'A2020-NX-01-0006_3_.npy', 'A2019-NX-01-0683_3_.npy', 'A2020-NX-01-0464_3_.npy', 'A2020-NX-01-0724_3_.npy', 'A2019-NX-01-1331_3_.npy']
#patients_pre = os.listdir("/data/hdd1/dataset/Seoul_image/2000x100/t-02/mean-std-discard/O2-M1")
#for idx, _ in enumerate(patients_pre):
#    patients_pre[idx] = patients_pre[idx] + '.npy'

patients_pre = ['A2019-NX-01-0384_1_.npy', 'A2019-NX-01-0614_3_.npy', 'A2019-NX-01-0917_3_.npy']

patients = [item for item in patients if item not in patients_pre]
patients.sort()

#patients = ["A2020-NX-01-0737_3_.npy"]

print(len(patients))

#random.shuffle(patients)

idx = 0
#threshold = args.set

for norm in normalization[3:4]:
    print("**************************" + norm + " start **************************")
    for p in patients[227*idx:227*(idx+1)]:
    #for p in patients:
        print("===========" + p + " start ===========")

        datas = np.load(src_path / p)
        anns = np.load(ann_path / p)

        for sig_idx, signal in enumerate(signal_list):
            if not signal in allow_list:
                print("> " + signal + " skip")
                continue
            
            print("┌─ " + signal + " start")
            print("\t", end="")
            print(datetime.datetime.now())

            os.makedirs(dst_path / img_size / norm / signal / p.split(".")[0], exist_ok=True)
            
            draw_img(datas[sig_idx], dst_path / img_size / norm / signal / p.split(".")[0], anns, width, height, norm)

            print("└─ " + signal + " end")
            print("\t", end="")
            print(datetime.datetime.now())

