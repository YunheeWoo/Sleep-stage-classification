import os
import argparse
import numpy as np
from pathlib import Path

width = 2000
height = 100

def drawimg(data, path, w, h):

        cut_value = 192*1e-06

        for file_idx, file in enumerate(signal_files):
            data = np.load(npy_path+folder+'/'+file)[1]
            #data = (data - np.min(data)) / (np.max(data) - np.min(data))
            data = np.where(data < -cut_value, -cut_value, data)
            data = np.where(data > cut_value, cut_value, data)
            data = (data - np.min(data)) / (np.max(data) - np.min(data))
            data = np.reshape(data,(-1,6000))
            annotation = np.load(annotation_path+file)
            
            for j in range(data.shape[0]):
                plt.figure(figsize=(img_width/100,img_height/100), dpi=100)
                plt.ylim(0, 1)
                plt.xlim(0,6000)
                plt.box(on=None)
                plt.axis('off')
                plt.tight_layout()
                plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
                plt.plot(data[j], linewidth=1)
                plt.savefig(img_path+folder+'/'+str(annotation[j])+'/'+str(img_num).zfill(6)+".png")
                plt.close('all')
                plt.cla()
                plt.clf()
                img_num += 1

            print(file + " done")

src_path = Path("/home/eslab/wyh/data/npy")
dst_path = Path("/home/eslab/wyh/data/img")
ann_path = Path("/home/eslab/wyh/data/annotations")

signal_list = ["EMG", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"]

patients = os.listdir(src_path)
patients.sort()
for p in patients:
    os.makedirs(dst_path+p, exist_ok=True)
    datas = os.listdir(src_path / p)
    datas.sort()
    for data in datas:
        epoch = np.load(src_path / p / data)


