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

    #src_path = Path("/data/hdd1/dataset/Seoul_image/2000x100/t-02/mean-std-discard/")
    #src_path = Path("/data/ssd2/img/1920x83/t-02/mean-std-cut/")
    src_path2 = Path("/home/eslab/wyh/data/test/1920x249/t-02/mean-std-cut/Flow/")

    
    #src_path = Path("/home/eslab/wyh/data/img/2000x100/t-02/mean-std-discard/")

    src_path = Path("/home/eslab/wyh/data/test/1920x83/t-02/mean-std-cut/")
    dst_path = Path("/home/eslab/wyh/data/test/concat/")

    signals = ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "EMG", "ECG", "Flow", "Chest", "Abdomen"]
    #signals = ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "EMG", "ECG"]
    #signals = ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1"]
    #signals = ["C3-M2", "C4-M1", "O1-M2", "O2-M1"]

    img = None

    cnt = 0

    with open(src_path / signals[0] / path, 'rb') as f:
        # Open image
        img = Image.open(f)

        img = img.convert('L')

        img = PIL.ImageOps.invert(img)

    dst = Image.new('L', (1920, 1080))

    dst.paste(img, (0, 0))

    """
    for signal in signals[1:-1]:
        with open(src_path / signal / path, 'rb') as f:
            img = Image.open(f)

            img = img.convert('L')
        dst.paste(img, (0, img.height*signals.index(signal)))

    
    if "ECG" in signals:
        with open(src_path2 / path, 'rb') as f:
            img = Image.open(f)

            img = img.convert('L')

        dst.paste(img, (0, img.height*signals.index("ECG")))
    """

    for signal in signals[1:]:

        cnt += 1

        if signal == "Flow":
            with open(src_path2 / path, 'rb') as f:
                img = Image.open(f)

                img = img.convert('L')
                img = PIL.ImageOps.invert(img)
            dst.paste(img, (0, 83*cnt))

            cnt += 2

        else:
            with open(src_path / signal / path, 'rb') as f:
                img = Image.open(f)

                img = img.convert('L')
                img = PIL.ImageOps.invert(img)
            dst.paste(img, (0, 83*cnt))

    #img_resize = dst.resize((224, 224))
    #img_resize.save(dst_path / path)

    dst.save(dst_path / path)

if __name__ == '__main__':
    #src_path = Path("/data/ssd1/img/2000x100/t-05/mean-std-discard")
    src_path = Path("/data/ssd2/img/1920x83/t-02/mean-std-cut/")
    dst_path = Path("/home/eslab/wyh/data/img/original/1920x1080/t-02/mean-std-cut")


    src_path = Path("/home/eslab/wyh/data/test/1920x83/t-02/mean-std-cut/")
    dst_path = Path("/home/eslab/wyh/data/test/concat/")

    signals = os.listdir(src_path)

    signals.sort()

    signals = ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "EMG", "ECG", "Flow", "Chest", "Abdomen"]
    #signals = ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "EMG", "ECG"]
    #signals = ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1"]
    #signals = ["C3-M2", "C4-M1", "O1-M2", "O2-M1"]

    patient = os.listdir(src_path / signals[0])

    #patients_pre = ['A2019-NX-01-0384_1_.npy', 'A2019-NX-01-0614_3_.npy', 'A2019-NX-01-0917_3_.npy']
    #patient = [item for item in patient if item not in patients_pre]

    patient.sort()

    file_list = []

    for p in patient:
        img_list = os.listdir(src_path / signals[0] / p)

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


    

