import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import *
from lib import *

src_path = Path("/data/ssd1/dataset/seoul_nx_edf/First/2020/")
#dst_path = Path("/data/ssd1/original_edf/NX-2019")

file_list = os.listdir(src_path)
file_list.sort()

if "#" in file_list[0]:
    file_list = file_list[1:]

for f in file_list:
    os.system("cp /data/ssd1/dataset/seoul_nx_edf/First/2020/"+f+"/1.\ EDF/*"
                 + " /data/ssd1/original_edf/NX-2020/")

    os.system("cp /data/ssd1/dataset/seoul_nx_edf/First/2020/"+f+"/2.\ Event/*"
                 + " /data/ssd1/original_annotation/NX-2020/")

                 #/data/ssd1/dataset/seoul_nx_edf/Second/2019_originDirectory/A2019-NX-01-0404/1. EDF/