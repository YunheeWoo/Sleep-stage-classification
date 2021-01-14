import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import csv
from typing import Any, Callable, TypeVar, Generic, Sequence, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
import pandas as pd
import os
import itertools

from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import cohen_kappa_score

def csv2list(csv_file):
    f = open(csv_file, 'r')
    csvf = csv.reader(f)
    lst = []
    for item in csvf:
        lst.append(item)
    return lst

def makecsv(dir, f_name):
    f_list = os.listdir(dir)
    csvfile = open(f_name, 'w', newline="")
    csvwriter = csv.writer(csvfile)
    for item in f_list:
        csvwriter.writerow([item,])
    csvfile.close()

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def plot_confusion_matrix(cm, pth, cmap=None, labels=True, option=None,title='Confusion matrix'):
    cm = np.array(cm)
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    target_names=['Wake', 'N1', 'N2', 'N3', 'REM']

    if option == None:
        normalize = False
    else:
        normalize = True

    temp = []

    if cmap is None:
        cmap = plt.get_cmap('GnBu')

    if option=='recall':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title='Confusion matrix (Recall)'
    elif option=='precision':
        title='Confusion matrix (Precision)'
        cm_t = cm.sum(axis=0)
        cm = cm.astype('float')
        for i in range(5):
            for j in range(5):
                cm[i][j] /= cm_t[j]

    for i in range(0, 5):
        for j in range(0, 5):
            if i == j:
                temp.append(cm[i][j])

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    thresh = cm.max() / 1.5 if normalize else cm.max() / 2

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names)
        plt.yticks(tick_marks, target_names)

    if labels:
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            if normalize:
                plt.text(j, i, "{:0.2f}".format(cm[i, j]*100),
                         ha='center', va='center', color="white" if cm[i, j] > thresh else "black", fontSize=11, fontweight='bold')
                #plt.text(j, i, '%d' % z, ha='center', va='center', color='Red', fontSize=11, fontweight='bold')
            else:
                plt.text(j, i, "{:,}".format(cm[i, j]),
                         ha='center', va='center', color="white" if cm[i, j] > thresh else "black", fontSize=11, fontweight='bold')

    img_name = None
    if option==None:
        img_name = pth+"_Abs.png"
    elif option=='precision':
        img_name = pth+"_Precision.png"
    elif option=='recall':
        img_name = pth+"_Recall.png"

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig("./figures/"+img_name)

    return temp

def get_balanced_list(cm):
    lst_true = np.sum(cm, axis=1)
    lst_pred = np.sum(cm, axis=0)

    y_true = []
    y_pred = []

    for lst in lst_true:
        y_true += [np.where(lst_true == lst)[0][0]] * lst

    for lst in cm:
        for idx,l in enumerate(lst):
            y_pred += [idx] * l

    return y_true, y_pred

def draw_conf(cm, pth):
    pre = plot_confusion_matrix(cm, pth, option='precision')
    rec = plot_confusion_matrix(cm, pth, option='recall')
    _ = plot_confusion_matrix(cm, pth)

    #print_scores(cm, pth)

    #print(2*(np.array(rec)*np.array(pre))/(np.array(rec)+np.array(pre))*100)

def print_scores(cm, pth):
    f = open(pth + "_scores.txt", 'w')

    y_true, y_pred = get_balanced_list(cm)
    f.write("- acc")
    f.write("%.2lf" %accuracy_score(y_true, y_pred)*100)
    f.write("- bal_acc")
    f.write("%.2lf" %balanced_accuracy_score(y_true, y_pred)*100)
    f.write("- f1")
    f.write(f1_score(y_true, y_pred, average='None')*100)
    f.write("- f1_macro")
    f.write("%.2lf" %f1_score(y_true, y_pred, average='macro')*100)
    f.write("- cohen kappa")
    f.write("%.2lf" %cohen_kappa_score(y_true, y_pred)*100)

    f.close()
    

def count_labels(csv_file, path):
    lst = csv2list(csv_file)

    labels = [0,0,0,0,0]

    for l in lst:
        fs = os.listdir(path+"/"+l[0])
        for f in fs:
            labels[int(f[-5])] += 1

    print(labels)

def count_severity(csv_file):
    lst = csv2list(csv_file)

    labels = [0,0,0,0]

    for l in lst:
        labels[int(l[0][-2])] += 1

    print(labels)


#count_labels("/home/eslab/wyh/data/val.csv", "/home/eslab/wyh/data/img/fail/min-max-cut/EMG/")

#count_severity("/home/eslab/wyh/data/train.csv")