import os, sys
from unicodedata import name

from albumentations.augmentations.functional import normalize
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import *
from lib import *
import random
from efficientnet_pytorch import EfficientNet
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler
"""

from torch.optim.lr_scheduler import MultiStepLR 
from torch.optim.lr_scheduler import StepLR
#from pytorch-cosine-annealing-with-warmup import *

import os
import argparse
import sys

from models import *
#from utils import progress_bar
import numpy as np
from pathlib import Path
#from etc import *

np.random.seed(42)
random.seed(42)

"""
from SleepDataloader import *
from util import *
from resnet_dropout import *
from resnet import *
from cosine_annearing_with_warmup import *
"""

#torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='PyTorch Sleep Stage')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--conf', '-c', action='store_true', help='Draw Confusion Matrix')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
draw = True

data_path = Path("/home/eslab/wyh/data/img/resize/spindle_eeg_nb/")
checkpoint_name = 'Spindle_eeg_nb_resnet18_batch80_black.pth'

#batch_size = 128
batch_size = 80
#batch_size = 1
class_num = 2

print(checkpoint_name)


# Data
print('==> Preparing data..')

trainset = SpinleLoader("/home/eslab/wyh/spindle_train.csv", data_path, inv=False, #shuffle=True, 
                            transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.5], std=[0.5])
                                    #transforms.Normalize(mean=[0.1551], std=[0.1427])
                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ])
                            #transform=get_train_transforms()
)  

valset = SpinleLoader("/home/eslab/wyh/spindle_valid.csv", data_path, inv=False, #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    #transforms.Normalize(mean=[0.1551], std=[0.1427])
                                    #transforms.Normalize(mean=[0.5], std=[0.5])
                                    transforms.Normalize(mean=[0.5], std=[0.5]),
                            ])
                            #transform=get_valid_transforms()
                            )

testset = SpinleLoader("/home/eslab/wyh/spindle_test.csv", data_path, inv=False, #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    #transforms.Normalize(mean=[0.1551], std=[0.1427])
                                    #transforms.Normalize(mean=[0.5], std=[0.5])
                                    transforms.Normalize(mean=[0.5], std=[0.5]),
                            ])
                            #transform=get_valid_transforms()
                            )

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

# Model
#print('==> Building model..')
net = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)

# Resnet18
net.fc = nn.Linear(512,2)

# Resnet50
#net.fc = nn.Linear(2048,2)

#net = resnet18_grayscale(num_classes=2)

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+checkpoint_name)
    #print(checkpoint)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    scheduler = checkpoint['scheduler']
    optimizer = checkpoint['optimizer']
    print("best acc: %lf" %(best_acc))
else:
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[50,100], gamma=0.1)

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loop = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}{r_bar}')
    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        #####
        #with amp.scale_loss(loss, optimizer) as scaled_loss: 
        #    scaled_loss.backward()
        #####
        loss.backward()

        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        loop.set_description(f"Epoch [{epoch}/{300}]")
        loop.set_postfix(loss = train_loss/total, acc=(100.*correct/total), correct=correct, total=total)

    acc = 100.*correct/total

    print("acc=%.3f, loss=%.5f, correct=%d, total=%d" %(acc, train_loss/total, correct, total))
    #print(net.module.printbias())

def valid(epoch):
    global best_acc
    global draw
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(valloader), total=len(valloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}{r_bar}')
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_description(f"Epoch [{epoch}/{300}]")
            loop.set_postfix(loss = valid_loss/total, acc=(100.*correct/total), correct=correct, total=total)

    # Save checkpoint.
    acc = 100.*correct/total

    print("acc=%.3f, loss=%.5f, correct=%d, total=%d" %(acc, valid_loss/total, correct, total))

    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch+1,
            'scheduler' : scheduler,
            'optimizer' : optimizer,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+checkpoint_name)
        best_acc = acc
        draw = True
    else:
        draw = False

def test(epoch):
    global best_acc
    global draw
    net.eval()
    conf = np.zeros((class_num, class_num),dtype=int)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(testloader), total=len(testloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}{r_bar}')
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            item = predicted.to('cpu').numpy()
            ans = targets.to('cpu').numpy()
            for item_idx, c in enumerate(item):
                conf[ans[item_idx]][item[item_idx]] += 1

            loop.set_description(f"Epoch [{epoch}/{300}]")
            loop.set_postfix(loss = test_loss/total, acc=(100.*correct/total), correct=correct, total=total)

    acc=(100.*correct/total)

    print("acc=%.3f, loss=%.5f, correct=%d, total=%d" %(acc, test_loss/total, correct, total))

    if draw == True:
        #draw_conf(conf, checkpoint_name)
        print(conf.tolist())

if args.resume:
    test(start_epoch-1)

for epoch in range(start_epoch, start_epoch+300):
    train(epoch)
    valid(epoch)
    test(epoch)
    scheduler.step()
