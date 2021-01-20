import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import *
from lib import *
import random

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

data_path = Path('/home/eslab/wyh/data/')
checkpoint_name = 'resnet18-fix-nonflip-mean-std-discard-7-resize-gray-full.pth'

print(checkpoint_name)

# Data
print('==> Preparing data..')

trainset = SleepDataset("/home/eslab/wyh/train_full.csv", Path("/home/eslab/wyh/data/img/2000x100/t-02/mean-std-discard/"), ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "EMG"], inv=True, color="L", #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.Resize([224,224]),
                                    #transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.0044], std=[0.0396])]))

valset = SleepDataset("/home/eslab/wyh/val_full.csv", Path("/home/eslab/wyh/data/img/2000x100/t-02/mean-std-discard/"), ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "EMG"], inv=True, color="L", #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.Resize([224,224]),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.0044], std=[0.0396])]))

testset = SleepDataset("/home/eslab/wyh/test_full.csv", Path("/home/eslab/wyh/data/img/2000x100/t-02/mean-std-discard/"), ["C3-M2", "C4-M1", "O1-M2", "O2-M1", "E1-M2", "E2-M1", "EMG"], inv=True, color="L", #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.Resize([224,224]),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.0044], std=[0.0396])]))

batch_size = 256

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=8)

# Model
print('==> Building model..')
#net = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
#net.fc = nn.Linear(2048,5)
net = resnet18_grayscale()
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
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=30, cycle_mult=1.0, max_lr=0.1, min_lr=0.0001, warmup_steps=5, gamma=0.8)
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    #scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)

#######
#model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
#######
criterion = nn.CrossEntropyLoss()


conf = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loop = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:20}{r_bar}')
    for batch_idx, (inputs, targets) in loop:
        inputs, targets = inputs.to(device), targets.to(device)
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
        loop.set_postfix(loss = train_loss/total, acc=(100.*correct/total))

def valid(epoch):
    global best_acc
    net.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(valloader), total=len(valloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:20}{r_bar}')
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_description(f"Epoch [{epoch}/{300}]")
            loop.set_postfix(loss = valid_loss/total, acc=(100.*correct/total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch+1,
            'scheduler' : scheduler,
            'optimizer' : optimizer,
            #'optimizer': optimizer.state_dict(),
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
    net.eval()
    conf = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(testloader), total=len(testloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:20}{r_bar}')
        for batch_idx, (inputs, targets) in loop:
            inputs, targets = inputs.to(device), targets.to(device)
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
            loop.set_postfix(loss = test_loss/total, acc=(100.*correct/total))

    if draw == True:
        draw_conf(conf, checkpoint_name)
        print(conf)

#if args.resume:
#    test(start_epoch-1)

for epoch in range(start_epoch, start_epoch+300):
    train(epoch)
    valid(epoch)
    #test(epoch)
    scheduler.step()
