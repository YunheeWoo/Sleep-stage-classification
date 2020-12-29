'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data.sampler import SubsetRandomSampler

from torch.optim.lr_scheduler import MultiStepLR 
#from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import StepLR
from cosine_annearing_with_warmup import *

import os
import argparse

#from models import *
from utils import progress_bar
import numpy as np

#torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

test_idx = 0

# Data
print('==> Preparing data..')

trainset = ImageFolder(root='/home/eslab/wyh/medical/data/SNU/img/1920x40/Con_v_FFT_C3-M2,E2-M1,E1-M2/train',transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), 
                               transforms.Normalize(mean=[0.9755, 0.9819, 0.9867],
                                     std=[0.1405, 0.1195, 0.1016])
                           ]))
                    
testset = ImageFolder(root='/home/eslab/wyh/medical/data/SNU/img/1920x40/Con_v_FFT_C3-M2,E2-M1,E1-M2/test',transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), 
                               transforms.Normalize(mean=[0.9755, 0.9819, 0.9867],
                                     std=[0.1405, 0.1195, 0.1016])
                           ]))

valset = ImageFolder(root='/home/eslab/wyh/medical/data/SNU/img/1920x40/Con_v_FFT_C3-M2,E2-M1,E1-M2/val',transform=transforms.Compose([
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(), 
                               transforms.Normalize(mean=[0.9755, 0.9819, 0.9867],
                                     std=[0.1405, 0.1195, 0.1016])
                           ]))

batch_size = 5

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)

# Model
print('==> Building model..')
net = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
net.fc = nn.Linear(512,5)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/FFT_v.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)
#scheduler = MultiStepLR(optimizer, milestones=[30,60], gamma=0.1)
#scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)

scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=30, T_mult=1, eta_max=0.1, T_up=5, gamma=0.8)


conf = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        #scheduler_warmup.step(epoch)
        #for param_group in optimizer.param_groups:
        #    return param_group['lr']
        
        scheduler.step()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def valid(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/FFT_v.pth')
        best_acc = acc

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            
            item = predicted.to('cpu').numpy()
            ans = targets.to('cpu').numpy()
            for item_idx, c in enumerate(item):
                conf[ans[item_idx]][item[item_idx]] += 1
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

test(0)
print(conf)
#for epoch in range(start_epoch, start_epoch+300):
#    train(epoch)
#    valid(epoch)
#    test(epoch)
