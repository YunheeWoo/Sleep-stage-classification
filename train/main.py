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
from torch.optim.lr_scheduler import StepLR
#from pytorch-cosine-annealing-with-warmup import *

import os
import argparse
import sys

from models import *
from utils import progress_bar
import numpy as np
from pathlib import Path
from etc import *

sys.path.append('/home/eslab/wyh/Sleep-stage-classification/train/etc')
sys.path.append('/home/eslab/wyh/Sleep-stage-classification/train/models')

from SleepDataloader import *
from util import *
from resnet_dropout import *
from resnet import *
from cosine_annearing_with_warmup import *

#torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='PyTorch Sleep Stage')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--conf', '-c', action='store_true', help='Draw Confusion Matrix')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

data_path = Path('/home/eslab/wyh/data/')
checkpoint_name = 'min-max-cut-2-resize-gray.pth'

# Data
print('==> Preparing data..')

trainset = SleepDataset("/home/eslab/wyh/data/train.csv", Path("/home/eslab/wyh/data/img/2000x100/t-02/min-max-cut"), ["C3-M2", "E1-M2", "E2-M1"], #color="L",
                            transform=transforms.Compose([
                                    transforms.Resize([224,224]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.0044], std=[0.0396])]))

testset = SleepDataset("/home/eslab/wyh/data/test.csv", Path("/home/eslab/wyh/data/img/2000x100/t-02/min-max-cut"), ["C3-M2", "E1-M2", "E2-M1"], #color="L",
                            transform=transforms.Compose([
                                    transforms.Resize([224,224]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.0044], std=[0.0396])]))

valset = SleepDataset("/home/eslab/wyh/data/val.csv", Path("/home/eslab/wyh/data/img/2000x100/t-02/min-max-cut"), ["C3-M2", "E1-M2", "E2-M1"], #color="L",
                            transform=transforms.Compose([
                                    transforms.Resize([224,224]),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(), 
                                    transforms.Normalize(mean=[0.0044], std=[0.0396])]))

batch_size = 32

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=5)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=5)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=5)

# Model
print('==> Building model..')
net = torch.hub.load('pytorch/vision', 'resnet50', pretrained=False)
net.fc = nn.Linear(2048,5)
#net = resnet50_grayscale()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+checkpoint_name)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    scheduler = checkpoint['scheduler']
    optimizer = checkpoint['optimizer']
else:
    optimizer = optim.SGD(net.parameters(), lr=0.0001, momentum=0.9, weight_decay=1e-4)

    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=50, cycle_mult=1.0, max_lr=0.1, min_lr=0.0001, warmup_steps=5, gamma=0.8)
    #scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0=50, T_mult=1, eta_max=0.1, T_up=5, gamma=0.8)

#######
model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
#######
criterion = nn.CrossEntropyLoss()


#conf = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]


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
        #####
        with amp.scale_loss(loss, optimizer) as scaled_loss: 
            scaled_loss.backward()
        #####
        #loss.backward()

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
        for batch_idx, (inputs, targets) in enumerate(valloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
            'scheduler' : scheduler,
            'optimizer' : optimizer,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+checkpoint_name)
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
            """
            item = predicted.to('cpu').numpy()
            ans = targets.to('cpu').numpy()
            for item_idx, c in enumerate(item):
                conf[ans[item_idx]][item[item_idx]] += 1
            """
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            
            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

#test(0)
#print(conf)
for epoch in range(start_epoch, start_epoch+300):
    train(epoch)
    valid(epoch)
    test(epoch)
    scheduler.step()
