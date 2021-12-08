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

data_path = Path("/home/eslab/wyh/data/img/resize/spindle_each_channel_sk/")

checkpoint_name1 = 'SPindle_test_withB_resnet18_rgb_C3-M2.pth'
checkpoint_name2 = 'SPindle_test_withB_resnet18_rgb_C4-M1.pth'
checkpoint_name3 = 'SPindle_test_withB_resnet18_rgb_F3-M2.pth'
checkpoint_name4 = 'SPindle_test_withB_resnet18_rgb_F4-M1.pth'
checkpoint_name5 = 'SPindle_test_withB_resnet18_rgb_O1-M2.pth'
checkpoint_name6 = 'SPindle_test_withB_resnet18_rgb_O2-M1.pth'
checkpoint_name = 'SPindle_test_withB_resnet18_rgb_total_FC2.pth'

#batch_size = 128
batch_size = 32
#batch_size = 1
class_num = 2


# Data
print('==> Preparing data..')

trainset = SpinleLoaderM("/home/eslab/wyh/spindle_train.csv", data_path, signals=["C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"], inv=False, #shuffle=True, 
                            transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    #transforms.Normalize(mean=[0.5], std=[0.5])
                                    #transforms.Normalize(mean=[0.1551], std=[0.1427])
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ])
                            #transform=get_train_transforms()
)  

valset = SpinleLoaderM("/home/eslab/wyh/spindle_valid.csv", data_path, signals=["C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"], inv=False, #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    #transforms.Normalize(mean=[0.1551], std=[0.1427])
                                    #transforms.Normalize(mean=[0.5], std=[0.5])
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ])
                            #transform=get_valid_transforms()
                            )

testset = SpinleLoaderM("/home/eslab/wyh/spindle_test.csv", data_path, signals=["C3-M2", "C4-M1", "F3-M2", "F4-M1", "O1-M2", "O2-M1"], inv=False, #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    #transforms.Normalize(mean=[0.1551], std=[0.1427])
                                    #transforms.Normalize(mean=[0.5], std=[0.5])
                                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ])
                            #transform=get_valid_transforms()
                            )

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=8)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

# Model
#print('==> Building model..')
net1 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
net1.fc = nn.Linear(512,2)

net2 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
net2.fc = nn.Linear(512,2)

net3 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
net3.fc = nn.Linear(512,2)

net4 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
net4.fc = nn.Linear(512,2)

net5 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
net5.fc = nn.Linear(512,2)

net6 = torch.hub.load('pytorch/vision', 'resnet18', pretrained=True)
net6.fc = nn.Linear(512,2)
#net = resnet18_grayscale(num_classes=2)

net = FC2()

net1 = net1.to(device)
net2 = net2.to(device)
net3 = net3.to(device)
net4 = net4.to(device)
net5 = net5.to(device)
net6 = net6.to(device)
net = net.to(device)

if device == 'cuda':
    net1 = torch.nn.DataParallel(net1)
    net2 = torch.nn.DataParallel(net2)
    net3 = torch.nn.DataParallel(net3)
    net4 = torch.nn.DataParallel(net4)
    net5 = torch.nn.DataParallel(net5)
    net6 = torch.nn.DataParallel(net6)
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

checkpoint1 = torch.load('./checkpoint/'+checkpoint_name1)
checkpoint2 = torch.load('./checkpoint/'+checkpoint_name2)
checkpoint3 = torch.load('./checkpoint/'+checkpoint_name3)
checkpoint4 = torch.load('./checkpoint/'+checkpoint_name4)
checkpoint5 = torch.load('./checkpoint/'+checkpoint_name5)
checkpoint6 = torch.load('./checkpoint/'+checkpoint_name6)

net1.load_state_dict(checkpoint1['net'])
net2.load_state_dict(checkpoint2['net'])
net3.load_state_dict(checkpoint3['net'])
net4.load_state_dict(checkpoint4['net'])
net5.load_state_dict(checkpoint5['net'])
net6.load_state_dict(checkpoint6['net'])

net1.module.fc = Identity()
net2.module.fc = Identity()
net3.module.fc = Identity()
net4.module.fc = Identity()
net5.module.fc = Identity()
net6.module.fc = Identity()

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #checkpoint = torch.load('./checkpoint/'+checkpoint_name1)
    #print(checkpoint)
    #net.load_state_dict(checkpoint['net'])
    #best_acc = checkpoint['acc']
    #start_epoch = checkpoint['epoch']
    #scheduler = checkpoint['scheduler']
    #optimizer = checkpoint['optimizer']
    print("best acc: %lf" %(best_acc))
else:
    optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optimizer, milestones=[20,40], gamma=0.1)

criterion = nn.CrossEntropyLoss()


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    loop = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}{r_bar}')
    for batch_idx, (input1, input2, input3, input4, input5, input6, targets) in loop:
        input1 = input1.to(device, dtype=torch.float)
        input2 = input2.to(device, dtype=torch.float)
        input3 = input3.to(device, dtype=torch.float)
        input4 = input4.to(device, dtype=torch.float)
        input5 = input5.to(device, dtype=torch.float)
        input6 = input6.to(device, dtype=torch.float)

        targets = targets.to(device)

        optimizer.zero_grad()

        output1 = net1(input1)
        output2 = net2(input2)
        output3 = net3(input3)
        output4 = net4(input4)
        output5 = net5(input5)
        output6 = net6(input6)

        inputs = torch.cat([output1,output2,output3,output4,output5,output6], dim=1)

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
        for batch_idx, (input1, input2, input3, input4, input5, input6, targets) in loop:
            input1 = input1.to(device, dtype=torch.float)
            input2 = input2.to(device, dtype=torch.float)
            input3 = input3.to(device, dtype=torch.float)
            input4 = input4.to(device, dtype=torch.float)
            input5 = input5.to(device, dtype=torch.float)
            input6 = input6.to(device, dtype=torch.float)

            targets = targets.to(device)

            output1 = net1(input1)
            output2 = net2(input2)
            output3 = net3(input3)
            output4 = net4(input4)
            output5 = net5(input5)
            output6 = net6(input6)

            inputs = torch.cat([output1,output2,output3,output4,output5,output6], dim=1)

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
        for batch_idx, (input1, input2, input3, input4, input5, input6, targets) in loop:
            input1 = input1.to(device, dtype=torch.float)
            input2 = input2.to(device, dtype=torch.float)
            input3 = input3.to(device, dtype=torch.float)
            input4 = input4.to(device, dtype=torch.float)
            input5 = input5.to(device, dtype=torch.float)
            input6 = input6.to(device, dtype=torch.float)

            targets = targets.to(device)

            output1 = net1(input1)
            output2 = net2(input2)
            output3 = net3(input3)
            output4 = net4(input4)
            output5 = net5(input5)
            output6 = net6(input6)

            inputs = torch.cat([output1,output2,output3,output4,output5,output6], dim=1)

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
