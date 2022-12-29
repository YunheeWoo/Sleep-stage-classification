import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append("/home/eslab/wyh/Sleep-stage-classification/utils")
from utils import *
from lib import *
import random
from efficientnet_pytorch import EfficientNet
#from cosine_annearing_with_warmup import *

from SleepDataLSTMLoader import *

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
from torch.optim.lr_scheduler import CosineAnnealingLR
#from pytorch-cosine-annealing-with-warmup import *

import os
import argparse
import sys

from models import *
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
start_epoch = 1  # start from epoch 0 or last checkpoint epoch
draw = True


#data_path = Path("/home/eslab/wyh/data/img/resize/new_11_newTA_3-2/")
#data_path = Path("/home/eslab/wyh/data/img/resize/11channel/")
data_path = Path("/data/ssd2/medical_image/11channel/")
#cnn_checkpoint_name = 'final_test/Efficientb0-5class-11channel-back_normal-re.pth'
#cnn_checkpoint_name = 'final_test/Efficientb0-5class-11channel-back_small.pth'
cnn_checkpoint_name = 'IEEE_access/EfficientNetB0-5class-11channel-noback_fullsize.pth'
#checkpoint_name = cnn_checkpoint_name[:-4] + '_5class_lstm(batch_size=64)_adamw_cosinerestart.pth'
#checkpoint_name = cnn_checkpoint_name[:-4] + '-5class_lstm(batch_size=64)_adamw_cosinerestart.pth'
checkpoint_name = cnn_checkpoint_name[:-4] + "_lstm.pth"

#checkpoint_name = "test.pth"

batch_size = 10
cnn_class_num = 5
class_num = 5

ntokens = 5
emsize = 1280 
d_hid = 2048 #512 
nlayers = 6 
nhead = 8
dropout = 0.1
bptt = batch_size

print(checkpoint_name)

# Data
print('==> Preparing data..')

trainset = SleepDataLSTMset("/home/eslab/wyh/train_new.csv", data_path, inv=False, color="L", train=False,#shuffle=True,
                            transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    #transforms.Normalize(mean=[0.1551], std=[0.1427]),
                                    transforms.Normalize(mean=[0.5], std=[0.5]),
                            ]))

valset = SleepDataLSTMset("/home/eslab/wyh/valid_new.csv", data_path, inv=False, color="L", #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    #transforms.Normalize(mean=[0.1551], std=[0.1427]),
                                    transforms.Normalize(mean=[0.5], std=[0.5]),
                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ]))

testset = SleepDataLSTMset("/home/eslab/wyh/test_new.csv", data_path, inv=False, color="L", #shuffle=True,
                            transform=transforms.Compose([
                                    transforms.ToTensor(), 
                                    #transforms.Normalize(mean=[0.1551], std=[0.1427]),
                                    transforms.Normalize(mean=[0.5], std=[0.5]),
                                    #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                            ]))

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=16)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=16)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)

# Model

print('==> Building model..')

net = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1, num_classes=cnn_class_num)
#net = resnet50_grayscale()
#lstm = BiLSTM(input_size=1280, hidden_size=512, seq_length=5, batch_size=batch_size, num_classes=class_num)
#lstm = BiLSTM(input_size=1408, hidden_size=512, seq_length=5, batch_size=batch_size, num_classes=class_num)
#lstm = BiLSTM(input_size=1536, hidden_size=512, seq_length=5, batch_size=batch_size, num_classes=class_num)
#lstm = BiLSTM(input_size=2048, hidden_size=512, seq_length=5, batch_size=batch_size, num_classes=class_num)
lstm = BiLSTM(input_size=1280, hidden_size=256, seq_length=5, batch_size=batch_size, num_classes=5)

#lstm = TransformerModel(ntokens, emsize, nhead, d_hid, nlayers, class_num, dropout)

net = net.to(device)
lstm = lstm.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    lstm = torch.nn.DataParallel(lstm)
    cudnn.benchmark = True

cnn_checkpoint = torch.load('./checkpoint/'+cnn_checkpoint_name)
net.load_state_dict(cnn_checkpoint['net'])

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    #cnn_checkpoint = torch.load('./checkpoint/'+cnn_checkpoint_name)
    checkpoint = torch.load('./checkpoint/'+checkpoint_name)
    #print(checkpoint)
    #net.load_state_dict(cnn_checkpoint['net'])
    lstm.load_state_dict(checkpoint['lstm'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    scheduler = checkpoint['scheduler']
    optimizer = checkpoint['optimizer']
    print("best acc: %lf" %(best_acc))
else:
    #optimizer = optim.SGD(lstm.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    #optimizer = optim.SGD(lstm.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-5)

    #scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=30, cycle_mult=1.0, max_lr=0.1, min_lr=0.0001, warmup_steps=5, gamma=0.8)
    #optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    #scheduler = MultiStepLR(optimizer, milestones=[10,20], gamma=0.1)

    # defalut
    optimizer = optim.AdamW(lstm.parameters())
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=25, cycle_mult=1.0, max_lr=1e-3, min_lr=1e-5, warmup_steps=5, gamma=0.8)
    
    """
    optimizer = torch.optim.Adam(
        lstm.parameters(), 
        lr=1e-4,#1e-3,
        betas=(0.9, 0.999),
        weight_decay=0.0001
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=1270, eta_min=1e-5)
    """

net.module._fc = Identity()
#net.module.fc = Identity()

#######
#model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
#######
criterion = nn.CrossEntropyLoss()

#weights = [0.1552, 0.2742, 0.1098, 0.2155, 0.2454]
#class_weights = torch.FloatTensor(weights).cuda()
#criterion = nn.CrossEntropyLoss(weight=class_weights)

# Training
def train(epoch):
    global bptt
    print('\nEpoch: %d' % epoch)
    net.eval()
    lstm.train()
    train_loss = 0
    correct = 0
    total = 0
    loop = tqdm(enumerate(trainloader), total=len(trainloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}{r_bar}')
    for batch_idx, (input0, input1, input2, input3, input4, targets) in loop:
        inputs = torch.cat([input0,input1,input2,input3,input4], dim=1)
        # [batch, 5, channel, width, height]
        # print(f"inputs shape = {inputs.shape}")
        inputs = inputs.reshape(-1, input0.shape[1], input0.shape[2], input0.shape[3])
        # print(f'targets = {targets}')
        # print(f"inputs shape = {inputs.shape}")
        # exit(1)
        inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)
        
        with torch.no_grad():
            c_out = net(inputs)

        #c_out = net(inputs)

        #print(f'cnn_outputs.shape: {c_out.shape}')
        
        r_in = c_out.view(-1, 5, c_out.size(-1))
        #batch_size = input.size(0)//self.sequence_length

        #print(r_in.shape)

        outputs = lstm(r_in)
        
        loss = criterion(outputs, targets)
        #print(targets)
        #####
        #with amp.scale_loss(loss, optimizer) as scaled_loss: 
        #    scaled_loss.backward()
        #####
        optimizer.zero_grad()
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
    global bptt
    net.eval()
    lstm.eval()
    valid_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(valloader), total=len(valloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}{r_bar}')
        for batch_idx, (input0, input1, input2, input3, input4, targets) in loop:
            inputs = torch.cat([input0,input1,input2,input3,input4], dim=1)
            inputs = inputs.reshape(-1, input0.shape[1], input0.shape[2], input0.shape[3])

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)

            c_out = net(inputs)

            r_in = c_out.view(-1, 5, c_out.size(-1))

            outputs = lstm(r_in)

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
            'lstm': lstm.state_dict(),
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
    global draw
    global bptt
    net.eval()
    lstm.eval()
    conf = np.zeros((class_num, class_num),dtype=int)
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        loop = tqdm(enumerate(testloader), total=len(testloader), bar_format='{desc:<10}{percentage:3.0f}%|{bar:10}{r_bar}')
        for batch_idx, (input0, input1, input2, input3, input4, targets) in loop:
            inputs = torch.cat([input0,input1,input2,input3,input4], dim=1)
            inputs = inputs.reshape(-1, input0.shape[1], input0.shape[2], input0.shape[3])
            
            batch_size = inputs.shape[0]

            inputs, targets = inputs.to(device, dtype=torch.float), targets.to(device)

            c_out = net(inputs)
            r_in = c_out.view(-1, 5, c_out.size(-1))
            
            
            outputs = lstm(r_in)

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


for epoch in range(start_epoch, start_epoch+3000):
    train(epoch)
    valid(epoch)
    test(epoch)
    scheduler.step()
