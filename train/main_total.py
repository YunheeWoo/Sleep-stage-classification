'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

from torch.optim.lr_scheduler import MultiStepLR 

import os
import argparse

from utils import progress_bar
import numpy as np

from resnetdual import ResNetDual, ResNetSig
from CustomDataLoader import CustomDataset, SameOrderDataset

#torch.backends.cudnn.enabled = False

parser = argparse.ArgumentParser(description='Sleep Stage Classification')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',help='resume from checkpoint')
parser.add_argument('--ckpt', '-c', type=str, required=True, help='Checkpoint file name')
parser.add_argument('--batch', '-b', default=20, type=int, help='Batch size')
parser.add_argument('--conf', '-f', action='store_true', help='Test for confusion matrix')
parser.add_argument('--data', '-d', type=int, required=True, help='Type of data which want to training')
parser.add_argument('--multi', '-m', action='store_true', help='Use multi type of data')
parser.add_argument('--signal', '-s', action='store_true', help='Use raw data')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
chg_lr1 = 15
chg_lr2 = 30
data_flag = None
batch_size = args.batch

if args.data == 0:
    folder = "Fpz-Cz"
elif args.data == 1:
    folder = "Pz-Oz"
elif args.data == 2:
    folder = "EOG"
elif args.data == 3:
    folder = "Combine_Fpz_EOG"
elif args.data == 4:
    folder = "Combine"

if args.multi:
    print("Training model with Image and Signal data")
    data_flag = 2
elif args.signal:
    print("Training model with Signal data")
    data_flag = 1
else:
    print("Training model with Image data")
    data_flag = 0

# Data
print('==> Preparing data..')

img_path = '/home/wyh/medical_project/data/img/'

sig_path = '/home/wyh/medical_project/data/npy/'

ann_dir = '/home/wyh/medical_project/data/annotations'

if not data_flag==1: 
    trainset_img = ImageFolder(root=img_path+folder+'/train',transform=transforms.Compose([
                                   transforms.ToTensor(), 
                                   transforms.Normalize(mean=[0.9755, 0.9819, 0.9867],
                                         std=[0.1405, 0.1195, 0.1016])
                               ]))
    
    valset_img = ImageFolder(root=img_path+folder+'/val',transform=transforms.Compose([
                                   transforms.ToTensor(), 
                                   transforms.Normalize(mean=[0.9755, 0.9819, 0.9867],
                                         std=[0.1405, 0.1195, 0.1016])
                               ]))

    
    
if not data_flag==0:
    trainset_sig = CustomDataset(sig_path+folder+'/train', ann_dir+'/train')
    valset_sig = CustomDataset(sig_path+folder+'/val', ann_dir+'/val')

# Model
print('==> Building model..')
if data_flag==0:
    trainloader = torch.utils.data.DataLoader(trainset_img, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset_img, batch_size=batch_size, shuffle=True, num_workers=2)
    
    net = torch.hub.load('pytorch/vision', 'resnet18', pretrained=False)
    net.fc = nn.Linear(512,5)
    
elif data_flag==1:
    trainloader = torch.utils.data.DataLoader(trainset_sig, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset_sig, batch_size=batch_size, shuffle=True, num_workers=2)
    
    net = ResNetSig()

elif data_flag==2:
    trainset = SameOrderDataset(trainset_img, trainset_sig)
    valset = SameOrderDataset(valset_img, valset_sig)
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    net = ResNetDual()

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+args.ckpt)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']+1
    
    chg_lr1 = np.maximum(0, chg_lr1-start_epoch)
    chg_lr2 = chg_lr2-start_epoch
    
    

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
scheduler = MultiStepLR(optimizer, milestones=[chg_lr1,chg_lr2], gamma=0.1)

conf = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

# Training
def train_img(epoch):
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
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def val_img(epoch):
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
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+'best_'+args.ckpt)
        best_acc = acc
        
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }    
    torch.save(state, './checkpoint/'+args.ckpt)
    
# Training
def train_sig(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        targets = targets.squeeze(1)
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def val_sig(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            targets = targets.squeeze(1)
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
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+'best_'+args.ckpt)
        best_acc = acc
        
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }    
    torch.save(state, './checkpoint/'+args.ckpt)
    
# Training
def train_dual(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, ((inputs_img, targets),(inputs_sig,targets_sig)) in enumerate(trainloader):
        inputs_img, inputs_sig, targets = inputs_img.to(device), inputs_sig.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs_img, inputs_sig)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))


def val_dual(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, ((inputs_img, targets),(inputs_sig,_)) in enumerate(valloader):
            inputs_img, inputs_sig, targets = inputs_img.to(device), inputs_sig.to(device), targets.to(device)
            outputs = net(inputs_img, inputs_sig)
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
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/'+'best_'+args.ckpt)
        best_acc = acc
        
    state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }    
    torch.save(state, './checkpoint/'+args.ckpt)
    
def conf_img(epoch):
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
            item = predicted.to('cpu').numpy()
            ans = targets.to('cpu').numpy()
            for item_idx, c in enumerate(item):
                conf[ans[item_idx]][item[item_idx]] += 1
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                         
def conf_sig(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(valloader):
            targets = targets.squeeze(1)
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

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                         
def conf_dual(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, ((inputs_img, targets),(inputs_sig,_)) in enumerate(valloader):
            inputs_img, inputs_sig, targets = inputs_img.to(device), inputs_sig.to(device), targets.to(device)
            outputs = net(inputs_img, inputs_sig)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            item = predicted.to('cpu').numpy()
            ans = targets.to('cpu').numpy()
            for item_idx, c in enumerate(item):
                conf[ans[item_idx]][item[item_idx]] += 1
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(valloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
                         
if args.conf:
    epoch = 0
    if data_flag==0:
        conf_img(epoch)
    if data_flag==1:
        conf_sig(epoch)
    if data_flag==2:
        conf_dual(epoch)
    
    print(conf)
    
else:
    if data_flag==0:
        for epoch in range(start_epoch, 200):
            train_img(epoch)
            val_img(epoch)
            
    if data_flag==1:
        for epoch in range(start_epoch, 200):
            train_sig(epoch)
            val_sig(epoch)
    
    if data_flag==2:
        for epoch in range(start_epoch, 200):
            train_dual(epoch)
            val_dual(epoch)

