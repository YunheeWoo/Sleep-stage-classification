import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
from utils import *
from lib import *
import random
from efficientnet_pytorch import EfficientNet

from torch.optim.lr_scheduler import MultiStepLR 
from torch.optim.lr_scheduler import StepLR

import os
import argparse
import sys

from models import *
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser(description='PyTorch Sleep Stage')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--conf', '-c', action='store_true', help='Draw Confusion Matrix')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
draw = True
total_acc = 0

data_path = Path("/home/eslab/wyh/data/img/resize/new_11/")
cnn_checkpoint_name = 'Efficientb0-downsampling_full.pth'
#cnn_checkpoint_name = 'Efficientb0-10class.pth'
cnn_checkpoint_name = 'Efficientb0-10class-F_C_E_11channel_new_TA_back.pth'
#checkpoint_name = cnn_checkpoint_name[:-4] + '_5class_lstm(batch_size=64)_adamw_cosinerestart.pth'
checkpoint_name = cnn_checkpoint_name[:-4] + '-5class_lstm(batch_size=64)_adamw_cosinerestart.pth'
#signals = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "Saturation_40-100"]
signals = ["F3-M2", "F4-M1", "C3-M2", "C4-M1", "E1-M2", "E2-M1", "Saturation_85-100"]

patients = os.listdir(data_path)
#patients = csv2list("/home/eslab/wyh/test_new.csv")

batch_size = 64
cnn_class_num = 10
class_num = 5

print(checkpoint_name)

# Model

print('==> Building model..')

net = EfficientNet.from_pretrained('efficientnet-b0', in_channels=1, num_classes=cnn_class_num)
#net = resnet50_grayscale()
lstm = BiLSTM(input_size=1280, hidden_size=512, seq_length=5, batch_size=batch_size, num_classes=class_num)
#lstm = BiLSTM(input_size=1024, hidden_size=256, seq_length=5, batch_size=batch_size, num_classes=5)

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

    optimizer = optim.AdamW(lstm.parameters())
    scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=25, cycle_mult=1.0, max_lr=1e-3, min_lr=1e-5, warmup_steps=5, gamma=0.8)
    

net.module._fc = Identity()
#net.module.fc = Identity()

#######
#model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
#######
criterion = nn.CrossEntropyLoss()

#weights = [0.1552, 0.2742, 0.1098, 0.2155, 0.2454]
#class_weights = torch.FloatTensor(weights).cuda()
#criterion = nn.CrossEntropyLoss(weight=class_weights)


#conf = [[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
#conf = [[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0],[0,0,0,0,0]]

# Training

def test(epoch):
    global best_acc
    global draw
    global total_acc
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

    total_acc += acc

    print("acc=%.3f, loss=%.5f, correct=%d, total=%d" %(acc, test_loss/total, correct, total))

    if draw == True:
        #draw_conf(conf, checkpoint_name)
        print(conf.tolist())

#patients.sort()

for patient_0 in patients:
    #patient = patient_0[0]
    patient = patient_0
    print(patient)
    testset = SleepDataLSTMEach(data_path / patient, inv=False, color="L", #shuffle=True,
                                transform=transforms.Compose([
                                        #transforms.Resize([224,224]),
                                        transforms.ToTensor(), 
                                        #transforms.Normalize(mean=[0.1551], std=[0.1427]),
                                        transforms.Normalize(mean=[0.5], std=[0.5]),
                                        #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                                ]))

    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=16)

    test(0)

    print(testset.change_cnt)

print(total_acc)

print(total_acc / len(patients))

