import torch
from torch import Tensor
import torch.nn as nn
#from .utils import load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional
from efficientnet_pytorch import EfficientNet

class mymodel(nn.Module):
    def __init__(self, w, h, c, o):
        super(mymodel,self).__init__()
        self.pretrained_model = EfficientNet.from_pretrained('efficientnet-b4', in_channels=c, num_classes=o)
        self.bias = nn.Parameter(torch.ones(c,h,w))

    def forward(self, x):
        out = x + self.bias
        out = self.pretrained_model(out)
        return out

class mymodel2(nn.Module):
    def __init__(self):
        super(mymodel2,self).__init__()
        self.pretrained_model = EfficientNet.from_pretrained('efficientnet-b4', in_channels=1, num_classes=5)
        self.bias_eeg = nn.Parameter(torch.ones(1))
        self.bias_eog = nn.Parameter(torch.ones(1))
        self.bias_emg = nn.Parameter(torch.ones(1))
        self.bias_ecg = nn.Parameter(torch.ones(1))
        self.bias_flow = nn.Parameter(torch.ones(1))
        self.bias_Thorax = nn.Parameter(torch.ones(1))
        self.bias_Abdomen = nn.Parameter(torch.ones(1))

    def forward(self, x):
        out1 = x + self.bias_eeg
        out2 = x + self.bias_eog
        out3 = x + self.bias_emg
        out4 = x + self.bias_ecg
        out5 = x + self.bias_flow
        out6 = x + self.bias_Thorax
        out7 = x + self.bias_Abdomen

        out = torch.zeros(x.shape).cuda()

        out[:, :69, :] = out1[:, :69, :]
        out[:, 69:103, :] = out2[:, 69:103, :]
        out[:, 103:120, :] = out3[:, 103:120, :]
        out[:, 120:137, :] = out4[:, 120:137, :]
        out[:, 137:189, :] = out5[:, 137:189, :]
        out[:, 189:207, :] = out6[:, 189:207, :]
        out[:, 207:, :] = out7[:, 207:, :]

        out = self.pretrained_model(out)
        return out