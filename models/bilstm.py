import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, num_classes,num_layers=1, dropout=0):
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def init_hidden(self,batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_size))
        cell = Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_size))
        return hidden, cell
        
    def forward(self, input):
        


