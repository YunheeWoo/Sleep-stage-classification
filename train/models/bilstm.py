import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, seq_length, batch_size, num_classes ,num_layers=1,dropout=0):
        super(BiLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, bidirectional=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size*2, num_classes)
        
    def init_hidden(self, batch_size):
        # (num_layers * num_directions, batch_size, hidden_size)
        hidden = Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_size).cuda())
        cell = Variable(torch.zeros(self.num_layers*2,batch_size,self.hidden_size).cuda())  
        return hidden, cell
        
    def forward(self, x):
        #batch_size = x.size(0)//self.seq_length

        #print(x.shape)

        h0, c0 = self.init_hidden(x.size(0))

        #x = x.reshape(self.batch_size, self.seq_length, -1)

        #c_out = x.view(batch_size, self.seq_length, x.size(-1))

        self.lstm.flatten_parameters()

        out, _ = self.lstm(x, (h0, c0))

        out = self.fc(out[:,2,:])

        #print(f'out shape: {out.shape}')

        return out

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

