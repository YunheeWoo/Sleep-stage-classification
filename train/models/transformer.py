import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from einops.layers.torch import Rearrange, Reduce

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # pe = torch.zeros(max_len, 1, d_model)
        # position = torch.arange(max_len).unsqueeze(1)
        # div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # pe[:, 0, 0::2] = torch.sin(position * div_term)
        # pe[:, 0, 1::2] = torch.cos(position * div_term)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        # x = x + self.pe[:x.size(0)]
        x = x + self.pe
        return self.dropout(x)

class TransformerModel(nn.Module):

    def __init__(self, ntoken: int, d_model: int, nhead: int, d_hid: int, nlayers: int, num_classes: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.d_model = d_model
        # self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=64)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=5)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        #self.decoder = nn.Linear(d_model, num_classes)
        self.decoder = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, num_classes))

    def forward(self, src: Tensor) -> Tensor:
        #src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = output.mean(dim = 1)
        output = self.decoder(output)
        return output