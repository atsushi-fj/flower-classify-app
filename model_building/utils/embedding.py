import math
import torch
from torch import nn

class PositionalEmbedding(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 max_len=5000,  # offset
                 freq=10000.):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embedding_dim, 2).float() * 
                        (-math.log(freq) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div)  # Even
        pe[:, 1::2] = torch.cos(position * div)  # Odd
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x
