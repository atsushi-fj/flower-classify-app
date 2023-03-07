import torch 
from torch import nn
from torch.nn import functional as F

class TransformerClassifier(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 num_layers=7,
                 num_heads=4,
                 mlp_ratio=2,
                 dropout=0.1,
                 n_classes=102):
        super().__init__()
        dim_feedforward = int(embedding_dim * mlp_ratio)
        self.transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim,
                                       nhead=num_heads,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout,
                                       activation="gelu",
                                       batch_first=True,
                                       norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_encoder_layer,
                                                         num_layers=num_layers)
        
        self.attention_pool = nn.Linear(embedding_dim, 1) 
        self.norm = nn.LayerNorm(embedding_dim)
        self.mlp = nn.Linear(embedding_dim, n_classes)

    def forward(self, x):
        x = self.transformer_encoder(x)
        x = torch.matmul(F.softmax(self.attention_pool(x), dim=1).transpose(-1, -2), x).squeeze(-2)
        x = self.mlp(self.norm(x))
        return x
    