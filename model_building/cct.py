from torch import nn

from utils.embedding import PositionalEmbedding
from utils.tokenizer import Tokenizer
from utils.transformer import TransformerClassifier


class CCT(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 n_input_channels=3,
                 n_conv_layers=2,
                 kernel_size=7,
                 stride=2,
                 padding=3,
                 pooling_kernel_size=3,
                 pooling_stride=2,
                 pooling_padding=1,
                 num_layers=7,
                 num_heads=4,
                 mlp_ratio=2,
                 dropout=0.1,
                 n_classes=102):
        super().__init__()
        
        self.tokenizer = Tokenizer(kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding,
                                   pooling_kernel_size=pooling_kernel_size,
                                   pooling_stride=pooling_stride,
                                   pooling_padding=pooling_padding,
                                   n_conv_layers=n_conv_layers,
                                   n_input_channels=n_input_channels,
                                   n_output_channels=embedding_dim)
        
        self.pe = PositionalEmbedding(embedding_dim)
        
        self.transformer_classifier = TransformerClassifier(embedding_dim=embedding_dim,
                                                            num_layers=num_layers,
                                                            num_heads=num_heads,
                                                            mlp_ratio=mlp_ratio,
                                                            dropout=dropout,
                                                            n_classes=n_classes)
    
    def forward(self, x):
        x = self.pe(self.tokenizer(x))
        x = self.transformer_classifier(x)
        return x
