"""
Transformer implementation.
"""

__version__ = '1.0'
__author__ = 'Saul Alonso-Monsalve'
__email__ = "saul.alonso.monsalve@cern.ch"

from torch import Tensor
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class FittingTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,  # number of Transformer encoder layers
                 d_model: int,  # length of the new representation
                 n_head: int,  # number of heads
                 input_size: int,  # size of each item in the input sequence
                 output_size: int,  # size of each item in the output sequence
                 dim_feedforward: int = 512,  # dimension of the feedforward network of the Transformer
                 dropout: float = 0.1  # dropout value
                 ):
        super(FittingTransformer, self).__init__()
        encoder_layers = TransformerEncoderLayer(d_model=d_model,
                                                 nhead=n_head,
                                                 dim_feedforward=dim_feedforward,
                                                 dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)
        self.proj_input = nn.Linear(input_size, d_model)
        self.decoder = nn.Linear(d_model, output_size)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self, init_range=0.1) -> None:
        # weights initialisation
        self.proj_input.bias.data.zero_()
        self.proj_input.weight.data.uniform_(-init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

    def forward(self,
                src: Tensor,
                src_mask: Tensor,
                src_padding_mask: Tensor):
        # linear projection of the input
        src_emb = self.proj_input(src)
        # transformer encoder
        memory = self.transformer_encoder(src=src_emb, mask=src_mask,
                                          src_key_padding_mask=src_padding_mask)
        # dropout
        memory = self.dropout(memory)
        # linear projection of the output
        output = self.decoder(memory) + src[:, :, :3]  # learn residuals for x,y,z
        return output
