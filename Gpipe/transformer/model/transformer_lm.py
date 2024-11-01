import math
import torch
import torch.nn as nn
from typing import Any
import copy

class SequentialWithCustomForward(nn.Sequential):
    def forward(self, input_tensor):
        hidden_states = input_tensor 
        i=0 
        for layer in self:
            output = layer(hidden_states)  # Get the output from the layer
            if isinstance(output, tuple):
                hidden_states = output[0]  # Use the first item if it's a tuple
            else:
                hidden_states = output  # Otherwise, it's a single tensor
        return hidden_states
        
class EmbeddingLayer(nn.Embedding):
    """Wrapped nn.Embedding layer to allow for weight initialization."""

    def __init__(self, ntoken, embedding_dim, initrange):
        super().__init__(ntoken, embedding_dim)
        self.embedding_dim_sqrt = math.sqrt(embedding_dim)

        self.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        return super().forward(src) * self.embedding_dim_sqrt

class PositionalEncodingLayer(nn.Module):
    """PositionalEncoding layer for a given Transformer model."""

    def __init__(self, embedding_dim, dropout=0.1, max_len=5000):
        super(PositionalEncodingLayer, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class FeedForwardLayer(nn.Module):
    """FeedForward layer for a given Transformer model."""

    def __init__(self, hidden_dim, dim_feedforward, activation, dropout) -> None:
        super(FeedForwardLayer, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, dim_feedforward)
        self.activation = activation
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, hidden_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))


class Attention(nn.Module):
    def __init__(self, hidden_dim, nhead, dropout, layer_id) -> None:
        super(Attention, self).__init__()
        self.self_attn = nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout)
        self.on_GPU = False
        self.layer_id = layer_id

    def forward(self, x, attn_mask, key_padding_mask):
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return x

class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        nhead,
        ff_dim=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        layer_norm_eps=1e-5,
        norm_first=False,
        layer_id=-1,
        stage_id=-1
    ):
        super(TransformerDecoderLayer,self).__init__()
        self.self_attn=Attention(hidden_dim,nhead,dropout,layer_id)
        self.norm_first=norm_first
        self.norm1=nn.LayerNorm(hidden_dim,eps=layer_norm_eps)
        self.norm2=nn.LayerNorm(hidden_dim,eps=layer_norm_eps)
        self.dropout=nn.Dropout(dropout)
        self.stage_id=stage_id
        self.layer_id=layer_id
        self.ff_block=FeedForwardLayer(hidden_dim,ff_dim,activation,dropout)

    def forward(self,src,src_mask=None,src_key_padding_mask=None):
        x=src
        if self.norm_first:
            x=x+self._sa_block(self.norm1(x),src_mask,src_key_padding_mask)
            x=self.ff_block(x+self.norm2(x))
        else:
            x=self.norm1(x+self._sa_block(x,src_mask,src_key_padding_mask))
            x=self.norm2(x+self.ff_block(x))
        return x

    def _sa_block(self,x,attn_mask,key_padding_mask):
        x=self.self_attn(x,attn_mask=attn_mask,key_padding_mask=key_padding_mask)
        return self.dropout(x)

class TransformerLM(nn.Sequential):
    def __init__(
        self,
        hidden_dim,
        nhead,
        ff_dim,
        dropout,
        ndecoder,
        norm_first=False,
        use_fp16=False,
        stage_id=-1
    ):
        self.stage_id=stage_id

        layers=[]

        for layer_id in range(ndecoder):
            layers.append(TransformerDecoderLayer(hidden_dim,nhead,ff_dim,dropout,stage_id=self.stage_id,layer_id=layer_id+stage_id*ndecoder))

        super(TransformerLM,self).__init__(*layers)

    def forward(self,inputs,chunk_id=None):
        # print("foward inputs ",inputs)
        outputs=super().forward(inputs)
        return outputs








