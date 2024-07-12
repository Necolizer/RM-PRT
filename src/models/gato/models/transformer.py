import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import GatoConfig
from typing import Dict, Any, Union


class TransformerBlock(nn.Module):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 *args, **kwargs):
        super(TransformerBlock, self).__init__(*args, **kwargs)

        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config

        hidden_size = self.config.layer_width

        self.attention = nn.MultiheadAttention(embed_dim=hidden_size,
                                               num_heads=self.config.num_attention_heads,
                                               dropout=self.config.dropout_rate,
                                               batch_first=True)
        self.dropout = nn.Dropout(self.config.dropout_rate)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, self.config.feedforward_hidden_size),
            nn.GELU(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.feedforward_hidden_size, hidden_size),
            nn.Dropout(self.config.dropout_rate)
        )
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)

    def forward(self, inputs):
        residual = inputs
        x = self.layer_norm1(inputs)
        x = self.attention(x, x, x)[0]
        x = self.dropout(x)
        x = x + residual

        residual = x
        x = self.layer_norm2(inputs)
        x = self.feed_forward(x)
        x = x + residual
        return x

    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({
            'config': self.config.to_dict(),
        })
        return config
