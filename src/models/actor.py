from dataclasses import dataclass
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, unpack, repeat, reduce, rearrange
from dataset import Batch
from utils import init_weights, LossWithIntermediateLosses
from .robotic_transformer_pytorch import LayerNorm

class actor(nn.Module):
    def __init__(self,input_dims,output_dims,hidden_dims,dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            LayerNorm(input_dims),
            nn.Linear(input_dims, hidden_dims),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims, output_dims),
        )

    def __repr__(self) -> str:
        return "actor"
    
    def forward(self,x):
        x=self.mlp(x)
        return x
    
    def compute_loss(self, batch: Batch, world_model, **kwargs: Any) -> LossWithIntermediateLosses:
        with torch.no_grad():
            obs = batch['observations']
            states = rearrange(batch['states'], 'b s-> b 1 s')
            texts = batch['instr']
            actions = batch['actions']
            predict_token,pooled = world_model(obs,texts,states)

        predict_actions = self(pooled)
        loss_actions = F.mse_loss(predict_actions, actions)
        return LossWithIntermediateLosses(loss_actions=loss_actions)