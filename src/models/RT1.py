import torch
import torch.nn as nn

from einops import pack, unpack, repeat, reduce, rearrange
from einops.layers.torch import Rearrange, Reduce

from .robotic_transformer_pytorch import MaxViT, RT1, LayerNorm, FilmEfficientNet
from .robotic_transformer_pytorch import Tokenizer, RTJ
from typing import List, Optional, Callable, Tuple

class RT1State(nn.Module):
    def __init__(self,RT1,state_net,num_actions,action_bins,state_dims,dropout=0.1):
        super().__init__()
        self.RT1=RT1
        self.state_net=state_net
        self.mlp_extractor=nn.Sequential(
            LayerNorm(512),
        )
        self.action_net=nn.Linear(512, num_actions)
    
    def forward(self,img,instruction,state):
        token=self.RT1(img,instruction,state)
        # b=self.state_net(state[:,:7+3])
        # encoded_tensor=torch.cat([token,b], dim=1)
        extractor_tensor=self.mlp_extractor(token)
        action=self.action_net(extractor_tensor)
        return action, None
    
def RT1_state(action_nums=22, bins=50,state_num=3,dropout=0.1):
    efficientnet=FilmEfficientNet('efficientnet-b3')
    
    state_dims=512
    state_net=nn.Sequential(

            nn.Linear(state_num, 128),
            nn.Tanh(),
            # nn.Dropout(dropout),
            nn.Linear(128, state_dims),
            nn.Tanh(),
            # LayerNorm(512),
        )
    rt1= RT1(
                efficientnet = efficientnet,
                num_actions = action_nums,
                action_bins = bins,
                depth = 6,
                heads = 8,
                dim_head = 64,
                cond_drop_prob = 0.2,
                state_network = state_net,
                dropout=dropout
            )
    
    rt1_state=RT1State(rt1,state_net,action_nums,bins,state_dims)
    return rt1_state

