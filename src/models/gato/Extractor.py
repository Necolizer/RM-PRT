import torch.nn as nn
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import transforms
from .models import Gato
from .config import GatoConfig

    
class GatoModel(nn.Module):
    def __init__(self, state_nums, action_nums):
        super().__init__()
        config = GatoConfig.tiny()
        gato = Gato(config)
        self.model=gato
        self.mlp_net=nn.Sequential(nn.Linear(768, 256), nn.ReLU())
        self.state_net=nn.Linear(state_nums, 64)
        self.action_net=nn.Sequential(
            nn.Linear(256+64, action_nums)
        )
        # self.transform=transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225])

    def forward(self, img,instruction,state) -> torch.Tensor:
       imgs = img.squeeze(1)
       state = state.squeeze(1)
       state_token = self.state_net(state)
       token = self.model(img,instruction)
       token = self.mlp_net(token)
       token = torch.cat([token,state_token], dim=1)
       action =self.action_net(token)
       return action