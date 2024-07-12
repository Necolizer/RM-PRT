import torch.nn as nn
import torch
import pickle
from torchvision import transforms
# from robotic_transformer_pytorch import MaxViT, RT1
from .bcz_model import get_BCZ, get_BCZ_XL

class return2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs[0], inputs[1]
    
class BCZModel(nn.Module):
    def __init__(self, state_nums, action_nums):
        super().__init__()
        
        BCZmodel = get_BCZ()
        self.model=BCZmodel
        self.mlp_net=nn.Sequential(nn.Linear(768, 256), nn.ReLU())
        self.state_net=nn.Linear(state_nums, 64)
        self.action_net=nn.Sequential(
            nn.Linear(256+64, 128),
            nn.Linear(128, action_nums)
        )

        self.transform=transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def forward(self, img,instruction,state) -> torch.Tensor:
        imgs = img.squeeze(1)
        state = state.squeeze(1)
        state_token = self.state_net(state)
        token = self.model(img,instruction)
        token = self.mlp_net(token)
        token = torch.cat([token,state_token], dim=1)
        action =self.action_net(token)
        return action, None