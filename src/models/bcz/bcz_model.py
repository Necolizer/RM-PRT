'''
Modified version https://github.com/chahyon-ku/bcz-pytorch/blob/master/lib/bcz_model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .classifier_free_guidance_pytorch.t5 import T5Adapter
from .classifier_free_guidance_pytorch.open_clip import OpenClipAdapter
from .film_resnet import film_resnet18, film_resnet34

class BCZModel(torch.nn.Module):
    def __init__(self, vision_encoders, text_models, hid_dim, out_dim, task_embed_std, fusion, model_path=None) -> None:
        super().__init__()
        self.vision_encoders = vision_encoders
        self.text_models = text_models
        self.task_embed_std = task_embed_std
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.fusion = fusion
        self.fc = nn.Linear(self.hid_dim, self.out_dim)
        if model_path is not None:
            self.load_state_dict(torch.load(model_path))

    def embed_texts(self, texts,device):
        # device = self.text_models[0].device
        text_embeds = []
        for text_model in self.text_models:
            text_model.to(device)
            text_embed = text_model.embed_text(texts=texts)
            text_embeds.append(text_embed)

        return torch.cat(text_embeds, dim = -1)

    def forward(self, images, texts):
        
        text_embeds = self.embed_texts(texts,device=images.device)

        # print(text_embeds.shape) #torch.Size([1, 768])

        text_embeds += torch.randn_like(text_embeds) * self.task_embed_std

        # print(text_embeds.shape) #torch.Size([1, 768])

        B = images.shape[0]
        
        embed = self.vision_encoders(images, text_embeds)

        # print(embed.shape) # torch.Size([1, 1, 512])

        embed = embed.view(B, -1)

        embed = self.fc(embed)

        # view: [B, O, C, H, W]
        # task_embed += torch.randn_like(task_embed) * self.task_embed_std
        # processed_images = {}
        # for view_mode, frames in images.items():
        #     view = view_mode.split('_')[0]
        #     if view not in processed_images:
        #         processed_images[view] = []
        #     processed_images[view].append(frames)
        # # modes: V, [B, O, C, H, W]
        # if self.fusion == 'early':
        #     images = [torch.concat(modes, dim=2) for view, modes in processed_images.items()]
        #     # images: V', [B, O, C', H, W]
        # elif self.fusion == 'late':
        #     images = [mode for modes in processed_images.values() for mode in modes]
        #     # images: V, [B, O, C, H, W]
        # else:
        #     raise NotImplementedError
        
        # for i in range(len(images)):
        #     if i == 0:
        #         embed = self.vision_encoders[i](images[i], task_embed)
        #     else:
        #         embed += self.vision_encoders[i](images[i], task_embed)
        # embed: V, [B, O, V, D]
        # embed = torch.sum(embed)
        # print(embed.shape)
        # embed: [B, O, 1, D]
        return embed
    
def get_BCZ():
    text_models = [T5Adapter(None)]
    vision_encoder = film_resnet18(in_channels=3, task_embed_dim=768, film_on=True)
    return BCZModel(vision_encoders=vision_encoder, text_models=text_models, hid_dim=512, out_dim=768, task_embed_std=0., fusion='early')

def get_BCZ_XL():
    text_models = [T5Adapter(None)]
    vision_encoder = film_resnet34(in_channels=3, task_embed_dim=768, film_on=True)
    return BCZModel(vision_encoders=vision_encoder, text_models=text_models, hid_dim=512, out_dim=768, task_embed_std=0., fusion='early')