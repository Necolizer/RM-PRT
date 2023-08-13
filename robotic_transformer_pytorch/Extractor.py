from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import gym
import pickle
import pandas as pd
from torchvision import transforms
from robotic_transformer_pytorch import MaxViT, RT1

from utils import *
from gen_data import *

class return2(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs[0], inputs[1]
    
class CustomExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict,action_nums,bins,use_mask=False):
        super().__init__(observation_space, features_dim=1)
        self.database=pd.read_csv('instructions/training.csv')
        self.instructions=get_instructions()
        self.jointsArrange=torch.Tensor(initJointsArrange()).cuda()
        extractors = {}
        total_concat_size = 0
        feature_size = 256
        
        for key, subspace in observation_space.spaces.items():
            if key == "head_rgb":
                vit = MaxViT(
                    num_classes = 1000,
                    dim_conv_stem = 64,
                    dim = 96,
                    dim_head = 32,
                    depth = (2, 2, 5, 2),
                    window_size = 7,
                    mbconv_expansion_rate = 4,
                    mbconv_shrinkage_rate = 0.25,
                    dropout = 0.1,
                    channels=4
                )

                RT1model = RT1(
                    vit = vit,
                    num_actions = action_nums,
                    action_bins = bins,
                    depth = 6,
                    heads = 8,
                    dim_head = 64,
                    cond_drop_prob = 0.2,
                    use_mask=use_mask
                )
                fc = nn.Sequential(nn.Linear(768, feature_size), nn.ReLU())
                self.RT1=RT1model.cuda()
                extractors["head_rgb"] = fc
                # extractors["head_rgb"] = nn.Sequential(return2(), RT1model, fc)

                total_concat_size += feature_size
            if key == "state":
                # for state data we simply pass it through a single linear layer
                # continue
                state_size = subspace.shape[0]-3-7-3
                extractors["state"] = nn.Linear(state_size, 64)
                total_concat_size += 64

        self.extractors = nn.ModuleDict(extractors).cuda()
        print('model device: ',next(RT1model.parameters()).device)
        
        self._features_dim = total_concat_size

        self.transform=transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

    def forward(self, observations) -> torch.Tensor:
        # print(observations)
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == "head_rgb":
                image = observations["head_rgb"].permute((0, 3, 1, 2))
                image = image.unsqueeze(2)
                names=[]
                instructions=[]
                for i in range(observations['instruction'].shape[0]):     
                    index=int(observations['instruction'][i][0].item())
                    instruction=self.database[self.database['id']==index]['instruction'].values[0]
                    """
                    Add LLM here for instruction parsing
                    """
                    instructions.append(instruction)
               
                logits=self.RT1(image,instructions)
                features = extractor(logits) #batch_size*768
                if torch.isnan(features).any():
                    print('features of rgb appear nan')
                    print('image',image)
                encoded_tensor_list.append(features)
            else:
                observations[key][:,:3] /= 2000
                observations[key][:,3:3+21] /= self.jointsArrange[:,1]
                observations[key][:,3+21:3+21+3] /= 100
                inputs = torch.cat((observations[key][:,3:3+7],observations[key][:,-7-4:-4],observations[key][:,-1].reshape(-1,1)),dim=1)
                features=extractor(inputs)
                encoded_tensor_list.append(features)#batch_size*256
                if torch.isnan(features).any():
                    print('features of state appear nan')
                    print('state',observations[key][:,3:])
                    
        return torch.cat(encoded_tensor_list, dim=1)