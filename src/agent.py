from pathlib import Path
from typing import Any, Optional, Tuple
from dataset import Batch

import torch
from torch.distributions.categorical import Categorical
import torch.nn as nn
import torch.nn.functional as F
from einops import pack, unpack, repeat, reduce, rearrange

from models.actor_critic import ActorCritic
from models.tokenizer import Tokenizer
from models.world_model import WorldModel
from utils import extract_state_dict
from utils import init_weights, LossWithIntermediateLosses
from models import RT1_state

class Agent(nn.Module):
    def __init__(self, model,loss_weight, use_origin_img = False):
        super().__init__()
        self.use_origin_img = use_origin_img
        self.loss_weight = loss_weight
        self.model = model 

    @property
    def device(self):
        return self.actor_critic.conv1.weight.device

    def __repr__(self) -> str:
        return "agent"
    
    def load(self, path_to_checkpoint: Path, device: torch.device, load_tokenizer: bool = True, load_world_model: bool = True, load_actor: bool = True) -> None:
        agent_state_dict = torch.load(path_to_checkpoint, map_location=device)
        self.load_state_dict(agent_state_dict)
        # if load_tokenizer:
        #     self.tokenizer.load_state_dict(extract_state_dict(agent_state_dict, 'tokenizer'))
        # if load_world_model:
        #     self.world_model.load_state_dict(extract_state_dict(agent_state_dict, 'world_model'))
        # if load_actor:
        #     self.actor.load_state_dict(extract_state_dict(agent_state_dict, 'actor'))

    # def act(self, obs: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
    #     input_ac = obs if self.actor_critic.use_original_obs else torch.clamp(self.tokenizer.encode_decode(obs, should_preprocess=True, should_postprocess=True), 0, 1)
    #     logits_actions = self.actor_critic(input_ac).logits_actions[:, -1] / temperature
    #     act_token = Categorical(logits=logits_actions).sample() if should_sample else logits_actions.argmax(dim=-1)
    #     return act_token
    
    def act(self, batch: torch.FloatTensor, should_sample: bool = True, temperature: float = 1.0) -> torch.LongTensor:
        with torch.no_grad():
            obs = batch['observations']
            states = batch['states']
            texts = batch['instr']

        predict_actions, prediction = self(obs,texts,states)
        return predict_actions

    def update_target_tokenizer(self):
        if self.tokenizer is None:
            return
        for model_param, shadow_param in zip(self.tokenizer.parameters(), self.target_tokenizer.parameters()):
            shadow_param.data = (1.0 - self.momentum) * shadow_param.data + self.momentum * model_param.data
        self.momentum = min(1., self.momentum+self.momentum_delta)

    def forward(self,obs,texts,states):
        actions,prediction = self.model(obs,texts,states)
        return actions, prediction
    
    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:

        obs = batch['observations']
        next_obs = batch['next_observations']
        states = batch['states']
        texts = batch['instr']
        actions = batch['actions']
        # print('states',states)
        # print('actions',actions)
        predict_actions, prediction = self(obs,texts,states)
        # print(predict_actions[:,-1], actions[:,-1],F.binary_cross_entropy_with_logits(predict_actions[:,-1], actions[:,-1]))
        # loss_actions = (F.mse_loss(predict_actions[:,:-1], actions[:,:-1])+F.binary_cross_entropy_with_logits(predict_actions[:,-1], actions[:,-1]))*self.loss_weight[0]
        # print('predict_actions',predict_actions[0, -3 - 2:-2])
        # print('actions',actions[0, -3 - 2:-2])
        loss_actions = (F.mse_loss(predict_actions[:,-3-2:-2], actions[:,-3-2:-2])+F.binary_cross_entropy_with_logits(predict_actions[:,-1], actions[:,-1]))*self.loss_weight[0]
        # loss_actions = (F.binary_cross_entropy_with_logits(predict_actions[:,-1], actions[:,-1]) ) * self.loss_weight[0]
        loss_observation=loss_actions*0
        if prediction is not None:
            training=self.training
            self.eval()
            with torch.no_grad():
                next_token,text_token = self.model(next_obs,texts,states,return_embed=True)
            loss_observation = F.mse_loss(prediction, next_token)*self.loss_weight[1]
            
            if training:
                self.train()
        return LossWithIntermediateLosses(loss_actions=loss_actions,loss_observation=loss_observation)