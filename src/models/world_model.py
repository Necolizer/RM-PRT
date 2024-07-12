from dataclasses import dataclass
from typing import Any, Optional, Tuple

from einops import pack, unpack, repeat, reduce, rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Batch
from .kv_caching import KeysValues
from .slicer import Embedder, Head
from .tokenizer import Tokenizer
from .transformer import Transformer, TransformerConfig
from utils import init_weights, LossWithIntermediateLosses
from .robotic_transformer_pytorch import TextEmbedder

@dataclass
class WorldModelOutput:
    output_sequence: torch.FloatTensor
    logits_observations: torch.FloatTensor
    logits_actions: torch.FloatTensor
    # logits_rewards: torch.FloatTensor
    # logits_ends: torch.FloatTensor

class Embedder(nn.Module):
    def __init__(self, obs_embedder, state_embedder, text_embedder):
        super().__init__()
        self.obs_embedder = obs_embedder
        self.state_embedder = state_embedder
        self.text_embedder = text_embedder
        
    
    def forward(self,obs,states,texts):
        obs = self.obs_embedder(obs)
        states = self.state_embedder(states)
        text_embed = self.text_embedder(texts)
        return torch.concat([obs,states,text_embed],axis=-2)
        
class WorldModel(nn.Module):
    def __init__(self, obs_vocab_size: int, act_vocab_size: int, state_size:int, config: TransformerConfig, loss_weight: float) -> None:
        super().__init__()
        self.obs_vocab_size, self.act_vocab_size = obs_vocab_size, act_vocab_size
        self.config = config
        self.transformer = Transformer(config)
        self.loss_weight = loss_weight

        all_but_last_obs_tokens_pattern = torch.ones(config.tokens_per_block)
        all_but_last_obs_tokens_pattern[-2] = 0
        act_tokens_pattern = torch.zeros(self.config.tokens_per_block)
        act_tokens_pattern[-1] = 1
        obs_tokens_pattern = 1 - act_tokens_pattern

        self.pos_emb = nn.Embedding(config.max_tokens, config.embed_dim)
        # self.embedder = nn.Embedding(obs_vocab_size, config.embed_dim)
        self.embedder = Embedder(
            obs_embedder = nn.Embedding(obs_vocab_size, config.embed_dim),
            state_embedder = nn.Linear(state_size, config.embed_dim),
            text_embedder = TextEmbedder(text_embed_stem_dim=config.embed_dim)
        )

        self.head_observations = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, obs_vocab_size)
            )
        

        self.head_actions = nn.Sequential(
                nn.Linear(config.embed_dim, config.embed_dim),
                nn.ReLU(),
                nn.Linear(config.embed_dim, act_vocab_size)
            )

        # self.head_ends = Head(
        #     max_blocks=config.max_blocks,
        #     block_mask=act_tokens_pattern,
        #     head_module=nn.Sequential(
        #         nn.Linear(config.embed_dim, config.embed_dim),
        #         nn.ReLU(),
        #         nn.Linear(config.embed_dim, 2)
        #     )
        # )

        self.apply(init_weights)

    def __repr__(self) -> str:
        return "world_model"

    def forward(self, tokens: torch.LongTensor, text: str, states: torch.FloatTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:

        num_steps = tokens.size(1)  # (B, T)
        assert num_steps <= self.config.max_tokens
        prev_steps = 0 if past_keys_values is None else past_keys_values.size

        # sequences = self.embedder(tokens, num_steps, prev_steps) + self.pos_emb(prev_steps + torch.arange(num_steps, device=tokens.device))
        sequences = self.embedder(tokens,states,text)
        x = self.transformer(sequences, past_keys_values)
        logits_observations = self.head_observations(x[:,:-2])
        # x = reduce(x, 'b n d -> b d', 'mean')
        logits_actions = self.head_actions(x[:,-2])
        # logits_ends = self.head_ends(x, num_steps=num_steps, prev_steps=prev_steps)

        return WorldModelOutput(x, logits_observations, logits_actions)

    def compute_loss(self, batch: Batch, tokenizer: Tokenizer, **kwargs: Any) -> LossWithIntermediateLosses:

        with torch.no_grad():
            obs_tokens = tokenizer.encode(batch['observations'], should_preprocess=True).tokens  
            next_obs_tokens = tokenizer.encode(batch['next_observations'], should_preprocess=True).tokens
            obs_tokens = rearrange(obs_tokens, 'b t k -> (b t) k')
            next_obs_tokens = rearrange(next_obs_tokens, 'b t k -> (b t) k')
            states = rearrange(batch['states'], 'b s-> b 1 s')
            text = batch['instr']

        # tokens = rearrange(torch.cat((obs_tokens, act_tokens), dim=2), 'b l k1 -> b (l k1)')  # (B, L(K+1))
        tokens = obs_tokens
        outputs = self(tokens,text,states)
        logits_observations = rearrange(outputs.logits_observations, 'bt k d -> (bt k) d')
        labels_observations = rearrange(next_obs_tokens, 'bt k -> (bt k)')
        loss_obs = F.cross_entropy(logits_observations, labels_observations)*self.loss_weight[0]
        
        logits_actions = outputs.logits_actions
        labels_actions = batch['actions']
        loss_actions = F.mse_loss(logits_actions, labels_actions)*self.loss_weight[1]
        # loss_rewards = F.cross_entropy(rearrange(outputs.logits_rewards, 'b t e -> (b t) e'), labels_rewards)
        # loss_ends = F.cross_entropy(rearrange(outputs.logits_ends, 'b t e -> (b t) e'), labels_ends)

        return LossWithIntermediateLosses(loss_obs=loss_obs,loss_actions=loss_actions)

    def compute_labels_world_model(self, obs_tokens: torch.Tensor,  ends: torch.Tensor=None, mask_padding: torch.BoolTensor=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # assert torch.all(ends.sum(dim=1) <= 1)  # at most 1 done
        mask_fill = torch.logical_not(mask_padding)
        labels_observations = rearrange(obs_tokens, 'b t k -> b (t k)')[:, 1:] # obs_tokens.masked_fill(mask_fill.unsqueeze(-1).expand_as(obs_tokens), -100)
        # labels_rewards = (rewards.sign() + 1).masked_fill(mask_fill, -100).long()  # Rewards clipped to {-1, 0, 1}
        # labels_ends = ends.masked_fill(mask_fill, -100)
        return labels_observations.reshape(-1)

@dataclass
class WorldModelPredictiveOutput:
    token: torch.FloatTensor
    pooled: torch.FloatTensor

class WorldModelPredictive(nn.Module):
    def __init__(self, tokenizer, target_tokenizer, network, loss_weight: float) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.target_tokenizer = target_tokenizer
        self.momentum = 0.996
        self.momentum_delta = 0.0001
        self.network = network
        self.loss_weight = loss_weight

    def __repr__(self) -> str:
        return "world_model"

    def forward(self, obs: torch.FloatTensor, texts: str, states: torch.FloatTensor, past_keys_values: Optional[KeysValues] = None) -> WorldModelOutput:
        token = self.tokenizer(obs,texts)
        predict_token = self.network(token,states)
        predict_token = predict_token[:,:-1]
        pooled = reduce(predict_token, 'b fn d -> b d', 'mean')
        return predict_token,pooled

    def compute_loss(self, batch: Batch, **kwargs: Any) -> LossWithIntermediateLosses:

        obs = batch['observations']
        next_obs = batch['next_observations']
        states = rearrange(batch['states'], 'b s-> b 1 s')
        texts = batch['instr']

        predict_token,pooled = self(obs,texts,states)
        self.target_tokenizer.eval()
        with torch.no_grad():
            next_token = self.target_tokenizer(next_obs,texts)
        loss_observation = F.mse_loss(predict_token, next_token)*self.loss_weight[0]
        return LossWithIntermediateLosses(loss_observation=loss_observation)
