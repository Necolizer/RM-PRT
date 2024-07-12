import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import GatoConfig


def _randomized_positions(from_v, to_v):
    pos = torch.rand_like(from_v)
    pos = pos * (to_v - from_v)
    pos = torch.floor(pos)
    return pos


def _rounded_mean_positions(from_v, to_v):
    pos = (from_v + to_v) / 2
    pos = torch.round(pos)
    return pos


def _broadcast(row_pos, col_pos, row_ones, col_ones):
    row_pos = row_pos.unsqueeze(1)
    row_pos = torch.matmul(row_pos, col_ones.transpose(0, 1))
    row_pos = row_pos.view(-1)
    row_pos = row_pos.detach()

    col_pos = col_pos.unsqueeze(1)
    col_pos = torch.matmul(row_ones, col_pos.transpose(0, 1))
    col_pos = col_pos.view(-1)
    col_pos = col_pos.detach()

    return row_pos, col_pos


class PatchPositionEncoding(nn.Module):
    def __init__(self, config: GatoConfig):
        super(PatchPositionEncoding, self).__init__()
        self.config = config
        self.embedding_dim = self.config.layer_width
        self.discretize_depth = self.config.discretize_depth
        self.patch_size = self.config.img_patch_size

        self.row_embedding = nn.Embedding(self.discretize_depth, self.embedding_dim)
        self.col_embedding = nn.Embedding(self.discretize_depth, self.embedding_dim)

    def _discretize(self, pos):
        return torch.round(pos * self.discretize_depth)

    def _discretize_interval(self, interval):
        pos_from, pos_to = interval
        return self._discretize(pos_from), self._discretize(pos_to)

    def forward(self, inputs):
        # training = inputs[1].get('training', False)
        input_ids, (row_pos, col_pos) = inputs
        row_pos_from, row_pos_to = self._discretize_interval(row_pos)
        col_pos_from, col_pos_to = self._discretize_interval(col_pos)

        # if training:
        #     row_pos = row_pos_from + _randomized_positions(row_pos_from, row_pos_to)
        #     col_pos = col_pos_from + _randomized_positions(col_pos_from, col_pos_to)
        # else:
        row_pos = _rounded_mean_positions(row_pos_from, row_pos_to)
        col_pos = _rounded_mean_positions(col_pos_from, col_pos_to)
        col_pos = col_pos.to(torch.int32)
        row_pos = row_pos.to(torch.int32)
        
        return input_ids + self.row_embedding(row_pos) + self.col_embedding(col_pos)


class ResidualUnit(nn.Module):
    def __init__(self, num_groups, filters):
        super(ResidualUnit, self).__init__()
        self.num_groups = num_groups
        self.filters = filters
        self.gn1 = nn.GroupNorm(self.num_groups, self.filters // 2)
        self.gn2 = nn.GroupNorm(self.num_groups, self.filters)
        self.conv1 = nn.Conv2d(self.filters // 2, self.filters, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.conv2 = nn.Conv2d(self.filters, self.filters, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
        self.conv_proj = nn.Conv2d(self.filters // 2, self.filters, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False)
        self.gn_proj = nn.GroupNorm(self.num_groups, self.filters // 2)

    def forward(self, inputs):
        x = inputs

        residual = self.conv_proj(self.gn_proj(x))

        x = F.gelu(self.gn1(x))
        x = self.conv1(x)

        x = F.gelu(self.gn2(x))
        x = self.conv2(x)

        return x + residual


class ResidualEmbedding(nn.Module):
    def __init__(self, config: GatoConfig):
        super(ResidualEmbedding, self).__init__()
        self.config = config
        self.conv_proj = None
        self.residual_units = None

        if self.config.input_dim != self.config.layer_width:
            self.conv_proj = nn.Conv2d(self.config.input_dim, self.config.layer_width, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False)
        self.root_conv = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.GroupNorm(self.config.num_group_norm_groups, 96),
            nn.GELU()
        )
        self.residual_units = nn.ModuleList([
            ResidualUnit(self.config.num_group_norm_groups, 96 * 2 ** (i + 1))
            for i in range(3)
        ])

    def forward(self, inputs):
        x = self.root_conv(inputs)

        for block in self.residual_units:
            x = block(x)
        if self.conv_proj is not None:
            x = self.conv_proj(x)
        x = x.view(-1, 224, self.config.layer_width)
        return x


class LocalPositionEncoding(nn.Module):
    def __init__(self, config: GatoConfig):
        super(LocalPositionEncoding, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.token_sequence_length, self.config.layer_width)

    def forward(self, inputs):
        obs_pos, obs_mask = inputs
        embed = self.embedding(obs_pos)

        ones = torch.ones(embed.size(0), 1, self.config.layer_width).to(obs_pos.device)
        obs_mask = obs_mask.float()
        
        obs_mask = torch.matmul(obs_mask, ones)
        return embed * obs_mask


class DiscreteEmbedding(nn.Module):
    def __init__(self, config: GatoConfig):
        super(DiscreteEmbedding, self).__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.embedding_input_size, self.config.layer_width)

    def forward(self, inputs):
        return self.embedding(inputs)
