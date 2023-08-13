import torch
import torch.nn as nn
import torch.nn.functional as F
from gato.models.transformer import TransformerBlock
from gato.models.embedding import PatchPositionEncoding, ResidualEmbedding, LocalPositionEncoding, DiscreteEmbedding
# from gato.models.tokenizers import ContinuousValueTokenizer
from gato import GatoConfig
from typing import Dict, Any, Union
from classifier_free_guidance_pytorch.t5 import T5Adapter


class Gato(nn.Module):

    def __init__(self, config: Union[GatoConfig, Dict[str, Any]], **kwargs):
        super(Gato, self).__init__(**kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.image_embedding = PatchEmbedding(config)
        self.discrete_embedding = DiscreteEmbedding(config)
        # self.continuous_encoding = ContinuousValueTokenizer(config)
        self.transformer = Transformer(config)
        self.local_pos_encoding = LocalPositionEncoding(config)
        self.img_conv = nn.Conv2d(4, 3, (1,1))

        self.text_models = [T5Adapter(None)]

        self.row_pos = (
        torch.tensor([[0/16, 1/16, 2/16, 3/16, 4/16, 5/16, 6/16, 7/16, 8/16, 9/16, 10/16, 11/16, 12/16, 13/16, 14/16, 15/16]*16], requires_grad=False).cuda(),  # pos_from
        torch.tensor([[1/16, 2/16, 3/16, 4/16, 5/16, 6/16, 7/16, 8/16, 9/16, 10/16, 11/16, 12/16, 13/16, 14/16, 15/16, 1.00]*16], requires_grad=False).cuda()   # pos_to
        )
        self.col_pos = (
        torch.tensor([[0/16, 1/16, 2/16, 3/16, 4/16, 5/16, 6/16, 7/16, 8/16, 9/16, 10/16, 11/16, 12/16, 13/16, 14/16, 15/16]*16], requires_grad=False).cuda(),  # pos_from
        torch.tensor([[1/16, 2/16, 3/16, 4/16, 5/16, 6/16, 7/16, 8/16, 9/16, 10/16, 11/16, 12/16, 13/16, 14/16, 15/16, 1.00]*16], requires_grad=False).cuda()   # pos_to
        )

        self.obs_pos = torch.linspace(0,256,steps=257,requires_grad=False).int().unsqueeze(0).cuda()
        self.obs_mask = torch.ones(257, requires_grad=False).int().unsqueeze(1).cuda()

    def embed_texts(self, texts):
        # device = self.text_models[0].device
        text_embeds = []
        for text_model in self.text_models:
            text_embed = text_model.embed_text(texts=texts)
            text_embeds.append(text_embed.cuda())

        return torch.cat(text_embeds, dim = -1)

    def forward(self, images, texts):
        #images torch.Size([1, 4, 1, 256, 256])
        # input_ids, (encoding, row_pos, col_pos), (obs_pos, obs_mask) = inputs

        # encoding = torch.tensor([0]*16+[2])
        # encoding = F.one_hot(encoding, num_classes=3).float()

        # ones = torch.ones((input_ids.shape[0], 1, self.config.layer_width)).float()

        # print(images.shape) # torch.Size([1, 1, 4, 256, 256])
        images = images.permute(0,2,1,3,4) # torch.Size([1, 1, 4, 256, 256])
        images = images.squeeze(1) # torch.Size([1, 4, 256, 256])
        images = self.img_conv(images) #torch.Size([1, 3, 256, 256])
        # images = images.view(images.shape[0] * 16 * 16, images.shape[0], 16, 16, 3)
        # print(images.shape)
        image_embed = self.image_embedding((images, (self.row_pos, self.col_pos)))
        # print(image_embed.shape) #torch.Size([1, 256, 768])

        # image_embed *= torch.matmul(encoding[..., 0], ones.transpose(1, 2))

        # continuous_embed = self.continuous_encoding(input_ids[..., 0])
        # continuous_embed = self.discrete_embedding(continuous_embed)
        # continuous_embed *= torch.matmul(encoding[..., 1], ones.transpose(1, 2))

        # discrete_embed = self.discrete_embedding(input_ids[..., 0])
        # discrete_embed *= torch.matmul(encoding[..., 2], ones.transpose(1, 2))
        text_embeds = self.embed_texts(texts).unsqueeze(1)

        # print(text_embeds.shape) #torch.Size([1, 1, 768])

        # embed = image_embed + continuous_embed + discrete_embed
        embed = torch.cat([image_embed, text_embeds], dim=1)
        embed += self.local_pos_encoding((self.obs_pos, self.obs_mask))

        hidden_states = self.transformer(embed)
        hidden_states = torch.mean(hidden_states, dim=1)
        # print(hidden_states.shape) # torch.Size([1, 768])
        return hidden_states


class Transformer(nn.Module):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 **kwargs):
        super(Transformer, self).__init__(**kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.encoders = nn.ModuleList([
            TransformerBlock(config=self.config)
            for idx in range(self.config.num_transformer_blocks)
        ])

    def forward(self, inputs):
        x = inputs
        for encoder in self.encoders:
            x = encoder(x)
        return x


class PatchEmbedding(nn.Module):

    def __init__(self,
                 config: Union[GatoConfig, Dict[str, Any]],
                 **kwargs):
        super(PatchEmbedding, self).__init__(**kwargs)
        if isinstance(config, dict):
            config = GatoConfig(**config)
        self.config = config
        self.residual_embedding = ResidualEmbedding(config)
        self.pos_encoding = PatchPositionEncoding(config)

    def forward(self, inputs):
        input_ids, (row_pos, col_pos) = inputs
        patch_size = self.config.img_patch_size
        depth = self.config.input_dim // (patch_size * patch_size)

        x = input_ids.contiguous().view(-1, depth, patch_size, patch_size)
        # x = input_ids.view(input_ids.shape[0], depth, (input_ids.shape[2] // patch_size), patch_size, (input_ids.shape[3] // patch_size), patch_size)
        # x = x.permute(0, 2, 4, 1, 3, 5)
        # x = x.view(-1, depth, patch_size, patch_size)
        x = self.residual_embedding(x)
        x = self.pos_encoding((x, (row_pos, col_pos)))
        return x
