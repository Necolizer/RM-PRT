import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim) -> None:
        super().__init__()
        self.xyz_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 30),
        )
        self.axangle_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 30),
        )
        self.gripper_branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 10),
            nn.Sigmoid(),
        )
    
    def forward(self, image_embed):
        # image_embed: (batch_size, 512)
        B, O, D = image_embed.shape
        image_embed = image_embed.view(B, O * D)
        xyz = self.xyz_branch(image_embed)
        axangle = self.axangle_branch(image_embed)
        gripper = self.gripper_branch(image_embed)

        xyz = torch.reshape(xyz, (-1, 10, 3))
        axangle = torch.reshape(axangle, (-1, 10, 3))
        gripper = torch.reshape(gripper, (-1, 10, 1))
        # xyz: (batch_size, 10, 3)
        # axangle: (batch_size, 10, 4)
        # gripper: (batch_size, 10, 1)
        return xyz, axangle, gripper