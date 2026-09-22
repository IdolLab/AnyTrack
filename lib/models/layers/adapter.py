import torch
from torch import nn
import timm
import math
from timm.models.layers import Mlp
from mmcls_custom.models.backbones.vrwkv import VRWKV_Block
import torch.nn.functional as F


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)

class VRWKV_adapter(nn.Module):
    def __init__(self,
                 dim=512,
                 depth=1,
                 ):
        super().__init__()

        # self.tem_resolution = [8, 8]
        # self.adapter_down = nn.Linear(768, dim)  # equivalent to 1 * 1 Conv
        # self.act = QuickGELU()
        self.patch_resolution = [16, 16]
        self.VRWKV_Block=VRWKV_Block(
                n_embd=dim,
                n_layer=depth,
                layer_id=0
            )
        # self.dropout = nn.Dropout(0.1)
        # self.adapter_up = nn.Linear(dim, 768*2)

    def forward(self, x_down):
        # x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        # x_down = self.act(x_down)
        # x_tem = x_down[:, :64]
        # x_patch = x_down[:, 64:]
        # x_tem = self.VRWKV_Block(x_tem, self.tem_resolution)
        x_down = self.VRWKV_Block(x_down, self.patch_resolution)
        # x_down = torch.cat([x_tem, x_patch], dim=1)
        # x_down = self.act(x_down)
        # x_down = self.dropout(x_down)
        # x_up = self.adapter_up(x_down)

        return x_down

class MoE_adapter(nn.Module):
    def __init__(self, dim=512, compress_dim=32, top_k=1, interaction_layer=2):
        super().__init__()
        # self.act = QuickGELU()
        self.adapter_down = nn.Linear(dim * 2, compress_dim)  # equivalent to 1 * 1 Conv
        self.router = NoisyTopkRouter(n_embed=compress_dim, num_experts=4, top_k=top_k)
        self.MLP_adapter = MLP_adapter(dim=compress_dim)
        self.Mamba_adapter = Mamba_adapter(dim=compress_dim)
        self.Convpass_adapter = Convpass_adapter(dim=compress_dim)
        self.VRWKV_adapter = VRWKV_adapter(dim=compress_dim)
        self.adapter_up = nn.Linear(compress_dim, dim * 2)
        self.embed_dim = dim
        self.interaction_layer = interaction_layer
    def forward(self, x_r, x_x):
        x = torch.cat([x_r, x_x], dim=-1)
        x_down = self.adapter_down(x)  # equivalent to 1 * 1 Conv
        l_aux_layer = torch.tensor(0.0, device=x_r.device)
        for i in range(self.interaction_layer):
            adapter_weigts, l_aux = self.router(x_down)
            x_MLP_adapter = self.MLP_adapter(x_down)
            x_Mamba_adapter  = self.Mamba_adapter(x_down)
            x_Convpass_adapter = self.Convpass_adapter(x_down)
            x_VRWKV_adapter = self.VRWKV_adapter(x_down)
            x_adapter = torch.stack([x_MLP_adapter, x_Mamba_adapter, x_Convpass_adapter, x_VRWKV_adapter], 2)
            adapter_weigts = adapter_weigts.unsqueeze(-1)
            x_adapter = adapter_weigts * x_adapter
            x_down = torch.sum(x_adapter, dim=2)
            l_aux_layer += l_aux
        x_up = self.adapter_up(x_down)
        x_adap = x_up[:, :, :self.embed_dim]
        xi_adap = x_up[:, :, self.embed_dim:]

        return x_adap, xi_adap, l_aux_layer


