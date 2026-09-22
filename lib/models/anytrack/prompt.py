import copy
from typing import Any, Optional, Tuple, Type

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import pdb
import torch.nn as nn
from itertools import repeat
import numpy as np
from lib.models.anytrack.transformer_prompt_weight import TwoWayTransformer_rep_weight
from lib.utils.pos_embed import get_sinusoid_encoding_table
class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
        self, coords_input: torch.Tensor
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        coords = coords_input.clone()
        # coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        # coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

class PromptEncoder_traj_rep_weight(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        tracklen: int,
        log=False,
            trunc=False,
            alpha=7.23
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)

        self.num_point_embeddings: int = 2  # 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)

        # rep_tokens = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        # self.rep_tokens = nn.ModuleList(rep_tokens)#0 for iou_pred(weight),1-2 for corner points

        # self.tpe = nn.Embedding(tracklen, embed_dim)
        # self.tpe = nn.ModuleList(tpe)

        self.tracklen=tracklen
        # self.pos_embed = nn.Parameter(torch.zeros(1, self.tracklen, embed_dim))
        self.alpha=alpha
        # if log==-1 or log==-5:
        #     print('tpe using log=',log)
        #     pos_embed = get_sinusoid_encoding_table(self.tracklen, self.pos_embed.shape[-1], cls_token=False,log=5,alpha=self.alpha)
        # elif log==-2:
        #     print('tpe using log=', log)
        #     pos_embed = get_sinusoid_encoding_table(self.tracklen, self.pos_embed.shape[-1], cls_token=False, log=1)
        # elif log==-3:
        #     print('tpe using log=', log)
        #     pos_embed = get_sinusoid_encoding_table(self.tracklen, self.pos_embed.shape[-1], cls_token=False, log=-1)
        # else:
        #     pos_embed = get_sinusoid_encoding_table(self.tracklen, self.pos_embed.shape[-1], cls_token=False, log=log)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # self.not_a_point_embed = nn.Embedding(1, embed_dim)
        self.log=log
        self.trunc=trunc
    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        bs,t,_=boxes.shape
        """Embeds box prompts."""
        # boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(bs,t, 2, 2)
        if self.trunc:
            coords=torch.max(coords, torch.tensor([0.]).to(coords)) # Truncate out-of-range values
            coords = torch.min(coords, torch.tensor([1.]).to(coords))
        corner_embedding = self.pe_layer.forward_with_coords(coords)
        corner_embedding[:,:, 0, :] += self.point_embeddings[0].weight.unsqueeze(0).expand(bs,t,-1)
        corner_embedding[:,:, 1, :] += self.point_embeddings[1].weight.unsqueeze(0).expand(bs,t,-1)
        return corner_embedding
    # def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
    #     """Embeds box prompts."""
    #     # boxes = boxes + 0.5  # Shift to center of pixel
    #     coords = boxes.reshape(-1, 2, 2)
    #     corner_embedding = self.pe_layer.forward_with_coords(coords)
    #     corner_embedding[:, 0, :] += self.point_embeddings[0].weight
    #     corner_embedding[:, 1, :] += self.point_embeddings[1].weight
    #     return corner_embedding
    def get_dense_pe(self,image_embedding_size) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(image_embedding_size).unsqueeze(0)


    def _get_batch_size(
        self,
        boxes: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if boxes is not None:
            return boxes.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
        self,boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments:
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(boxes)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
        # tokens = torch.cat((box_embeddings.reshape(bs, -1, self.embed_dim),
        #                     self.rep_tokens[0].weight.expand(bs, -1, -1)+self.point_embeddings[0].weight.unsqueeze(0).expand(bs,-1,-1),
        #                     self.rep_tokens[1].weight.expand(bs, -1, -1)+self.point_embeddings[1].weight.unsqueeze(0).expand(bs,-1,-1)
        #                     ), dim=1)
        tokens = box_embeddings.reshape(bs, -1, self.embed_dim)
        return tokens

class PromptDecoder_traj_reg_rep_weight(nn.Module):
    # use two representative tokens + enc_point_embeddings(tl/br)
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            iou_head_hidden_dim=256,
            iou_head_depth=2
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        # self.token_head = MLP(
        #     transformer_dim, iou_head_hidden_dim, 2, iou_head_depth, sigmoid_output=False
        # )
        # self.weight_head = MLP(
        #     transformer_dim, iou_head_hidden_dim, 1, iou_head_depth, sigmoid_output=False
        # )
    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            sparse_prompt_pe: torch.Tensor,
    ):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        '''todo check dims; add iou_token(merge box head?)'''
        # bs, t_traj, t_p, c = sparse_prompt_embeddings.shape

        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        prompt_tokens, src_new, attn = self.transformer(src, pos_src, sparse_prompt_embeddings, sparse_prompt_pe)
        # rep_tokens=prompt_tokens[:,-2:,:]
        # token_pred = self.token_head(rep_tokens.reshape(-1,c))#[bs*2,2]
        # token_pred=token_pred.reshape(b,-1,2)
        return prompt_tokens, src_new



class PromptDecoder_traj_implicit(nn.Module):
    # no iou tokens
    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: nn.Module,
            iou_head_hidden_dim=256,
            iou_head_depth=2
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer

        self.iou_token = nn.Embedding(1, transformer_dim)
        self.iou_prediction_head = MLP(
            transformer_dim, transformer_dim, transformer_dim, iou_head_depth,sigmoid_output=True
        )

    def forward(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
            sparse_prompt_pe: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        '''todo check dims; add iou_token(merge box head?)'''
        bs, t_traj, t_p, c = sparse_prompt_embeddings.shape
        output_tokens = self.iou_token.weight.expand(bs, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings.reshape(bs, -1, c)), dim=1)
        pos_token = sparse_prompt_pe.reshape(bs, -1, c)
        pos_token=torch.cat((torch.zeros_like(pos_token[:,:1,:]), pos_token), dim=1)
        # Expand per-image data in batch direction to be per-mask
        src = image_embeddings
        pos_src = image_pe
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src,tokens,pos_token)
        iou_token_out = hs[:, 0, :]
        iou_pred = self.iou_prediction_head(iou_token_out)
        return iou_token_out, src, iou_pred
def build_promptEncoder_traj(cfg):
    return PromptEncoder_traj_rep_weight(
        embed_dim=cfg.PROMPT.HIDDEN_DIM,
        tracklen=cfg.PROMPT.TRACKLEN,
        log=cfg.PROMPT.PELOG,
        trunc=cfg.PROMPT.TRUNC,
        alpha=cfg.PROMPT.alpha
    )

def build_promptDecoder_traj(cfg):
    return PromptDecoder_traj_reg_rep_weight(
        transformer=TwoWayTransformer_rep_weight(
            depth=2,
            embedding_dim=cfg.PROMPT.HIDDEN_DIM,
            mlp_dim=1024,  # 2048
            num_heads=8,
        ),
        transformer_dim=cfg.PROMPT.HIDDEN_DIM,
    )

